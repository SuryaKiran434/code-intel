"""
core/llm.py

Handles GPT-4o query construction and response generation.

Key responsibilities:
  - Token-budget-aware context assembly
  - Automatic fallback to summary chunks when full chunks are too large
  - Clear source attribution in every answer
  - Structured response with answer + metadata
"""

import tiktoken
from openai import OpenAI

from config import (
    OPENAI_API_KEY,
    LLM_MODEL,
    LLM_MAX_TOKENS,
    LLM_CONTEXT_TOKEN_LIMIT,
)

client  = OpenAI(api_key=OPENAI_API_KEY)
encoder = tiktoken.encoding_for_model("gpt-4o")


# ── Token utilities ────────────────────────────────────────────────────────────

def count_tokens(text: str) -> int:
    """Count the number of GPT-4o tokens in a string."""
    return len(encoder.encode(text))


# ── Context assembly ───────────────────────────────────────────────────────────

def build_context(
    chunks: list[dict],
    context_limit: int = LLM_CONTEXT_TOKEN_LIMIT,
) -> tuple[str, list[dict]]:
    """
    Assemble retrieved chunks into a context block that fits within
    context_limit tokens (defaults to LLM_CONTEXT_TOKEN_LIMIT).

    Strategy for each chunk:
        1. Try to include the full content
        2. If full content won't fit → truncate to first 30 lines + notice
        3. If even truncated version won't fit → stop adding chunks

    Returns:
        context_text  — formatted string to send to GPT-4o
        used_chunks   — subset of chunks that made it into context
                        (for source attribution in the response)
    """
    context_parts = []
    used_chunks   = []
    total_tokens  = 0

    # Separate direct vector-search hits ([C]) from graph-expanded deps ([G])
    direct_chunks = [c for c in chunks if c.get("retrieval_source") != "graph"]
    graph_chunks  = [c for c in chunks if c.get("retrieval_source") == "graph"]
    ordered = direct_chunks + graph_chunks

    # Counters for each label series
    c_idx = 0
    g_idx = 0

    for chunk in ordered:
        is_graph = chunk.get("retrieval_source") == "graph"
        if is_graph:
            g_idx += 1
            label = f"G{g_idx}"
        else:
            c_idx += 1
            label = f"C{c_idx}"

        rel_path = chunk["file_path"]
        source_note = " [graph dependency]" if is_graph else ""
        header = (
            f"### [{label}]{source_note} {chunk['symbol_name']} "
            f"({'lines ' + str(chunk['start_line']) + '–' + str(chunk['end_line'])})\n"
            f"File: {rel_path}\n"
            f"Language: {chunk['language']} | "
            f"Type: {chunk['chunk_type']} | "
            f"Relevance: {chunk['score']}"
        )

        full_block  = f"{header}\n\n```{chunk['language']}\n{chunk['content']}\n```"

        remaining = context_limit - total_tokens
        block_tokens = count_tokens(full_block)

        if block_tokens <= remaining:
            context_parts.append(full_block)
            used_chunks.append({**chunk, "_label": label})
            total_tokens += block_tokens

        else:
            # Try a truncated version — first 30 lines + notice.
            content_lines = chunk["content"].splitlines()
            truncated     = "\n".join(content_lines[:30])
            truncated += (
                f"\n# ... [truncated — {len(content_lines) - 30} more lines]"
                f"\n# See full source: {rel_path}"
            )
            trunc_block  = f"{header}\n\n```{chunk['language']}\n{truncated}\n```"
            trunc_tokens = count_tokens(trunc_block)

            if trunc_tokens <= remaining:
                context_parts.append(trunc_block)
                used_chunks.append({**chunk, "_truncated": True, "_label": label})
                total_tokens += trunc_tokens
            else:
                # Even truncated doesn't fit — stop here
                break

    context_text = "\n\n---\n\n".join(context_parts)
    return context_text, used_chunks


# ── System prompt ──────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are an expert code assistant embedded in a developer's \
code intelligence tool. Your job is to help developers understand their codebase \
by answering questions about specific code, its purpose, and how it works.

Code chunks in the context are labelled in two series:
  [C1], [C2], ... — directly retrieved chunks (highest relevance to your question)
  [G1], [G2], ... — graph-expanded dependency chunks (functions/classes called by \
the [C] chunks, included for cross-file context)

Rules you must follow:
1. Answer using ONLY the code context provided — do not hallucinate code or behaviour \
that isn't shown.
2. Cite chunks inline using their labels, e.g. "The router is registered in [C2]." \
and "It delegates to the helper defined in [G1]." Use these labels wherever you draw \
a conclusion from a specific chunk.
3. When a [G] chunk is relevant, explain why it is a dependency of the [C] chunk that \
called it.
4. If a chunk is marked as truncated, say so and tell the user to check the source file \
for the full implementation.
5. If the provided context does not contain enough information to answer the question \
confidently, say so explicitly rather than guessing.
6. Keep answers concise but complete. Use code blocks when showing examples.
7. When explaining how something works, walk through the logic step by step."""


# ── Main query function ────────────────────────────────────────────────────────

def ask_stream(
    question:      str,
    chunks:        list[dict],
    history:       list[dict] | None = None,
    context_limit: int = LLM_CONTEXT_TOKEN_LIMIT,
):
    """
    Stream the LLM answer token-by-token using the OpenAI streaming API.

    Yields dicts in two shapes:
        {"type": "token", "text": "..."}          — one per streamed chunk
        {"type": "done",  "sources": [...],
         "tokens": N, "answer": "full text"}      — once, after the last token

    Callers accumulate "token" payloads to show progressive output, then use
    the "done" payload for final rendering, source attribution, and telemetry.
    """
    if not chunks:
        msg = ("No relevant code was found for your question. "
               "Try rephrasing or check that the repository is indexed.")
        yield {"type": "token",  "text": msg}
        yield {"type": "done",   "sources": [], "tokens": 0, "answer": msg}
        return

    context_text, used_chunks = build_context(chunks, context_limit=context_limit)

    user_message = (
        f"Here is the relevant code context from the codebase:\n\n"
        f"{context_text}\n\n"
        f"---\n\n"
        f"Question: {question}"
    )

    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    stream = client.chat.completions.create(
        model=LLM_MODEL,
        max_tokens=LLM_MAX_TOKENS,
        temperature=0.1,
        messages=messages,
        stream=True,
        stream_options={"include_usage": True},
    )

    full_text  = ""
    tokens_used = 0

    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            delta = chunk.choices[0].delta.content
            full_text += delta
            yield {"type": "token", "text": delta}
        if chunk.usage:
            tokens_used = chunk.usage.total_tokens

    # Fallback token count if stream_options usage wasn't returned
    if not tokens_used:
        tokens_used = count_tokens(user_message) + count_tokens(full_text)

    sources = [
        {
            "label":            c.get("_label", f"C{idx + 1}"),
            "file":             c["file_path"],
            "symbol":           c["symbol_name"],
            "lines":            f"{c['start_line']}–{c['end_line']}",
            "score":            c["score"],
            "chunk_type":       c["chunk_type"],
            "retrieval_source": c.get("retrieval_source", "direct"),
            "truncated":        c.get("_truncated", False),
        }
        for idx, c in enumerate(used_chunks)
    ]

    yield {"type": "done", "sources": sources, "tokens": tokens_used, "answer": full_text}


def ask(
    question:      str,
    chunks:        list[dict],
    history:       list[dict] | None = None,
    context_limit: int = LLM_CONTEXT_TOKEN_LIMIT,
) -> dict:
    """
    Send a question + retrieved code context to GPT-4o and return the answer.

    Args:
        question  — the user's natural language question
        chunks    — retrieved chunks from core/retriever.py
        history   — optional prior conversation turns as OpenAI message dicts
                    [{role: "user"|"assistant", content: "..."}]
                    Prepended between the system prompt and the current turn.

    Returns a dict with:
        answer         — GPT-4o's response string
        tokens_used    — total tokens consumed (prompt + completion)
        context_chunks — number of chunks included in context
        truncated      — True if any chunks were truncated to fit
        sources        — list of source metadata dicts for UI display
    """
    if not chunks:
        return {
            "answer":         "No relevant code was found for your question. "
                              "Try rephrasing or check that the repository is indexed.",
            "tokens_used":    0,
            "context_chunks": 0,
            "truncated":      False,
            "sources":        [],
        }

    context_text, used_chunks = build_context(chunks, context_limit=context_limit)
    any_truncated = any(c.get("_truncated") for c in used_chunks)

    user_message = (
        f"Here is the relevant code context from the codebase:\n\n"
        f"{context_text}\n\n"
        f"---\n\n"
        f"Question: {question}"
    )

    # Build messages: system → prior history turns → current user message
    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model=LLM_MODEL,
        max_tokens=LLM_MAX_TOKENS,
        temperature=0.1,
        messages=messages,
    )

    answer = response.choices[0].message.content

    # Build clean source list for CLI and web UI display
    sources = [
        {
            "label":            c.get("_label", f"C{idx + 1}"),
            "file":             c["file_path"],
            "symbol":           c["symbol_name"],
            "lines":            f"{c['start_line']}–{c['end_line']}",
            "score":            c["score"],
            "chunk_type":       c["chunk_type"],
            "retrieval_source": c.get("retrieval_source", "direct"),
            "truncated":        c.get("_truncated", False),
        }
        for idx, c in enumerate(used_chunks)
    ]

    return {
        "answer":         answer,
        "tokens_used":    response.usage.total_tokens,
        "context_chunks": len(used_chunks),
        "truncated":      any_truncated,
        "sources":        sources,
    }