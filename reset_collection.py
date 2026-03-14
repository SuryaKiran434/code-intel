#!/usr/bin/env python3
"""
reset_collection.py

Drops and recreates the Milvus collection with the correct schema
for voyage-code-3 (dim=1024, COSINE/IVF_FLAT).

Run this ONCE when switching embedding models or dimension.

Usage:
    python reset_collection.py
"""

from dotenv import load_dotenv
load_dotenv()

from pymilvus import connections, utility, Collection
from config import MILVUS_HOST, MILVUS_PORT, COLLECTION_NAME

def main():
    print(f"Connecting to Milvus at {MILVUS_HOST}:{MILVUS_PORT}...")
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)

    if utility.has_collection(COLLECTION_NAME):
        confirm = input(
            f"\n⚠️  Collection '{COLLECTION_NAME}' exists and will be DELETED.\n"
            f"   All existing embeddings will be lost.\n"
            f"   Type 'yes' to confirm: "
        )
        if confirm.strip().lower() != "yes":
            print("Cancelled.")
            return

        utility.drop_collection(COLLECTION_NAME)
        print(f"✓ Dropped collection '{COLLECTION_NAME}'.")
    else:
        print(f"Collection '{COLLECTION_NAME}' does not exist — nothing to drop.")

    # Recreate via vector_store (which has the correct schema)
    from core.vector_store import get_or_create_collection
    collection = get_or_create_collection()
    print(f"✓ Collection '{COLLECTION_NAME}' recreated with dim=1024 (voyage-code-3).")
    print("\nPartitions are created automatically per repo on first 'add'.")
    print("Re-index all repos: python cli.py add <repo_name> --force")

if __name__ == "__main__":
    main()