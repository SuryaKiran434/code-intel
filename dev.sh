#!/bin/bash
# dev.sh — Code Intel service manager
# Run from ~/Desktop/code-intel/

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ATTU_CONTAINER="code-intel-attu"
ATTU_PORT=8000
MILVUS_PORT=19530
WEBUI_PORT=7860
WEBUI_PID_FILE="$PROJECT_DIR/.webui.pid"

# ── Colors ────────────────────────────────────────────────────────────────────
G="\033[0;32m"  # green
C="\033[0;36m"  # cyan
Y="\033[0;33m"  # yellow
R="\033[0;31m"  # red
B="\033[1m"     # bold
N="\033[0m"     # reset

# ── Helpers ───────────────────────────────────────────────────────────────────
info()    { echo -e "${C}▶ $*${N}"; }
success() { echo -e "${G}✓ $*${N}"; }
warn()    { echo -e "${Y}⚠ $*${N}"; }
error()   { echo -e "${R}✗ $*${N}"; exit 1; }

wait_for_port() {
  local host=$1 port=$2 label=$3 retries=20
  info "Waiting for $label..."
  for i in $(seq 1 $retries); do
    if nc -z "$host" "$port" 2>/dev/null; then
      success "$label is ready"
      return 0
    fi
    sleep 2
  done
  error "$label did not become ready in time"
}

milvus_running() {
  docker compose -f "$PROJECT_DIR/docker-compose.yml" ps --status running 2>/dev/null \
    | grep -q "milvus"
}

webui_running() {
  [ -f "$WEBUI_PID_FILE" ] && kill -0 "$(cat "$WEBUI_PID_FILE")" 2>/dev/null
}

attu_running() {
  docker ps --filter "name=$ATTU_CONTAINER" --filter "status=running" -q 2>/dev/null \
    | grep -q .
}

# ── Preflight helpers ─────────────────────────────────────────────────────────
ensure_docker() {
  if docker info > /dev/null 2>&1; then
    success "Docker is already running"
    return 0
  fi
  info "Docker is not running — launching Docker Desktop..."
  open -a Docker
  local retries=30  # 30 × 2s = 60s max wait
  for i in $(seq 1 $retries); do
    if docker info > /dev/null 2>&1; then
      echo ""
      success "Docker Desktop is ready"
      return 0
    fi
    printf "\r  ${C}Waiting for Docker daemon... (%d/%d)${N}" "$i" "$retries"
    sleep 2
  done
  echo ""
  error "Docker did not start within 60s — please launch Docker Desktop manually and retry"
}

ensure_venv() {
  if [ ! -f "$PROJECT_DIR/.venv/bin/activate" ]; then
    info "No .venv found — creating virtual environment..."
    python3 -m venv "$PROJECT_DIR/.venv"
    # shellcheck source=/dev/null
    source "$PROJECT_DIR/.venv/bin/activate"
    info "Installing dependencies (this takes a minute the first time)..."
    pip install -r "$PROJECT_DIR/requirements.txt" --quiet
    success "Virtual environment created and ready"
  else
    # shellcheck source=/dev/null
    source "$PROJECT_DIR/.venv/bin/activate"
    success "Virtual environment activated"
  fi
}

# ── Actions ───────────────────────────────────────────────────────────────────
start_webui() {
  if webui_running; then
    warn "Web UI is already running → http://localhost:$WEBUI_PORT"
    return
  fi
  if [ ! -f "$PROJECT_DIR/.venv/bin/activate" ]; then
    error "No .venv found in $PROJECT_DIR — run: python3 -m venv .venv && pip install -r requirements.txt"
  fi
  info "Starting Web UI..."
  source "$PROJECT_DIR/.venv/bin/activate"
  cd "$PROJECT_DIR"
  nohup uvicorn app:app --port "$WEBUI_PORT" --host 127.0.0.1 \
    > "$PROJECT_DIR/.webui.log" 2>&1 &
  echo $! > "$WEBUI_PID_FILE"
  wait_for_port localhost $WEBUI_PORT "Web UI"
  success "Web UI ready → http://localhost:$WEBUI_PORT"
}

stop_webui() {
  if webui_running; then
    info "Stopping Web UI..."
    kill "$(cat "$WEBUI_PID_FILE")" 2>/dev/null
    rm -f "$WEBUI_PID_FILE"
    success "Web UI stopped"
  else
    warn "Web UI is not running"
  fi
}

start_milvus() {
  if milvus_running; then
    warn "Milvus is already running"
  else
    info "Starting Milvus (etcd + minio + milvus)..."
    if ! docker compose -f "$PROJECT_DIR/docker-compose.yml" up -d 2>/dev/null; then
      warn "Start failed — stale containers or network detected, cleaning up and retrying..."
      docker compose -f "$PROJECT_DIR/docker-compose.yml" down --remove-orphans 2>/dev/null || true
      docker network prune -f > /dev/null 2>&1 || true
      docker compose -f "$PROJECT_DIR/docker-compose.yml" up -d
    fi
    wait_for_port localhost $MILVUS_PORT "Milvus"
  fi
}

stop_milvus() {
  info "Stopping Milvus..."
  docker compose -f "$PROJECT_DIR/docker-compose.yml" down
  success "Milvus stopped"
}

start_attu() {
  if attu_running; then
    warn "Attu is already running → http://localhost:$ATTU_PORT"
    return
  fi
  info "Starting Attu..."
  docker run -d \
    --name "$ATTU_CONTAINER" \
    --rm \
    -p "$ATTU_PORT:3000" \
    -e MILVUS_URL="host.docker.internal:$MILVUS_PORT" \
    zilliz/attu:latest \
    > /dev/null
  wait_for_port localhost $ATTU_PORT "Attu"
  success "Attu ready → http://localhost:$ATTU_PORT"
}

stop_attu() {
  if attu_running; then
    info "Stopping Attu..."
    docker stop "$ATTU_CONTAINER" > /dev/null
    success "Attu stopped"
  else
    warn "Attu is not running"
  fi
}

start_all() {
  ensure_docker
  ensure_venv
  start_milvus
  start_attu
  start_webui
  echo ""
  success "All services running"
  echo -e "  ${B}Milvus${N}   localhost:$MILVUS_PORT"
  echo -e "  ${B}Attu${N}     http://localhost:$ATTU_PORT"
  echo -e "  ${B}Web UI${N}   http://localhost:$WEBUI_PORT"
}

stop_all() {
  stop_webui
  stop_attu
  stop_milvus
  deactivate 2>/dev/null || true
  info "Quitting Docker Desktop..."
  if osascript -e 'quit app "Docker"' > /dev/null 2>&1; then
    success "Docker Desktop quit"
  else
    warn "Docker Desktop was not running"
  fi
  success "All services stopped"
}

show_status() {
  echo ""
  echo -e "${B}── Service Status ───────────────────────────────${N}"
  if milvus_running; then
    echo -e "  Milvus    ${G}● running${N}   localhost:$MILVUS_PORT"
  else
    echo -e "  Milvus    ${R}○ stopped${N}"
  fi
  if attu_running; then
    echo -e "  Attu      ${G}● running${N}   http://localhost:$ATTU_PORT"
  else
    echo -e "  Attu      ${R}○ stopped${N}"
  fi
  if webui_running; then
    echo -e "  Web UI    ${G}● running${N}   http://localhost:$WEBUI_PORT"
  else
    echo -e "  Web UI    ${R}○ stopped${N}"
  fi
  echo ""
}

show_menu() {
  echo ""
  echo -e "${B}⚡ Code Intel — Service Manager${N}"
  echo -e "────────────────────────────────"
  echo -e "  ${C}1${N}  Start all          (Milvus + Attu + Web UI)"
  echo -e "  ${C}2${N}  Stop all"
  echo -e "  ${C}3${N}  Start Milvus only"
  echo -e "  ${C}4${N}  Stop Milvus only"
  echo -e "  ${C}5${N}  Start Attu only"
  echo -e "  ${C}6${N}  Stop Attu only"
  echo -e "  ${C}7${N}  Start Web UI only"
  echo -e "  ${C}8${N}  Stop Web UI only"
  echo -e "  ${C}9${N}  Status"
  echo -e "  ${C}q${N}  Quit"
  echo ""
  read -rp "Choose an option: " choice

  case "$choice" in
    1) start_all ;;
    2) stop_all ;;
    3) start_milvus ;;
    4) stop_milvus ;;
    5) start_attu ;;
    6) stop_attu ;;
    7) start_webui ;;
    8) stop_webui ;;
    9) show_status ;;
    q|Q) echo "Bye!"; exit 0 ;;
    *) warn "Invalid option: $choice" ;;
  esac
}

# ── Entry point ───────────────────────────────────────────────────────────────
# Supports both interactive menu and direct flags:
#   ./dev.sh            → interactive menu
#   ./dev.sh start      → start all
#   ./dev.sh stop       → stop all
#   ./dev.sh status     → show status

case "${1:-}" in
  start)  start_all ;;
  stop)   stop_all ;;
  status) show_status ;;
  "")     show_menu ;;
  *)      error "Unknown command '$1'. Use: start | stop | status" ;;
esac
