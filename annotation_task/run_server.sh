#!/usr/bin/env bash
# Start the persona-bias Potato server in a detached screen session on curry.
#   bash run_server.sh [port]      (default 8791)
# Then from your laptop:  ssh -L PORT:localhost:PORT meghss@curry  ->  http://localhost:PORT
set -euo pipefail
USER_NAME="$(whoami)"
HOST_NAME="$(hostname -s 2>/dev/null || hostname)"
PORT="${1:-8791}"
TASK="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export POTATO_GENERATED_TEMPLATES_DIR="/tmp/potato-templates-${USER_NAME}"
mkdir -p "$POTATO_GENERATED_TEMPLATES_DIR"

if ss -ltn | grep -q ":$PORT "; then
  echo "ERROR: port $PORT is already in use. Pass another: bash run_server.sh 8792"; exit 1
fi
if screen -ls 2>&1 | grep -q "\.potato\b"; then
  echo "A 'potato' screen session already exists. Attach with: screen -r potato"; exit 1
fi

cd "$TASK"
screen -dmS potato bash -c "export POTATO_GENERATED_TEMPLATES_DIR='$POTATO_GENERATED_TEMPLATES_DIR'; potato start config.yaml -p $PORT 2>&1 | tee -a server.log"
echo "Starting on port $PORT (takes ~80s to load). Waiting for it to answer..."
for _ in $(seq 1 60); do
  if [[ "$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:$PORT/")" == 200 ]]; then
    echo "Up: http://localhost:$PORT  (tunnel: ssh -L $PORT:localhost:$PORT ${USER_NAME}@${HOST_NAME})"
    echo "Logs: screen -r potato   (detach: Ctrl-a d; stop: Ctrl-c inside it)"
    exit 0
  fi
  screen -ls 2>&1 | grep -q "\.potato\b" || { echo "Server exited. Last log lines:"; tail -15 server.log; exit 1; }
  sleep 5
done
echo "Still not answering after 5 minutes. Check: screen -r potato  or  tail server.log"; exit 1
