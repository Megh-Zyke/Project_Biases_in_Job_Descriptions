#!/usr/bin/env bash
# Start the persona-bias Potato server in a detached screen session on curry.
#   bash run_server.sh [port]      (default 8791)
# Then from your laptop:  ssh -L PORT:localhost:PORT meghss@curry  ->  http://localhost:PORT
set -euo pipefail
PORT="${1:-8791}"
TASK=/home/meghss/Project_Biases_in_Job_Descriptions/annotation_task
TPL=/opt/anaconda/lib/python3.12/site-packages/potato/templates

# Potato 2.3.0 writes generated pages into its install dir; it crashes if it can't.
if ! { [[ -d "$TPL/generated" && -w "$TPL/generated" ]] || [[ -w "$TPL" ]]; }; then
  echo "ERROR: $TPL/generated is missing or not writable. Potato will crash on start."
  echo "Ask a cluster admin to create it writable by the lab, or to upgrade potato-annotation."
  exit 1
fi
if ss -ltn | grep -q ":$PORT "; then
  echo "ERROR: port $PORT is already in use. Pass another: bash run_server.sh 8792"; exit 1
fi
if screen -ls | grep -q "\.potato\b"; then
  echo "A 'potato' screen session already exists. Attach with: screen -r potato"; exit 1
fi

cd "$TASK"
screen -dmS potato bash -c "potato start config.yaml -p $PORT 2>&1 | tee -a server.log"
echo "Starting on port $PORT (takes ~80s to load). Waiting for it to answer..."
for _ in $(seq 1 60); do
  if [[ "$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:$PORT/")" == 200 ]]; then
    echo "Up: http://localhost:$PORT  (tunnel: ssh -L $PORT:localhost:$PORT meghss@curry)"
    echo "Logs: screen -r potato   (detach: Ctrl-a d; stop: Ctrl-c inside it)"
    exit 0
  fi
  screen -ls | grep -q "\.potato\b" || { echo "Server exited. Last log lines:"; tail -15 server.log; exit 1; }
  sleep 5
done
echo "Still not answering after 5 minutes. Check: screen -r potato  or  tail server.log"; exit 1
