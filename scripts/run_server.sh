#!/usr/bin/env bash
# ============================================================
# pisto-gpt-64m — Run the Go web server + Cloudflare quick tunnel
# Exposes the local UI publicly via a free trycloudflare.com URL.
# Useful for Google Colab, where the VM has no public IP.
# ============================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$REPO/ui/pisto-server"
LOG="$REPO/.server.log"

# Port from config/web.json (fallback 8080)
PORT="$(python3 -c 'import json; print(json.load(open("config/web.json")).get("port", 8080))' 2>/dev/null || echo 8080)"

echo "==> [1/4] Building the Go server..."
if command -v go >/dev/null 2>&1; then
    (cd "$REPO" && go build -o ui/pisto-server ./ui/)
elif [ ! -x "$BIN" ]; then
    echo "ERROR: 'go' not found and $BIN is missing."
    echo "  Install Go (https://go.dev/dl/) or copy a prebuilt pisto-server binary here."
    exit 1
fi

echo "==> [2/4] Starting server on port $PORT (log: $LOG)..."
(cd "$REPO" && nohup "$BIN" >"$LOG" 2>&1 &)
SERVER_PID=$!
trap 'echo; echo "Stopping server (pid $SERVER_PID)..."; kill "$SERVER_PID" 2>/dev/null || true' EXIT

# Wait until the server answers
for i in $(seq 1 10); do
    if curl -s -o /dev/null "http://localhost:$PORT/"; then
        break
    fi
    sleep 1
done
curl -s -o /dev/null "http://localhost:$PORT/" || { echo "Server failed to start — check $LOG"; exit 1; }
echo "    Server is up: http://localhost:$PORT"

echo "==> [3/4] Ensuring cloudflared is installed..."
if ! command -v cloudflared >/dev/null 2>&1; then
    echo "    Installing cloudflared (linux-amd64)..."
    TMP="$(mktemp -d)"
    if curl -fsSL "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64" -o "$TMP/cloudflared"; then
        sudo install -m 0755 "$TMP/cloudflared" /usr/local/bin/cloudflared 2>/dev/null \
            || install -m 0755 "$TMP/cloudflared" "$HOME/.local/bin/cloudflared"
        rm -rf "$TMP"
        export PATH="$HOME/.local/bin:$PATH"
    else
        echo "ERROR: could not download cloudflared."
        echo "  Install manually: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/"
        exit 1
    fi
    command -v cloudflared >/dev/null 2>&1 || { echo "ERROR: cloudflared still not found on PATH."; exit 1; }
fi
echo "    cloudflared: $(command -v cloudflared)"

echo "==> [4/4] Opening Cloudflare tunnel -> http://localhost:$PORT"
echo "    (Ctrl+C to stop both server and tunnel)"
cloudflared tunnel --url "http://localhost:$PORT" --no-autoupdate
