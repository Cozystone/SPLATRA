#!/usr/bin/env bash
# Build the static Vercel bundle (web/) from the viewer + export sample cartridges.
# Sample export needs a running API (defaults to the local GPU server on :8000).
set -euo pipefail
cd "$(dirname "$0")/.."
API="${API:-http://127.0.0.1:8000}"

cp viewer/studio.html web/index.html
mkdir -p web/samples

python - "$API" <<'PY'
import sys, json, time, urllib.request
B = sys.argv[1]
def chat(m, t=240):
    r = urllib.request.Request(B+"/v1/chat", data=json.dumps({"message": m}).encode(),
                               headers={"Content-Type": "application/json"})
    urllib.request.urlopen(r, timeout=t).read()
def save(fn):
    b = urllib.request.urlopen(B+"/v1/cartridge?_=%d" % time.time(), timeout=30).read()
    open("web/samples/"+fn, "wb").write(b); print("wrote web/samples/"+fn, len(b), "bytes")
chat("show a knowledge graph with 28 nodes"); save("graph.bin")
chat("a blue torus"); save("torus.bin")
chat("a cute pikachu"); save("pikachu.bin")
PY
echo "web/ rebuilt. Deploy:  cd web && vercel deploy --prod"
