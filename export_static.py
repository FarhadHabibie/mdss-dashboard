#!/usr/bin/env python3
"""Export dashboard as static index.html for GitHub Pages."""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dashboard import app  # noqa: E402

payload = app.build_payload("data")
raw = json.dumps(payload, default=str)
safe = raw.replace("</", "<\\/")
html = app.HTML_PAGE.replace("__PAYLOAD__", safe, 1)

out = os.path.join("docs", "index.html")
os.makedirs("docs", exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    f.write(html)
print(f"written: {out} ({len(html)} bytes)")