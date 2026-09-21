#!/usr/bin/env python
"""Serve the crown labeller.

The labeller reads site chips with fetch(), which browsers block for file://
URLs. This serves the kit over http on localhost so it just works.

    python scripts/03_serve_labeller.py
    python scripts/03_serve_labeller.py --port 8900 --no-open

Nothing is uploaded anywhere. The server is local, read-only, and your labels
live in the browser until you press Export.
"""
from __future__ import annotations

import argparse
import http.server
import socketserver
import sys
import threading
import webbrowser
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _kit import KIT_ROOT  # noqa: E402


class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):  # keep the console quiet
        pass

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    chips = KIT_ROOT / "data" / "sites" / "chips" / "index.json"
    if not chips.exists():
        print("No site chips yet. Run these first:")
        print("  python scripts/01_select_reference_sites.py")
        print("  python scripts/02_extract_site_chips.py")
        return 1

    handler = partial(Handler, directory=str(KIT_ROOT))
    socketserver.TCPServer.allow_reuse_address = True
    port = args.port
    for attempt in range(10):
        try:
            httpd = socketserver.TCPServer(("127.0.0.1", port), handler)
            break
        except OSError:
            port += 1
    else:
        print(f"Could not bind a port near {args.port}")
        return 1

    url = f"http://127.0.0.1:{port}/labeller/index.html"
    print(f"\n  Crown labeller: {url}")
    print("  Press Ctrl+C to stop.\n")
    print("  Reminder: Export your labels before you close the tab.")
    print("  Save the file into data/labels/ and keep every export.\n")

    if not args.no_open:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
