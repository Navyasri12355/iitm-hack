#!/usr/bin/env python3
"""
Simple HTTP server to serve the static frontend for testing.
"""

import http.server
import socketserver
import os
from pathlib import Path

PORT = 8080
DIRECTORY = "static"

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

if __name__ == "__main__":
    # Change to the project root directory
    os.chdir(Path(__file__).parent)
    
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Serving at http://localhost:{PORT}")
        print(f"Serving files from: {os.path.abspath(DIRECTORY)}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")