import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Reuse the real handler (and its logging + model init).
from handler import handler  # type: ignore


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0

        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            job = json.loads(raw.decode("utf-8") or "{}")
        except Exception:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "invalid_json"}).encode("utf-8"))
            return

        try:
            resp = handler(job)
            body = json.dumps(resp).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as exc:
            body = json.dumps({"error": "handler_exception", "detail": str(exc)}).encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    # Quiet default request logging; rely on our app logger instead.
    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return


def main() -> None:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8080"))
    httpd = ThreadingHTTPServer((host, port), _Handler)
    print(f"Local server listening on http://{host}:{port} (POST /)")
    httpd.serve_forever()


if __name__ == "__main__":
    main()

