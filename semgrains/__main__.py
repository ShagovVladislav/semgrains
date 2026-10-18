import argparse
from pathlib import Path
import socket
import sys


def find_port(first):
    for port in range(first, min(first + 20, 65536)):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(("127.0.0.1", port))
            return port
        except OSError:
            continue
    raise RuntimeError("Нет свободного порта. Укажите другой через run.cmd -Port <номер>")


def main():
    parser = argparse.ArgumentParser(prog="semgrains")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("Порт должен быть между 1024 и 65535")
    try:
        port = find_port(args.port)
    except RuntimeError as exc:
        parser.exit(1, str(exc) + "\n")
    from streamlit.web import cli
    app = Path(__file__).resolve().parents[1] / "app.py"
    sys.argv = ["streamlit", "run", str(app), "--server.address", "127.0.0.1",
                "--server.port", str(port), "--browser.gatherUsageStats", "false",
                "--server.headless", "true" if args.no_browser else "false"]
    return cli.main()


if __name__ == "__main__":
    sys.exit(main())
