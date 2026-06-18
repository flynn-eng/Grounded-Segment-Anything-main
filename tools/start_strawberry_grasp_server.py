#!/usr/bin/env python3
"""Start the strawberry grasp API on an available port."""

from __future__ import annotations

import argparse
import socket
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start strawberry grasp server.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--preferred-port", type=int, default=8765)
    parser.add_argument("--max-tries", type=int, default=50)
    parser.add_argument("--jobs-dir", default=str(PROJECT_ROOT / "strawberry_server_jobs"))
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def port_is_available(host: str, port: int) -> bool:
    probe_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((probe_host, port)) != 0


def find_port(host: str, preferred_port: int, max_tries: int) -> int:
    for port in range(preferred_port, preferred_port + max_tries):
        if port_is_available(host, port):
            return port
    raise RuntimeError(
        f"No free port found from {preferred_port} to {preferred_port + max_tries - 1}"
    )


def main() -> None:
    args = parse_args()
    port = find_port(args.host, args.preferred_port, args.max_tries)
    print(f"Starting strawberry grasp server on http://{args.host}:{port}", flush=True)
    print(f"Jobs dir: {args.jobs_dir}", flush=True)

    sys.path.insert(0, str(PROJECT_ROOT))
    import uvicorn
    from server_strawberry_grasp import build_app

    app = build_app(
        argparse.Namespace(
            jobs_dir=args.jobs_dir,
            device=args.device,
            box_threshold=0.3,
            text_threshold=0.25,
            text_prompt="strawberry",
            config=str(PROJECT_ROOT / "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"),
            grounded_checkpoint=str(PROJECT_ROOT / "groundingdino_swint_ogc.pth"),
            sam_checkpoint=str(PROJECT_ROOT / "sam_vit_h_4b8939.pth"),
        )
    )
    uvicorn.run(app, host=args.host, port=port)


if __name__ == "__main__":
    main()
