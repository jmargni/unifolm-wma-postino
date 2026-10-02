"""
Fake policy server for local testing of `robot_client.py` without the model.

It answers POST /predict_action in the same format as
`scripts/evaluation/real_eval_server.py`, but no model is involved: the
returned actions are a slow sine wave around the first joint state it receives,
so the robot visibly moves while staying close to its starting pose.
"""

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

ACTION_CHUNK = 16  # the client hard-codes 16 predicted future steps


class MockPolicy:
    def __init__(self, amplitude: float, period_steps: int):
        self.amplitude = amplitude
        self.period_steps = period_steps
        self.center = None
        self.step = 0
        self.num_requests = 0
        self.lock = threading.Lock()

    def predict_action(self, payload: dict) -> list:
        states = np.asarray(payload["observation.state"], dtype=np.float32)
        images = np.asarray(payload["observation.images.top"], dtype=np.uint8)

        with self.lock:
            if self.center is None:
                self.center = states[-1].copy()
            steps = self.step + np.arange(ACTION_CHUNK)
            self.step += ACTION_CHUNK
            self.num_requests += 1
            num_requests = self.num_requests

        dim = self.center.shape[0]
        # Shift the phase per joint so the joints do not all move in sync.
        phase = 2 * np.pi * steps[:, None] / self.period_steps + np.arange(dim)[None, :]
        actions = self.center[None, :] + self.amplitude * (np.sin(phase) - np.sin(np.arange(dim))[None, :])

        print(
            f"[{num_requests}] instruction: {payload['language_instruction']!r} | "
            f"state {states.shape} | images {images.shape} "
            f"({'all black' if images.max() == 0 else 'has content'}) "
            f"-> actions {actions.shape}",
            flush=True,
        )
        return actions.tolist()


def make_handler(policy: MockPolicy):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            if self.path != "/predict_action":
                self.send_error(404)
                return
            try:
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                response = {"result": "ok", "action": policy.predict_action(payload), "desc": "success"}
            except Exception as e:
                response = {"result": "error", "desc": repr(e)}
            body = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--amplitude", type=float, default=0.15, help="Sine amplitude added to every joint (rad).")
    parser.add_argument("--period_steps", type=int, default=120, help="Sine period in control steps.")
    args = parser.parse_args()

    policy = MockPolicy(args.amplitude, args.period_steps)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(policy))
    print(f">>> Mock policy server is ready on http://{args.host}:{args.port} ...", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    print(">>> Mock policy server stops ...")


if __name__ == "__main__":
    main()
