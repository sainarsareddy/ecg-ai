"""
ecg_receiver.py  —  Run this ON your PC (alongside the dashboard)
==================================================================
Connects to the Raspberry Pi ECG streamer over TCP,
buffers the incoming 360Hz signal, and exposes it via a
simple local HTTP endpoint so the Streamlit dashboard can fetch it.

USAGE:
  1. Start ecg_streamer.py on your Raspberry Pi
  2. Run this on your PC:
       python ecg_receiver.py --pi-ip 192.168.1.100
  3. Open the Streamlit dashboard — select "Raspberry Pi Live"
     and click "Fetch from Receiver"

INSTALL on PC:
  pip install flask

The receiver runs on:  http://localhost:8765
"""

import argparse
import json
import socket
import threading
import time
from collections import deque

# ─── CONFIG ──────────────────────────────────────────────────────────────────
PI_PORT       = 9000     # Must match ecg_streamer.py PORT
RECEIVER_PORT = 8765     # Local port the dashboard fetches from
BUFFER_SIZE   = 3600     # Keep last 10 seconds of samples (360Hz × 10)
SAMPLE_RATE   = 360

# ─── SHARED STATE ────────────────────────────────────────────────────────────
buffer       = deque(maxlen=BUFFER_SIZE)
buffer_lock  = threading.Lock()
connected    = False
stream_info  = {"mode": "unknown", "sample_rate": SAMPLE_RATE}


# ─── PI READER THREAD ────────────────────────────────────────────────────────
def read_from_pi(pi_ip):
    """
    Connects to the Pi streamer and continuously reads samples into the buffer.
    Auto-reconnects if the connection drops.
    """
    global connected, stream_info

    while True:
        try:
            print(f"[INFO] Connecting to Pi at {pi_ip}:{PI_PORT} ...")
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10)
            sock.connect((pi_ip, PI_PORT))
            sock.settimeout(None)   # Blocking reads after connect
            print(f"[OK]   Connected to Pi")

            raw = b""
            first_line = True

            while True:
                chunk = sock.recv(512)
                if not chunk:
                    raise ConnectionError("Pi closed connection")
                raw += chunk

                while b"\n" in raw:
                    line, raw = raw.split(b"\n", 1)
                    try:
                        msg = json.loads(line.decode())
                    except Exception:
                        continue

                    if first_line and msg.get("type") == "handshake":
                        stream_info = {
                            "mode":        msg.get("mode", "hardware"),
                            "sample_rate": msg.get("sample_rate", SAMPLE_RATE),
                        }
                        connected = True
                        first_line = False
                        print(f"[OK]   Handshake received: {stream_info}")
                        continue

                    if "v" in msg:
                        with buffer_lock:
                            buffer.append(msg["v"])

        except Exception as e:
            connected = False
            print(f"[WARN] Pi connection lost: {e} — retrying in 3s ...")
            time.sleep(3)
        finally:
            try:
                sock.close()
            except Exception:
                pass


# ─── LOCAL HTTP SERVER ────────────────────────────────────────────────────────
def start_http_server():
    """
    Simple Flask server exposing two endpoints:
      GET /status          → connection status + buffer fill level
      GET /signal?n=1800   → last N samples as JSON list
    """
    from flask import Flask, jsonify, request
    app = Flask(__name__)

    @app.route("/status")
    def status():
        with buffer_lock:
            buf_len = len(buffer)
        return jsonify({
            "connected":   connected,
            "buffer_size": buf_len,
            "mode":        stream_info.get("mode", "unknown"),
            "sample_rate": stream_info.get("sample_rate", SAMPLE_RATE),
            "ready":       buf_len >= 1800,
        })

    @app.route("/signal")
    def signal():
        n = int(request.args.get("n", 1800))
        n = max(187, min(n, BUFFER_SIZE))   # Clamp to valid range
        with buffer_lock:
            samples = list(buffer)[-n:]
        if len(samples) < 187:
            return jsonify({"error": "Not enough samples yet", "have": len(samples)}), 503
        return jsonify({
            "ecg_signal":  samples,
            "n_samples":   len(samples),
            "sample_rate": stream_info.get("sample_rate", SAMPLE_RATE),
            "mode":        stream_info.get("mode", "hardware"),
        })

    print(f"[OK]   Local receiver API running on http://localhost:{RECEIVER_PORT}")
    print(f"[OK]   Dashboard fetches signal from: http://localhost:{RECEIVER_PORT}/signal")
    app.run(host="0.0.0.0", port=RECEIVER_PORT, debug=False, use_reloader=False)


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ECG Receiver — connects Pi to dashboard")
    parser.add_argument("--pi-ip",   required=True,          help="Raspberry Pi IP address")
    parser.add_argument("--pi-port", type=int, default=9000,  help="Pi streamer port (default 9000)")
    parser.add_argument("--port",    type=int, default=8765,  help="Local receiver port (default 8765)")
    args = parser.parse_args()

    PI_PORT       = args.pi_port
    RECEIVER_PORT = args.port

    print("=" * 52)
    print("  ECG Receiver")
    print(f"  Pi address    :  {args.pi_ip}:{PI_PORT}")
    print(f"  Local API     :  http://localhost:{RECEIVER_PORT}")
    print(f"  Buffer        :  {BUFFER_SIZE} samples ({BUFFER_SIZE//SAMPLE_RATE}s)")
    print("=" * 52)

    # Start Pi reader in background
    threading.Thread(
        target=read_from_pi,
        args=(args.pi_ip,),
        daemon=True
    ).start()

    # Start local HTTP server (blocks)
    start_http_server()