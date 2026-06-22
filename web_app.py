from flask import Flask, render_template, request, Response
from werkzeug.utils import secure_filename
import cv2
import time
import os
import numpy as np
from collections import deque
from queue import Queue, Empty
import json

from detection import ViolenceDetector
from alert import send_sms_alert

app = Flask(__name__)

# Defaults aligned with main.py
DEFAULTS = {
    "prob_threshold": 0.6,
    "window_size": 20,
    "window_ratio": 0.5,
    "alert_cooldown": 30,
    "sequence_length": 20,
    "frame_skip": 1,
    "ema_alpha": 0.2,
    "input_w": 224,
    "input_h": 224,
}

EVENTS: Queue = Queue(maxsize=1000)

def parse_float(arg, default):
    try:
        return float(arg)
    except Exception:
        return default


def parse_int(arg, default):
    try:
        return int(arg)
    except Exception:
        return default


def push_event(event: dict):
    try:
        payload = json.dumps(event)
        EVENTS.put_nowait(payload)
    except Exception:
        pass


def sse_stream():
    while True:
        try:
            payload = EVENTS.get(timeout=15)
            yield f"data: {payload}\n\n"
        except Empty:
            yield "data: {\"type\": \"ping\"}\n\n"


def draw_overlay(display, prob, smoothed, prob_threshold, pred_window, ratio=None):
    prob_display = int(max(0, min(100, smoothed * 100)))
    color = (0, 255, 0) if smoothed < prob_threshold else (0, 0, 255)
    cv2.rectangle(display, (10, 80), (10 + prob_display, 100), color, -1)
    cv2.putText(display, f"Prob: {prob:.2f} | EMA: {smoothed:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    if ratio is not None:
        cv2.putText(display, f"Win: {sum(pred_window)}/{len(pred_window)} ({ratio:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def _sanitize_source_path(p: str) -> str:
    if not p:
        return p
    p = p.strip()
    if len(p) >= 2 and ((p[0] == '"' and p[-1] == '"') or (p[0] == "'" and p[-1] == "'")):
        p = p[1:-1]
    # Normalize slashes is optional; OpenCV accepts backslashes. Keep as-is to avoid breaking UNC.
    return p


def gen_stream(mode: str, path: str, params: dict):
    """Yield MJPEG frames with detection overlay."""
    # Open source
    if mode == "webcam":
        source = 0
        # Try DirectShow first on Windows for more reliable webcam access
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(0)
    else:
        source = _sanitize_source_path(path)
        if not source:
            blank = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "Empty file path", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            ret, jpg = cv2.imencode('.jpg', blank)
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")
            push_event({"type": "stream_error", "message": "Empty file path"})
            return
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        # Generate a single frame with error text
        blank = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(blank, f"Cannot open source: {source if mode!='webcam' else 'webcam'}", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        ret, jpg = cv2.imencode('.jpg', blank)
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")
        push_event({"type": "stream_error", "message": f"Cannot open source: {source if mode!='webcam' else 'webcam'}"})
        return

    detector = ViolenceDetector(
        model_path=os.path.join("models", "violence_detection_model.h5"),
        input_size=(params["input_w"], params["input_h"]),
        sequence_length=params["sequence_length"],
    )

    frame_skip = params["frame_skip"]
    prob_threshold = params["prob_threshold"]
    window_size = params["window_size"]
    window_ratio = params["window_ratio"]
    ema_alpha = params["ema_alpha"]
    alert_cooldown = params["alert_cooldown"]

    pred_window = deque(maxlen=window_size)
    ema_prob = None
    last_alert_time = 0.0

    frame_number = 0
    try:
        push_event({"type": "stream_start", "mode": mode, "path": source if mode != 'webcam' else 'webcam'})
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_number += 1
            display = frame.copy()

            if (frame_number % frame_skip) == 0:
                try:
                    prob = detector.predict_frame(frame)
                except Exception as e:
                    prob = None
                    cv2.putText(display, f"Predict error: {str(e)[:40]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                if prob is not None:
                    ema_prob = prob if ema_prob is None else (ema_alpha * prob + (1 - ema_alpha) * ema_prob)
                    smoothed = ema_prob
                    is_violent = 1 if smoothed >= prob_threshold else 0
                    pred_window.append(is_violent)

                    ratio = None
                    if len(pred_window) >= window_size:
                        ratio = sum(pred_window) / len(pred_window)
                        if ratio >= window_ratio:
                            now = time.time()
                            if now - last_alert_time >= alert_cooldown:
                                # Visual alert
                                cv2.rectangle(display, (0, 0), (display.shape[1], display.shape[0]), (0, 0, 255), 10)
                                cv2.putText(display, "VIOLENCE ALERT", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                                push_event({"type": "alert", "prob": round(float(prob), 3), "ema": round(float(smoothed), 3), "ratio": round(float(ratio), 3), "ts": time.time()})
                                # SMS alert
                                try:
                                    msg = f"ALERT: Violence detected | Prob={prob:.2f} EMA={smoothed:.2f} Ratio={ratio:.2f}"
                                    push_event({"type": "sms", "status": "sending", "ts": time.time()})
                                    sid = send_sms_alert(msg)
                                    push_event({"type": "sms", "status": "sent" if sid else "error", "message": "" if sid else "No SID returned", "ts": time.time()})
                                except Exception as sms_e:
                                    # Draw SMS error info unobtrusively
                                    cv2.putText(display, f"SMS err", (display.shape[1]-120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                                    push_event({"type": "sms", "status": "error", "message": str(sms_e)[:120], "ts": time.time()})
                                last_alert_time = now

                    draw_overlay(display, prob, smoothed, prob_threshold, pred_window, ratio)
                else:
                    # show buffering status
                    buf = detector.buffer_len() if hasattr(detector, 'buffer_len') else 0
                    cv2.putText(display, f"Buffering {buf}/{params['sequence_length']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            else:
                # still draw last known EMA bar if available
                if ema_prob is not None:
                    draw_overlay(display, ema_prob, ema_prob, prob_threshold, pred_window)

            # Encode frame
            try:
                ret, buffer = cv2.imencode('.jpg', display)
            except Exception:
                continue
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
    finally:
        cap.release()
        push_event({"type": "stream_end"})


@app.route("/events")
def events():
    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return Response(sse_stream(), headers=headers)


@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("file")
    if not f or f.filename == "":
        return {"ok": False, "error": "No file"}, 400
    filename = secure_filename(f.filename)
    upload_dir = os.path.join(os.path.dirname(__file__), "uploads")
    try:
        os.makedirs(upload_dir, exist_ok=True)
        save_path = os.path.join(upload_dir, filename)
        f.save(save_path)
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}, 500
    return {"ok": True, "path": save_path}


@app.route("/video_info")
def video_info():
    path = request.args.get("path", "")
    info = {
        "ok": False,
        "path": path,
        "exists": False,
        "size": 0,
        "width": 0,
        "height": 0,
        "fps": 0.0,
        "frame_count": 0,
        "duration": 0.0,
    }
    p = _sanitize_source_path(path)
    if not p or not os.path.exists(p):
        return info
    info["exists"] = True
    try:
        info["size"] = os.path.getsize(p)
    except Exception:
        pass
    try:
        cap = cv2.VideoCapture(p)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            info.update({
                "width": w,
                "height": h,
                "fps": round(fps, 3) if fps else 0.0,
                "frame_count": frames,
                "duration": round(frames / fps, 3) if fps else 0.0,
            })
        cap.release()
    except Exception:
        pass
    info["ok"] = True
    return info


@app.route("/")
def index():
    return render_template("index.html", defaults=DEFAULTS)


@app.route("/video_feed")
def video_feed():
    mode = request.args.get("mode", "webcam")  # webcam | file
    path = request.args.get("path", "")

    params = {
        "prob_threshold": parse_float(request.args.get("prob_threshold"), DEFAULTS["prob_threshold"]),
        "window_size": parse_int(request.args.get("window_size"), DEFAULTS["window_size"]),
        "window_ratio": parse_float(request.args.get("window_ratio"), DEFAULTS["window_ratio"]),
        "alert_cooldown": parse_int(request.args.get("alert_cooldown"), DEFAULTS["alert_cooldown"]),
        "sequence_length": parse_int(request.args.get("sequence_length"), DEFAULTS["sequence_length"]),
        "frame_skip": parse_int(request.args.get("frame_skip"), DEFAULTS["frame_skip"]),
        "ema_alpha": parse_float(request.args.get("ema_alpha"), DEFAULTS["ema_alpha"]),
        "input_w": parse_int(request.args.get("input_w"), DEFAULTS["input_w"]),
        "input_h": parse_int(request.args.get("input_h"), DEFAULTS["input_h"]),
    }

    return Response(gen_stream(mode, path, params), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/test_alert", methods=["POST"]) 
def test_alert():
    push_event({"type": "alert", "prob": 0.9, "ema": 0.88, "ratio": 0.75, "ts": time.time()})
    return {"ok": True}


@app.route("/test_sms", methods=["POST"]) 
def test_sms():
    try:
        sid = send_sms_alert("Test alert from UI")
        push_event({"type": "sms", "status": "sent" if sid else "error", "message": "" if sid else "No SID returned", "ts": time.time()})
        return {"ok": True, "sid": sid}
    except Exception as e:
        push_event({"type": "sms", "status": "error", "message": str(e)[:200], "ts": time.time()})
        return {"ok": False}, 500


if __name__ == "__main__":
    # Run on localhost:5000; threaded for SSE + MJPEG, disable reloader to avoid double processes
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True, use_reloader=False)
