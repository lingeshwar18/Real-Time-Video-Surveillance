import cv2
import time
import os
import numpy as np
from detection import ViolenceDetector
from alert import send_sms_alert
from collections import deque

# === CONFIG ===
# Create a list of video paths you want to process
VIDEO_PATHS = [
    r"C:\Users\linge\Downloads\DATASET\dataset 1.mp4",
    r"C:\Users\linge\Downloads\DATASET\dataset 2.mp4",
    r"C:\Users\linge\Downloads\DATASET\dataset 3.mp4",
    r"C:\Users\linge\Downloads\DATASET\dataset 4.mp4",
    r"C:\Users\linge\Downloads\DATASET\dataset 5.mp4"
]
USE_WEBCAM = False
SINGLE_MODEL_PATH = "models/violence_detection_model.h5"

INPUT_FPS = 20
FRAME_SKIP = 1
INPUT_SIZE = (224, 224)
PROB_THRESHOLD = 0.6  # relaxed for debugging
WINDOW_SIZE = 20      # relaxed for debugging
WINDOW_TRIGGER_RATIO = 0.5  # relaxed for debugging
ALERT_COOLDOWN = 30
SEQUENCE_LENGTH = 20
FORCE_ALERT_ON_START = False  # send a one-time SMS to validate pipeline
EMA_ALPHA = 0.2  # smoothing factor for probability (0=no smoothing, 1=instant)

# === prepare detector ===
detector = ViolenceDetector(
    model_path=SINGLE_MODEL_PATH,
    input_size=INPUT_SIZE,
    sequence_length=SEQUENCE_LENGTH
)

# === video processing loop ===
force_alert_sent = False
for video_path in VIDEO_PATHS:
    print("-" * 50)
    print(f"Starting to process video: {os.path.basename(video_path)}")
    print("-" * 50)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Warning: Could not open video source at {video_path}. Skipping.")
        continue

    # Reset per-video state
    detector.reset()
    frame_buffer = deque(maxlen=INPUT_FPS * 8)
    pred_window = deque(maxlen=WINDOW_SIZE)
    last_alert_time = 0
    
    # One-time forced SMS to confirm Twilio path works end-to-end
    if FORCE_ALERT_ON_START and not force_alert_sent:
        try:
            test_msg = f"Test alert: pipeline check for {os.path.basename(video_path)}"
            print("[DEBUG] Sending forced test SMS...")
            send_sms_alert(test_msg)
            force_alert_sent = True
        except Exception as e:
            print(f"[DEBUG] Forced SMS failed: {e}")
    
    ema_prob = None
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Finished processing {os.path.basename(video_path)}.")
            break

        frame_buffer.append(frame)
        display = frame.copy()
        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        if (frame_number % FRAME_SKIP) == 0:
            prob = detector.predict_frame(frame)
            
            if prob is not None:
                # Exponential moving average smoothing
                ema_prob = prob if ema_prob is None else (EMA_ALPHA * prob + (1 - EMA_ALPHA) * ema_prob)
                smoothed = ema_prob
                is_violent = 1 if smoothed >= PROB_THRESHOLD else 0
                
                prob_display = int(smoothed * 100)
                color = (0, 255, 0) if smoothed < PROB_THRESHOLD else (0, 0, 255)
                cv2.rectangle(display, (10, 80), (10 + prob_display, 100), color, -1)
                cv2.putText(display, f"Prob: {prob:.2f} | EMA: {smoothed:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                # Debug probability each step
                if frame_number % (FRAME_SKIP * 5) == 0:
                    print(f"[DEBUG] frame={frame_number} prob={prob:.3f} ema={smoothed:.3f} violent={is_violent}")
                
                pred_window.append(is_violent)

                if len(pred_window) >= WINDOW_SIZE:
                    ratio = sum(pred_window) / len(pred_window)
                    print(f"[DEBUG] window len={len(pred_window)} ratio={ratio:.3f}")
                    if ratio >= WINDOW_TRIGGER_RATIO:
                        now = time.time()
                        if now - last_alert_time >= ALERT_COOLDOWN:
                            print(f"\n[ALERT] Violence detected in {os.path.basename(video_path)}!")
                            print(f"  - Trigger Ratio: {ratio:.2f}")
                            print(f"  - Current Probability: {prob:.2f}  EMA: {smoothed:.2f}\n")
                            
                            # Twilio-only alert message
                            message = (
                                f"ALERT: Violence detected in {os.path.basename(video_path)} | "
                                f"Prob={prob:.2f}, EMA={smoothed:.2f}, Ratio={ratio:.2f}"
                            )
                            try:
                                send_sms_alert(message)
                            except Exception as e:
                                print(f"SMS failed: {e}")
                            last_alert_time = now

                        cv2.rectangle(display, (0, 0), (display.shape[1], display.shape[0]), (0, 0, 255), 10)
                        cv2.putText(display, "VIOLENCE ALERT", (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                    else:
                        cv2.putText(display, f"Normal ({sum(pred_window)}/{len(pred_window)})", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                buf = detector.buffer_len() if hasattr(detector, 'buffer_len') else 0
                cv2.putText(display, f"Buffering frames {buf}/{SEQUENCE_LENGTH}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.imshow(f"Video Surveillance: {os.path.basename(video_path)}", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    if key == ord("q"):
        break

cv2.destroyAllWindows()