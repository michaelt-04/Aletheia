#!/usr/bin/env python3
"""
yolo_live.py - Live Camera YOLO Detection for Project Aletheia
===============================================================

Runs YOLO26 object detection on the Raspberry Pi camera feed in real-time.
Outputs detections as JSON to stdout so the GUI process can read them,
OR writes to a shared state file for inter-process communication.

Features:
    - Thermal throttling: monitors RPi CPU temp, slows down if > 80°C
    - Configurable detection interval (default 2-3 seconds)
    - Outputs detections + carbon velocity for the Aletheia HUD
    - Can run standalone or be imported as a thread

Usage (standalone):
    python yolo_live.py
    python yolo_live.py --model yolo26n_xnnpack.pte --interval 2.0
    python yolo_live.py --headless  # No preview window

Usage (as module in aletheia_os.py):
    from yolo_live import DetectionThread
    dt = DetectionThread(model_path="meta-yolo/yolo26n_xnnpack.pte", shared_state=state, lock=lock)
    dt.start()

Requirements:
    pip install executorch opencv-python numpy torch
"""

import argparse
import json
import os
import sys
import threading
import time
import numpy as np

try:
    import cv2
except ImportError:
    print("ERROR: OpenCV not found. Install with: pip install opencv-python")
    sys.exit(1)

try:
    from picamera2 import Picamera2
    HAS_PICAMERA2 = True
except ImportError:
    HAS_PICAMERA2 = False

# Add parent directory to path so we can import yolo_engine
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yolo_engine import YOLODetector


# --- Thermal Management ---

def get_cpu_temp():
    """
    Read Raspberry Pi CPU temperature.
    Returns temperature in Celsius, or -1 if not available.
    """
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            temp = int(f.read().strip()) / 1000.0
            return temp
    except (FileNotFoundError, ValueError):
        return -1.0


def get_throttle_delay(temp, base_interval):
    """
    Calculate extra delay based on CPU temperature.

    Thermal zones:
        < 70°C  : No throttle (run at base_interval)
        70-75°C : Slight slowdown (+1s)
        75-80°C : Moderate slowdown (+3s)
        80-85°C : Heavy throttle (+6s)
        > 85°C  : Emergency throttle (+10s)

    Returns:
        Total interval (base + throttle) in seconds
    """
    if temp < 0:
        return base_interval  # Temp not available, no throttle

    if temp < 70:
        return base_interval
    elif temp < 75:
        return base_interval + 1.0
    elif temp < 80:
        return base_interval + 3.0
    elif temp < 85:
        return base_interval + 6.0
    else:
        return base_interval + 10.0


# --- Detection Thread (for integration with aletheia_os.py) ---

class DetectionThread(threading.Thread):
    """
    Background thread that captures camera frames, runs YOLO detection,
    and updates a shared state dictionary.

    Args:
        model_path: Path to .pte model file
        shared_state: Dict with keys "detected_objects", "carbon_velocity", "app_quit"
        state_lock: threading.Lock for safe access to shared_state
        camera_index: Camera device index (default 0)
        base_interval: Seconds between detections (default 2.5)
        confidence: Detection confidence threshold (default 0.25)
    """

    def __init__(self, model_path, shared_state, state_lock,
                 camera_index=0, base_interval=2.5, confidence=0.25):
        super().__init__(daemon=True)
        self.model_path = model_path
        self.shared_state = shared_state
        self.state_lock = state_lock
        self.camera_index = camera_index
        self.base_interval = base_interval
        self.confidence = confidence
        self._detector = None

    def run(self):
        print(f"[DetectionThread] Starting...")
        print(f"[DetectionThread] Model: {self.model_path}")
        print(f"[DetectionThread] Interval: {self.base_interval}s")
        print(f"[DetectionThread] Confidence: {self.confidence}")

        # Load model
        try:
            self._detector = YOLODetector(
                self.model_path,
                confidence_threshold=self.confidence
            )
        except Exception as e:
            print(f"[DetectionThread] ERROR loading model: {e}")
            return

        # Open camera using Picamera2 (Pi Camera) or OpenCV (USB webcam)
        picam = None
        cap = None

        if HAS_PICAMERA2:
            try:
                picam = Picamera2()
                config = picam.create_still_configuration(
                    main={"size": (640, 480), "format": "RGB888"}
                )
                picam.configure(config)
                picam.start()
                time.sleep(1.0)  # Let camera warm up
                print(f"[DetectionThread] Picamera2 opened successfully.")
            except Exception as e:
                print(f"[DetectionThread] Picamera2 failed: {e}, falling back to OpenCV")
                picam = None

        if picam is None:
            cap = cv2.VideoCapture(self.camera_index)
            if not cap.isOpened():
                print(f"[DetectionThread] ERROR: Could not open camera {self.camera_index}")
                return
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            print(f"[DetectionThread] OpenCV camera opened.")

        print(f"[DetectionThread] Running detection loop...")

        frame_count = 0
        while True:
            # Check quit signal
            with self.state_lock:
                if self.shared_state.get("app_quit", False):
                    break

            # Capture frame
            frame = None
            if picam is not None:
                try:
                    rgb_frame = picam.capture_array()
                    # Picamera2 with RGB888 gives RGB, but YOLO engine expects BGR
                    frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                except Exception as e:
                    print(f"[DetectionThread] Capture error: {e}")
            else:
                ret, frame = cap.read()
                if not ret:
                    frame = None

            if frame is None:
                print("[DetectionThread] WARNING: Failed to capture frame, retrying...")
                time.sleep(0.5)
                continue

            # Run detection
            t0 = time.time()
            detections = self._detector.detect(frame)
            inference_time = time.time() - t0

            # Compute carbon velocity
            carbon_v = YOLODetector.compute_carbon_velocity(detections)

            # Get thermal info
            cpu_temp = get_cpu_temp()

            # Update shared state
            with self.state_lock:
                self.shared_state["detected_objects"] = detections
                self.shared_state["carbon_velocity"] = carbon_v
                self.shared_state["cpu_temp"] = cpu_temp
                self.shared_state["inference_ms"] = inference_time * 1000
                self.shared_state["detection_count"] = len(detections)

            # Log to terminal
            frame_count += 1
            temp_str = f"{cpu_temp:.1f}°C" if cpu_temp > 0 else "N/A"
            labels = [d["label"] for d in detections[:5]]
            labels_str = ", ".join(labels) if labels else "none"
            if len(detections) > 5:
                labels_str += f" (+{len(detections)-5} more)"

            print(f"[Detection #{frame_count}] {len(detections)} objects | "
                  f"Carbon: {carbon_v:.2f} | {inference_time*1000:.0f}ms | "
                  f"Temp: {temp_str} | Objects: {labels_str}")

            # Thermal throttling
            interval = get_throttle_delay(cpu_temp, self.base_interval)
            if interval > self.base_interval:
                print(f"[DetectionThread] Thermal throttle: waiting {interval:.1f}s "
                      f"(temp: {cpu_temp:.1f}°C)")

            # Sleep until next detection
            sleep_time = max(0, interval - inference_time)
            # Sleep in small chunks so we can respond to quit signal quickly
            sleep_end = time.time() + sleep_time
            while time.time() < sleep_end:
                with self.state_lock:
                    if self.shared_state.get("app_quit", False):
                        break
                time.sleep(0.1)

        # Cleanup
        if picam is not None:
            picam.stop()
        if cap is not None:
            cap.release()
        print("[DetectionThread] Stopped.")


# --- Standalone Mode ---

def main():
    """Run YOLO live detection as a standalone process."""
    parser = argparse.ArgumentParser(description="Live YOLO26 Detection - Project Aletheia")
    parser.add_argument("--model", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             "yolo26n_xnnpack.pte"),
                        help="Path to .pte model")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera device index (default: 0)")
    parser.add_argument("--interval", type=float, default=2.5,
                        help="Seconds between detections (default: 2.5)")
    parser.add_argument("--confidence", type=float, default=0.25,
                        help="Confidence threshold (default: 0.25)")
    parser.add_argument("--headless", action="store_true",
                        help="No preview window (for SSH / headless Pi)")
    parser.add_argument("--json", action="store_true",
                        help="Output detections as JSON lines (for piping to GUI)")
    args = parser.parse_args()

    print("=" * 60)
    print("  YOLO26 Live Detection - Project Aletheia")
    print("=" * 60)

    # Load model
    print(f"\nLoading model: {args.model}")
    detector = YOLODetector(args.model, confidence_threshold=args.confidence)

    # Open camera
    picam = None
    cap = None

    if HAS_PICAMERA2:
        try:
            print(f"Opening Picamera2...")
            picam = Picamera2()
            config = picam.create_still_configuration(
                main={"size": (640, 480), "format": "RGB888"}
            )
            picam.configure(config)
            picam.start()
            time.sleep(1.0)
            print("Picamera2 opened successfully.")
        except Exception as e:
            print(f"Picamera2 failed: {e}, falling back to OpenCV")
            picam = None

    if picam is None:
        print(f"Opening OpenCV camera {args.camera}...")
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print("ERROR: Could not open camera.")
            sys.exit(1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print(f"Detection interval: {args.interval}s")
    print(f"Confidence threshold: {args.confidence}")
    print(f"Thermal throttling: enabled (target < 80°C)")
    print(f"\nStarting live detection... Press Ctrl+C to stop.\n")

    frame_count = 0
    try:
        while True:
            # Capture frame
            frame = None
            if picam is not None:
                try:
                    rgb_frame = picam.capture_array()
                    frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                except Exception as e:
                    print(f"Capture error: {e}")
            else:
                ret, frame = cap.read()
                if not ret:
                    frame = None

            if frame is None:
                print("WARNING: Failed to capture frame")
                time.sleep(0.5)
                continue

            # Run detection
            t0 = time.time()
            detections = detector.detect(frame)
            inference_time = time.time() - t0

            carbon_v = YOLODetector.compute_carbon_velocity(detections)
            cpu_temp = get_cpu_temp()

            frame_count += 1

            if args.json:
                # JSON output mode (for piping to another process)
                output = {
                    "frame": frame_count,
                    "detections": detections,
                    "carbon_velocity": carbon_v,
                    "inference_ms": round(inference_time * 1000, 1),
                    "cpu_temp": round(cpu_temp, 1),
                    "timestamp": time.time()
                }
                print(json.dumps(output), flush=True)
            else:
                # Human-readable output
                temp_str = f"{cpu_temp:.1f}°C" if cpu_temp > 0 else "N/A"
                print(f"\n--- Detection #{frame_count} ---")
                print(f"  Inference: {inference_time*1000:.0f}ms | Temp: {temp_str}")
                print(f"  Carbon Velocity: {carbon_v:.2f}")

                if detections:
                    print(f"  Objects ({len(detections)}):")
                    for d in detections:
                        x1, y1, x2, y2 = d["box"]
                        print(f"    {d['label']:<16} {d['confidence']:.2f}  "
                              f"({x1},{y1},{x2},{y2})  [{d['carbon_impact']}]")
                else:
                    print("  No objects detected.")

            # Show preview window (if not headless)
            if not args.headless and detections:
                preview = frame.copy()
                for d in detections:
                    x1, y1, x2, y2 = d["box"]
                    color = {"high": (0,0,255), "medium": (0,165,255),
                             "low": (0,200,0)}.get(d["carbon_impact"], (200,200,0))
                    cv2.rectangle(preview, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(preview, f"{d['label']} {d['confidence']:.2f}",
                                (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                cv2.imshow("Aletheia - YOLO Live", preview)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Thermal throttling
            interval = get_throttle_delay(cpu_temp, args.interval)
            if interval > args.interval and not args.json:
                print(f"  [THROTTLE] Temp {cpu_temp:.1f}°C -> interval {interval:.1f}s")

            sleep_time = max(0, interval - inference_time)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n\nStopping...")

    if picam is not None:
        picam.stop()
    if cap is not None:
        cap.release()
    if not args.headless:
        cv2.destroyAllWindows()
    print("Live detection stopped.")


if __name__ == "__main__":
    main()
