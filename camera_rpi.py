# camera_rpi.py
# Centralized camera controller for Raspberry Pi and generic webcams.
#
# Raspberry Pi OS Bookworm note:
# - The CLI apps were renamed from libcamera-* to rpicam-*, but the Python stack
#   is still Picamera2 + libcamera.
#
# IMPORTANT for virtualenv users:
# - If you installed Picamera2 via apt (python3-picamera2), it lives in the system
#   Python and will NOT be visible inside a default venv.
#   Create your venv with: python3 -m venv --system-site-packages venv

import time
import threading
import platform
import cv2

print("[CameraManager] Starting platform detection...")
IS_RPI = False
try:
    machine = platform.machine()
    print(f"[CameraManager] platform.machine() returned: '{machine}'")
    if machine.startswith(("arm", "aarch64")):
        print("[CameraManager] ARM architecture detected. This appears to be a Raspberry Pi.")
        IS_RPI = True
    else:
        print(f"[CameraManager] Non-ARM architecture ('{machine}') detected.")
except Exception as e:
    print(f"[CameraManager] An exception occurred during platform detection: {e}")

if IS_RPI:
    print("[CameraManager] Attempting to import 'picamera2'...")
    try:
        # Import only what we actually use (avoids pulling preview/window deps)
        from picamera2 import Picamera2
        from libcamera import controls
        print("[CameraManager] Successfully imported 'picamera2' and 'libcamera.controls'.")
    except ImportError as e:
        print("[CameraManager] FAILED to import 'picamera2' (or its libcamera bindings).")
        print(f"[CameraManager] Error details: {e}")
        print("[CameraManager] If you installed with apt (python3-picamera2), recreate your venv with:")
        print("               python3 -m venv --system-site-packages venv")
        IS_RPI = False
    except Exception as e:
        print("[CameraManager] Unexpected error while importing 'picamera2'.")
        print(f"[CameraManager] Error details: {e}")
        IS_RPI = False


class RPiCamera:
    """Thread-safe camera manager for the Raspberry Pi camera module (Picamera2)."""
    def __init__(self, width=1280, height=720):
        print(f"[RPiCamera] Initializing camera with resolution {width}x{height}...")
        self.picam2 = Picamera2()

        # RGB frames for MediaPipe (expects RGB)
        config = self.picam2.create_preview_configuration(
            main={"size": (width, height), "format": "RGB888"}
        )
        self.picam2.configure(config)

        # Camera Module 3: enable continuous autofocus when available
        try:
            af_value = None
            if hasattr(controls, "AfModeEnum") and hasattr(controls.AfModeEnum, "Continuous"):
                af_value = controls.AfModeEnum.Continuous
            elif hasattr(controls, "AfMode") and hasattr(controls.AfMode, "Continuous"):
                af_value = controls.AfMode.Continuous

            if af_value is not None:
                self.picam2.set_controls({"AfMode": af_value})
                print("[RPiCamera] Continuous autofocus enabled.")
            else:
                print("[RPiCamera] Autofocus enum not found; skipping AfMode control.")
        except Exception as e:
            print(f"[RPiCamera] Warning: could not set autofocus controls: {e}")

        self.frame = None
        self.lock = threading.Lock()
        self.running = False
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        print("[RPiCamera] Initialization complete.")

    def start(self):
        if self.running:
            return
        print("[RPiCamera] Starting camera...")
        self.picam2.start()
        # Warm up sensor/exposure/autofocus
        time.sleep(0.5)
        self.running = True
        self.thread.start()
        print("[RPiCamera] Capture thread started.")

    def stop(self):
        if not self.running:
            return
        print("[RPiCamera] Stopping camera thread...")
        self.running = False
        self.thread.join(timeout=2)
        self.picam2.stop()
        print("[RPiCamera] Camera stopped.")

    def _capture_loop(self):
        while self.running:
            try:
                captured_frame = self.picam2.capture_array()
                with self.lock:
                    self.frame = captured_frame
            except Exception as e:
                # Prevent tight loop if the camera hiccups
                print(f"[RPiCamera] Capture error: {e}")
                time.sleep(0.05)

            # Avoid pegging a CPU core unnecessarily
            time.sleep(0.01)
        print("[RPiCamera] Capture loop finished.")

    def get_frame(self):
        # Fast path: return latest frame reference (NO COPY).
        # IMPORTANT: treat returned array as READ-ONLY.
        with self.lock:
            return self.frame

    def get_frame_copy(self):
        # Safe path: returns a copy for any code that will draw/mutate the frame.
        with self.lock:
            return self.frame.copy() if self.frame is not None else None



class WebcamCamera:
    """Thread-safe camera manager for generic webcams using OpenCV."""
    def __init__(self, width=1280, height=720, camera_index=0):
        print(f"[WebcamCamera] Initializing camera #{camera_index} with resolution {width}x{height}...")
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera index {camera_index}.")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.frame = None
        self.lock = threading.Lock()
        self.running = False
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        print("[WebcamCamera] Initialization complete.")

    def start(self):
        if self.running:
            return
        print("[WebcamCamera] Starting capture thread...")
        self.running = True
        self.thread.start()
        print("[WebcamCamera] Capture thread started.")

    def stop(self):
        if not self.running:
            return
        print("[WebcamCamera] Stopping camera thread...")
        self.running = False
        self.thread.join(timeout=2)
        self.cap.release()
        print("[WebcamCamera] Camera released.")

    def _capture_loop(self):
        while self.running:
            ret, frame_bgr = self.cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                with self.lock:
                    self.frame = frame_rgb
            else:
                # Prevent a tight loop if the device isn't producing frames
                time.sleep(0.01)
        print("[WebcamCamera] Capture loop finished.")

    def get_frame(self):
    # Fast path: return latest frame reference (NO COPY).
        # IMPORTANT: treat returned array as READ-ONLY.
        with self.lock:
            return self.frame

    def get_frame_copy(self):
        # Safe path: returns a copy for any code that will draw/mutate the frame.
        with self.lock:
            return self.frame.copy() if self.frame is not None else None


def get_camera_manager(width=1280, height=720):
    """Return the appropriate camera manager based on detected hardware + availability."""
    if IS_RPI:
        return RPiCamera(width, height)
    else:
        print("[CameraManager] Using OpenCV VideoCapture fallback (no Picamera2 available).")
        return WebcamCamera(width, height)
