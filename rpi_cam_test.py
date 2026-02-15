# camera_rpi.py
# Updated for RPi OS Bookworm (rpicam-apps compatibility)

import time
import threading
import platform
import cv2

# --- Platform-specific camera implementation ---
IS_RPI = False
try:
    machine = platform.machine()
    if machine.startswith(('arm', 'aarch64')):
        IS_RPI = True
except Exception:
    IS_RPI = False

if IS_RPI:
    try:
        from picamera2 import Picamera2
        # Import libcamera controls for Autofocus support
        from libcamera import controls
        print("[CameraManager] Successfully imported 'picamera2' and 'libcamera' controls.")
    except ImportError:
        print("[CameraManager] FAILED to import 'picamera2'. Falling back to OpenCV.")
        IS_RPI = False

class RPiCamera:
    """Updated thread-safe camera manager for RPi Camera Module 3."""
    def __init__(self, width=1280, height=720):
        print(f"[RPiCamera] Initializing Module 3 at {width}x{height}...")
        self.picam2 = Picamera2()
        
        # Create configuration optimized for RGB processing
        self.config = self.picam2.create_preview_configuration(
            main={"size": (width, height), "format": "RGB888"}
        )
        self.picam2.configure(self.config)
        
        # CRITICAL: Set Continuous Autofocus for Module 3
        # This prevents the 'black screen' caused by the lens being stuck
        self.picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
        
        self.frame = None
        self.lock = threading.Lock()
        self.running = False
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)

    def start(self):
        if self.running: return
        print("[RPiCamera] Starting rpicam-backend capture...")
        self.picam2.start()
        # Small delay to allow the sensor to calibrate and focus
        time.sleep(0.5) 
        self.running = True
        self.thread.start()

    def stop(self):
        if not self.running: return
        self.running = False
        self.thread.join(timeout=2)
        self.picam2.stop()

    def _capture_loop(self):
        while self.running:
            # Efficiently capture the latest buffer into a numpy array
            captured_frame = self.picam2.capture_array()
            with self.lock:
                self.frame = captured_frame
            # Prevent CPU pegging while maintaining high framerate
            time.sleep(0.01) 

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

# (WebcamCamera and get_camera_manager classes remain the same as your original)