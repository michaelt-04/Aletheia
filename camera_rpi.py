# camera_rpi.py
# Centralized camera controller for Raspberry Pi and generic webcams.
#
# This script provides a thread-safe class to manage a camera.
# It automatically detects if it's running on a Raspberry Pi and uses the
# efficient `picamera2` library, otherwise it falls back to using `cv2.VideoCapture`
# for general webcam compatibility (e.g., on a Mac or PC).

import time
import threading
import platform
import cv2

# --- Platform-specific camera implementation ---

IS_RPI = platform.machine().startswith(('arm', 'aarch64'))

if IS_RPI:
    try:
        from picamera2 import Picamera2
        print("[CameraManager] Raspberry Pi detected. Using picamera2.")
    except ImportError:
        print("[CameraManager] WARNING: Running on a RPi-like architecture, but picamera2 is not installed.")
        IS_RPI = False

class RPiCamera:
    """A thread-safe camera manager for the Raspberry Pi camera module."""
    def __init__(self, width=1280, height=720):
        print(f"[RPiCamera] Initializing camera with resolution {width}x{height}...")
        self.picam2 = Picamera2()
        self.config = self.picam2.create_preview_configuration(
            main={"size": (width, height), "format": "RGB888"}
        )
        self.picam2.configure(self.config)
        
        self.frame = None
        self.lock = threading.Lock()
        self.running = False
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        print("[RPiCamera] Initialization complete.")

    def start(self):
        if self.running: return
        print("[RPiCamera] Starting camera...")
        self.picam2.start()
        self.running = True
        self.thread.start()
        print("[RPiCamera] Capture thread started.")

    def stop(self):
        if not self.running: return
        print("[RPiCamera] Stopping camera thread...")
        self.running = False
        self.thread.join(timeout=2)
        self.picam2.stop()
        print("[RPiCamera] Camera stopped.")

    def _capture_loop(self):
        while self.running:
            captured_frame = self.picam2.capture_array()
            with self.lock:
                self.frame = captured_frame
        print("[RPiCamera] Capture loop finished.")

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

class WebcamCamera:
    """A thread-safe camera manager for generic webcams using OpenCV."""
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
        if self.running: return
        print("[WebcamCamera] Starting capture thread...")
        self.running = True
        self.thread.start()
        print("[WebcamCamera] Capture thread started.")

    def stop(self):
        if not self.running: return
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
                # Add a small delay to prevent a tight loop on read error
                time.sleep(0.01)
        print("[WebcamCamera] Capture loop finished.")

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

# --- Factory Function ---

def get_camera_manager(width=1280, height=720):
    """
    Factory function that returns the appropriate camera manager based on the
    detected hardware.
    """
    if IS_RPI:
        return RPiCamera(width, height)
    else:
        print("[CameraManager] No RPi detected. Falling back to Webcam/OpenCV.")
        return WebcamCamera(width, height)

# --- Test Block ---

if __name__ == '__main__':
    print("--- CameraManager Test ---")
    
    # This test will now work on both RPi and other systems.
    camera_manager = get_camera_manager()
    camera_manager.start()
    
    try:
        for i in range(5):
            print(f"Loop {i+1}/5...")
            frame = camera_manager.get_frame()
            if frame is not None:
                print(f"  - Got frame of shape: {frame.shape} and dtype: {frame.dtype}")
            else:
                print("  - No frame available yet.")
            time.sleep(1)
            
    except (KeyboardInterrupt, SystemExit):
        print("
Interrupted.")
        
    finally:
        print("Shutting down camera.")
        camera_manager.stop()
        print("--- Test Complete ---")
