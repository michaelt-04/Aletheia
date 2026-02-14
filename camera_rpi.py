# camera_rpi.py
# Centralized camera controller for Raspberry Pi using picamera2
#
# This script creates a thread-safe class to manage the RPi camera.
# It runs the camera in a separate thread and continuously captures frames,
# making the latest frame available to other parts of the application.

import time
import threading
from picamera2 import Picamera2

class RPiCamera:
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
        """Starts the camera and the background capture thread."""
        if self.running:
            print("[RPiCamera] Camera is already running.")
            return
            
        print("[RPiCamera] Starting camera...")
        self.picam2.start()
        self.running = True
        self.thread.start()
        print("[RPiCamera] Capture thread started.")

    def stop(self):
        """Stops the camera and the background thread."""
        if not self.running:
            print("[RPiCamera] Camera is not running.")
            return
            
        print("[RPiCamera] Stopping camera thread...")
        self.running = False
        self.thread.join(timeout=2) # Wait for thread to finish
        self.picam2.stop()
        print("[RPiCamera] Camera stopped.")

    def _capture_loop(self):
        """The main loop that continuously captures frames from the camera."""
        while self.running:
            # capture_array() returns a numpy array
            captured_frame = self.picam2.capture_array()
            
            with self.lock:
                self.frame = captured_frame
        
        print("[RPiCamera] Capture loop finished.")

    def get_frame(self):
        """
        Returns the most recent frame captured by the camera.
        
        Returns:
            numpy.ndarray: The latest frame, or None if no frame is available.
        """
        with self.lock:
            if self.frame is not None:
                return self.frame.copy()
            return None

if __name__ == '__main__':
    # Example usage and test
    print("--- RPiCamera Test ---")
    
    # This test is meant to be run on a Raspberry Pi with a camera module.
    # It will initialize the camera, run it for 10 seconds, and then shut down.
    
    rpi_camera = RPiCamera()
    rpi_camera.start()
    
    try:
        for i in range(10):
            print(f"Loop {i+1}/10...")
            frame = rpi_camera.get_frame()
            if frame is not None:
                print(f"  - Got frame of shape: {frame.shape} and dtype: {frame.dtype}")
            else:
                print("  - No frame available yet.")
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("
Interrupted by user.")
        
    finally:
        print("Shutting down camera.")
        rpi_camera.stop()
        print("--- Test Complete ---")
