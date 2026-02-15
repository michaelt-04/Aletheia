# rpi_cam_test.py
# A simple script to diagnose the camera environment on the Raspberry Pi.

import platform
import sys

print(f"--- RPi Camera Diagnostic ---")
print(f"Python Executable: {sys.executable}")
print(f"Python Version: {sys.version}")

# 1. Check platform
print("[1/2] Checking platform architecture...")
machine = platform.machine()
print(f"  - platform.machine() returned: '{machine}'")
if machine.startswith(('arm', 'aarch64')):
    print("  - Result: Looks like a Raspberry Pi.")
else:
    print("  - Result: Does NOT look like a Raspberry Pi.")

# 2. Check picamera2 import
print("\n[2/2] Attempting to import picamera2...")
try:
    from picamera2 import Picamera2
    print("  - Result: Successfully imported 'picamera2'.")
except ImportError as e:
    print(f"  - Result: FAILED to import 'picamera2'.")
    print(f"  - Error: {e}")
    print("  - Note: If on a Pi, try: sudo apt install python3-picamera2")
except Exception as e:
    print(f"  - Result: An unexpected error occurred while importing picamera2.")
    print(f"  - Error: {e}")

print("\n--- Diagnostic Complete ---")