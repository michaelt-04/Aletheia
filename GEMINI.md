# Project Aletheia: Standalone RPi-AR OS
## Hardware Architecture
- Brain: Raspberry Pi 5 (8GB) running Raspberry Pi OS.
- Sensor: RPi Camera mounted to Xreal Air 2 Pro glasses.
- Display: HDMI to USB-C (SINK) adapter directly to glasses.
- Interaction: Hands-free gesture recognition (MediaPipe).

## Architecture Constraints
- No iPhone/Cloud: All inference (YOLO + MediaPipe) must run locally.
- GUI: Fullscreen (1920x1080) Pygame or Godot overlay with 0,0,0 black background for OLED transparency.
- Optimization: Use 'uv' for Python environments; optimize for RPi 5 CPU/GPU backends.