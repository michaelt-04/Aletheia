Project Aletheia: Standalone AR OS

Aletheia is an experimental standalone Augmented Reality operating system built for the Raspberry Pi 4. Developed for SF Hacks 2026, the project focuses on local high performance spatial computing without reliance on external smartphones or cloud processing.

Aletheia transforms standard AR glasses into a context aware workstation. By combining real time object detection with precise hand tracking, the system identifies environmental energy waste and gamifies carbon reduction through an immersive HUD. Instead of just displaying climate data, it makes the impact feel immediate through a visual embodiment of environmental cost.
System Architecture

The application is built on a modular, event driven pipeline designed to maximize the hardware resources of the Raspberry Pi 4.
Hardware Layer

    Brain: Raspberry Pi 4 (4GB) running Raspberry Pi OS.

    Sensor: RPi Camera Module mounted to the bridge of Xreal Air 2 Pro glasses for a first person perspective.

    Display: Xreal Air 2 Pro glasses connected via an HDMI to USB C adapter.

    Optics: The GUI uses a pure black background (0,0,0 RGB) to leverage OLED transparency, creating a floating interface in the wearer's field of view.

AI Inference Layer (The ExecuTorch Pipeline)

The use of ExecuTorch is integral to Aletheia because it enables high performance, low latency inference directly on the Raspberry Pi 4's edge hardware. Unlike standard wrappers that rely on heavy runtimes, ExecuTorch allows for a significantly reduced memory footprint and specialized kernel execution tailored for ARM architectures. This is critical for maintaining the high frame rates required for a comfortable AR experience without cloud dependency.
PTE Model Conversion

We manually converted specialized computer vision models into the .pte (PyTorch Edge) format to achieve bare metal performance. This involved tracing PyTorch models to capture the computational graph and applying XNNPACK backend delegates to accelerate operators for the RPi 4 ARM architecture.

The following custom converted models are utilized:

    Meta YOLO (v26n): Converted to .pte to identify energy waste items such as lights and monitors.

    BlazePalm and BlazeHand: Converted to .pte to provide local, hands free gesture recognition, ensuring all hand tracking remains on device for privacy and speed.

Software Layer

    Multiprocessing: The system uses a dedicated YOLO worker process to bypass the Python Global Interpreter Lock, utilizing Shared Memory to pass video frames with zero copy overhead.

    Asynchronous Hand Tracking: A separate process manages the .pte hand tracking models to ensure gesture recognition does not block the main HUD updates.

    Custom GUI: A Pygame based HUD featuring a Spirit Companion, a mission tracker, and a real time carbon savings widget.

Interaction and Quests

The system features an interactive quest function that bridges environmental detection with user action:

    Spatial Anchoring: The system extracts localized coordinates from the YOLO .pte model detections.

    Particle Overlay: The HUD generates dynamic particle effects at these specific coordinates to highlight "energy leaks" in the physical world.

    Pinch Interaction: When the user performs a pinch gesture at the particle location, detected via the BlazeHand process, the system triggers a popup quest or mission.

    Carbon Tracking: Completing these quests updates the carbon savings widget and mission tracker in real time.

Setup and Installation

The project uses uv for environment management.

    Configure Display:
    Run the provided script to force 1080p output for AR glasses.
    chmod +x setup_display.sh
    ./setup_display.sh

    Environment Setup:
    uv pip install -r requirements.txt

    Run Aletheia:
    python aletheia_os.py

SF Hacks 2026 | Built for the future of private local and sustainable AR.
