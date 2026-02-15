import cv2
import mediapipe as mp
import math
import time
from picamera2 import Picamera2
from libcamera import controls

# --- Setup MediaPipe ---
mp_hands = mp.solutions.hands
# We do NOT import mp_draw as we want a clean view without outlines
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# --- Setup RPi Camera ---
picam2 = Picamera2()
# 640x480 is the sweet spot for MediaPipe performance on RPi
config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"}
)
picam2.configure(config)

# Fix for Module 3: Enable Continuous Autofocus
picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})

# Fix for NoIR: Greyworld AWB helps compensate for IR-induced blue/purple tints
picam2.set_controls({"AwbMode": controls.AwbModeEnum.Greyworld})

picam2.start()

print("Hand Tracking Test Started.")
print("Controls: 'q' to quit | Result: No outlines, Corrected Colors")

try:
    while True:
        # 1. Capture frame (RGB format)
        frame_rgb = picam2.capture_array()
        
        # 2. Process with MediaPipe (Requires RGB)
        results = hands.process(frame_rgb)
        
        # 3. Fix "Blue Hand": Convert RGB to BGR for OpenCV display
        display_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                # We skipped mp_draw.draw_landmarks to remove the skeleton outline
                
                # Pinch Detection Logic
                thumb_tip = hand_lms.landmark[4]
                index_tip = hand_lms.landmark[8]
                
                # Calculate Euclidean distance
                dist = math.hypot(index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y)
                
                # If pinching, draw a single glowing indicator at the index tip
                if dist < 0.05:
                    ix = int(index_tip.x * 640)
                    iy = int(index_tip.y * 480)
                    
                    # Draw a solid green circle to indicate the "Pinch" action
                    cv2.circle(display_frame, (ix, iy), 15, (0, 255, 0), -1)
                    # Add a small white core for an 'ethereal' look
                    cv2.circle(display_frame, (ix, iy), 5, (255, 255, 255), -1)

        # Show the corrected frame
        cv2.imshow("Aletheia Hand Diagnostic", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()
    print("Test Stopped Cleanly.")