import cv2
import mediapipe as mp
import math
import time
from picamera2 import Picamera2
from libcamera import controls

# --- Setup MediaPipe ---
mp_hands = mp.solutions.hands
# static_image_mode=False is better for RPi video performance
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# --- Setup RPi Camera ---
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"}
)
picam2.configure(config)

# Fix for Module 3: Enable Continuous Autofocus
picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})

# FIXED: Corrected the case-sensitivity for Greyworld
try:
    # Most systems use 'Greyworld', some use 'GreyWorld'
    picam2.set_controls({"AwbMode": controls.AwbModeEnum.Greyworld})
except AttributeError:
    # Fallback for different library versions
    picam2.set_controls({"AwbMode": controls.AwbModeEnum.GreyWorld})

picam2.start()

print("Hand Tracking Test Started.")
print("Controls: 'q' to quit | Result: Corrected Colors, No Outlines")

try:
    while True:
        # 1. Capture frame (RGB format)
        frame_rgb = picam2.capture_array()
        
        # 2. Process with MediaPipe
        results = hands.process(frame_rgb)
        
        # 3. Fix "Blue Hand": Convert RGB to BGR for OpenCV display
        display_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                # Get coordinates for thumb tip (4) and index tip (8)
                thumb_tip = hand_lms.landmark[4]
                index_tip = hand_lms.landmark[8]
                
                # Calculate distance
                dist = math.hypot(index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y)
                
                # If pinching, draw the ethereal glow indicator
                if dist < 0.05:
                    ix = int(index_tip.x * 640)
                    iy = int(index_tip.y * 480)
                    
                    # Outer glow (Green)
                    cv2.circle(display_frame, (ix, iy), 15, (0, 255, 0), -1)
                    # Inner core (White)
                    cv2.circle(display_frame, (ix, iy), 5, (255, 255, 255), -1)

        cv2.imshow("Aletheia Hand Diagnostic", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()
    hands.close()
    print("Test Stopped Cleanly.")