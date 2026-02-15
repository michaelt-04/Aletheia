import cv2
import mediapipe as mp
import math
import time
from picamera2 import Picamera2
from libcamera import controls

# --- Setup MediaPipe ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# --- Setup RPi Camera ---
picam2 = Picamera2()
# Configure for 640x480 for higher FPS on RPi
config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"}
)
picam2.configure(config)
# Enable Continuous Autofocus for Camera Module 3
picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
picam2.start()

print("Hand Tracking Test Started. Press 'q' in the window to quit.")

try:
    while True:
        # 1. Capture frame directly as RGB for MediaPipe
        frame_rgb = picam2.capture_array()
        
        # 2. Process with MediaPipe
        results = hands.process(frame_rgb)
        
        # 3. Convert to BGR for OpenCV display
        display_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                # Draw the skeleton
                mp_draw.draw_landmarks(display_frame, hand_lms, mp_hands.HAND_CONNECTIONS)
                
                # Get specific coordinates for pinch detection
                # Index 4 = Thumb Tip, Index 8 = Index Tip
                thumb = hand_lms.landmark[4]
                index = hand_lms.landmark[8]
                
                # Calculate distance (normalized 0.0 to 1.0)
                dist = math.hypot(index.x - thumb.x, index.y - thumb.y)
                
                if dist < 0.05: # Pinch threshold
                    cv2.putText(display_frame, "PINCH!", (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    print("PINCH DETECTED!")

        cv2.imshow("Aletheia Hand Test", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()
    print("Test Stopped.")