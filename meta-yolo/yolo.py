from executorch.runtime import Runtime, Program, Method
import cv2
import numpy as np

# Load the model
runtime = Runtime.get()
program = runtime.load_program("yolo26n.pte")
method = program.load_method("forward")

# Capture from camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    # Preprocess: resize to 640x640, normalize, convert to tensor
    input_tensor = preprocess(frame)
    # Run inference
    output = method.execute([input_tensor])
    # Post-process: NMS, draw bounding boxes
    detections = postprocess(output)