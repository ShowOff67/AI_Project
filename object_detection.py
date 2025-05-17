import cv2
import numpy as np
import tensorflow as tf

# Load the detection model
MODEL_PATH = 'models/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model'
detect_fn = tf.saved_model.load(MODEL_PATH)

# Load class names from labelmap.txt
with open('label_map.txt', 'r') as f:
    class_names = f.read().strip().split('\n')

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to tensor
    image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = tf.convert_to_tensor([image_np], dtype=tf.uint8)
    detections = detect_fn(input_tensor)

    # Extract detection data
    boxes = detections['detection_boxes'][0].numpy()
    classes = detections['detection_classes'][0].numpy().astype(np.int32)
    scores = detections['detection_scores'][0].numpy()

    h, w, _ = frame.shape

    for i in range(len(scores)):
        if scores[i] > 0.5:
            y1, x1, y2, x2 = boxes[i]
            x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
            class_index = classes[i] - 1
            if 0 <= class_index < len(class_names):
                label = class_names[class_index]
            else:
                label = f'Unknown ({classes[i]})'
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label}: {int(scores[i]*100)}%', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Object Detection', frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()