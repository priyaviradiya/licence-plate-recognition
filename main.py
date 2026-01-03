import cv2
import os
import sys
import numpy as np
from ultralytics import YOLO
import easyocr
from collections import defaultdict, deque
import re

model_path = 'license_plate_best.pt'
if not os.path.exists(model_path):
    print(f"Error: The model file '{model_path}' was not found.")
    print("Please ensure the file is in the same directory as the script.")
    sys.exit(1)

model = YOLO(model_path) #fine-tuned weights
reader = easyocr.Reader(['en'], gpu=True)

# This regex is specific to the LLNNLLL format enforced by `correct_plate_format`.
plate_memory = re.compile(r'^[A-Z]{2}[0-9]{2}[A-Z]{3}$')

def correct_plate_format(ocr_plate):
    mapping_num_to_char = {'0': 'O', '1': 'I', '2': 'Z', '5': 'S', '8': 'B'}
    mapping_char_to_num = {'O': '0', 'I': '1', 'Z': '2', 'S': '5', 'B': '8'}

    ocr_text = ocr_plate.upper().replace(" ", "")
    if len(ocr_text) != 7:
        return None
    
    corrected = []
    for i, ch in enumerate(ocr_text):
        if i < 2 or i >= 4:  # Expected letters
            if ch.isdigit() and ch in mapping_num_to_char:
                corrected.append(mapping_num_to_char[ch])
            elif ch.isalpha():
                corrected.append(ch)
            else:
                return None
        else:  # Expected digits
            if ch.isalpha() and ch in mapping_char_to_num:
                corrected.append(mapping_char_to_num[ch])
            elif ch.isdigit():
                corrected.append(ch)
            else:
                return None
            
    return ''.join(corrected)

def recognize_plate(plate_img):
    if plate_img.size == 0:
        return None
    
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    plate_resized = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    try:
        ocr_result = reader.readtext(
            plate_resized, detail=0, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        )
        if len(ocr_result) > 0:
            candidate = correct_plate_format(ocr_result[0]) 
            if candidate and plate_memory.match(candidate):
                return candidate    
    
    except Exception as e:
        print(f"An error occurred during OCR: {e}")

    return None

plate_history = defaultdict(lambda: deque(maxlen=10))
plate_final = {}

def get_box_id(x1, y1, x2, y2):
    return f"{int(x1/10)}_{int(y1/10)}_{int(x2/10)}_{int(y2/10)}"

def get_stable_plate(box_id, new_text):
    if new_text:
        plate_history[box_id].append(new_text)
        most_common = max(set(plate_history[box_id]), key=plate_history[box_id].count)
        plate_final[box_id] = most_common
    return plate_final.get(box_id)

input_video_path = 'input_video.mp4'
output_video_path = 'output_video.mp4'

cap = cv2.VideoCapture(input_video_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc,
                      cap.get(cv2.CAP_PROP_FPS),
                      (int(cap.get(3)), int(cap.get(4))))

CONF_THRESH = 0.3

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            conf = box.conf.item()
            if conf < CONF_THRESH:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plate_img = frame[y1:y2, x1:x2]

            ocr_text = recognize_plate(plate_img)

            box_id = get_box_id(x1, y1, x2, y2)
            stable_plate = get_stable_plate(box_id, ocr_text)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

            if plate_img.size > 0:
                overlay_h, overlay_w = 150, 400
                plate_resized = cv2.resize(plate_img, (overlay_w, overlay_h))
                
                oy1 = max(0, y1 - overlay_h - 40)
                ox1 = x1
                oy2, ox2 = oy1 + overlay_h, ox1 + overlay_w

                if oy2 <= frame.shape[0] and ox2 <= frame.shape[1]:
                    frame[oy1:oy2, ox1:ox2] = plate_resized
                    
                    if stable_plate:
                        # Draw black outline
                        cv2.putText(frame, stable_plate, (x1, y1 - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 6) #black outline
                        # Draw white text
                        cv2.putText(frame, stable_plate, (x1, y1 - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3) #white text

    out.write(frame)
    cv2.imshow('License Plate Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Processing complete. Output saved to", output_video_path)