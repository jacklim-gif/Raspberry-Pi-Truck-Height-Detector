import cv2
from ultralytics import YOLO
import cvzone
from picamera2 import Picamera2
import time
import base64
import os
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import numpy as np
import threading
import re
from libcamera import Transform
from gpiozero import LED, Buzzer

# USER INPUT: Truck height limit in meters
try:
    truck_height_limit = float(input("Enter the truck height limit in meters (e.g., 3.5): "))
except ValueError:
    print("Invalid input. Using default height limit of 3.5m.")
    truck_height_limit = 3.5

# Load YOLOv8 model (TFLite)
try:
    model = YOLO('best_float32.tflite')  # Your trained vehicle model
except Exception as e:
    print(f"Failed to load YOLO model: {e}")
    exit(1)

# Setup Gemini model
GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
try:
    gemini_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.4)
except Exception as e:
    print(f"Failed to initialize Gemini: {e}")
    exit(1)

# Setup PiCamera2
try:
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (640, 640), "format": "RGB888"})
    config["transform"] = Transform(hflip=False, vflip=False)
    picam2.configure(config)
    picam2.start()
    time.sleep(2)
except Exception as e:
    print(f"Camera setup failed: {e}")
    exit(1)

# Setup LED on GPIO 26 and Buzzer on GPIO 17
try:
    alert_led = LED(26)  # GPIO 26 for LED
    alert_buzzer = Buzzer(11)  # GPIO 17 for Buzzer
    alert_led.off()  # Ensure LED is off at start
    alert_buzzer.off()  # Ensure Buzzer is off at start
except Exception as e:
    print(f"Failed to initialize LED or Buzzer: {e}")
    exit(1)

# Mouse callback (optional)
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print(f"Mouse moved to: [{x}, {y}]")

cv2.namedWindow("RGB")
cv2.setMouseCallback("RGB", RGB)

# Thread-safe OCR storage
ocr_results = {}  # {track_id: text}
ocr_lock = threading.Lock()

# Encode image to Base64 for Gemini
def encode_image_to_base64(image):
    try:
        _, img_buffer = cv2.imencode(".jpg", image)
        return base64.b64encode(img_buffer).decode("utf-8")
    except Exception as e:
        print(f"Image encoding failed: {e}")
        return None

# Send cropped image to Gemini to extract height
def send_to_gemini(cropped_image, track_id):
    try:
        base64_image = encode_image_to_base64(cropped_image)
        if not base64_image:
            return
        message = HumanMessage(
            content=[
                {"type": "text", "text": (
                    "You are a smart OCR agent. Extract ONLY the height from this image. "
                    "Valid formats: '4.5m', '3.0 m', '3.0', etc. "
                    "Do NOT return any other text"
                )},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        )
        response = gemini_model.invoke([message])
        text = response.content.strip()
        with ocr_lock:
            ocr_results[track_id] = text
        print(f"[Track {track_id}] Detected height: {text}")
    except Exception as e:
        print(f"Error with Gemini for Track {track_id}: {e}")

# Parse height string to float
def parse_height(text):
    if not text:
        return None
    text = text.replace(" ", "").replace("m", "").replace("M", "")
    try:
        return float(text)
    except ValueError:
        return None

# Main Loop
frame_count = 0
try:
    while True:
        frame = picam2.capture_array()
        frame_count += 1
        if frame_count % 2 != 0:
            continue  # Process every 2nd frame

        results = model.track(frame, persist=True, classes=[0], imgsz=256)

        # Turn off LED and Buzzer by default for this frame
        alert_led.off()
        alert_buzzer.off()

        if results[0].boxes.id is not None:
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

            for track_id, box in zip(ids, boxes):
                x1, y1, x2, y2 = box
                cropped_img = frame[y1:y2, x1:x2]
                if cropped_img.size == 0:
                    print(f"Empty crop for Track {track_id}")
                    continue

                with ocr_lock:
                    if track_id not in ocr_results:
                        threading.Thread(target=send_to_gemini, args=(cropped_img, track_id), daemon=True).start()
                    text = ocr_results.get(track_id, "")

                height_value = parse_height(text)
                if height_value is not None:
                    if height_value > truck_height_limit:
                        display_text = f"TRUCK HEIGHT {height_value:.1f}m OK"
                        color = (0, 255, 0)  # Green: Can go
                        thickness = 2
                        alert_led.off()  # Turn on LED for overheight truck
                        alert_buzzer.off()  # Turn on Buzzer for overheight truck
                    else:
                        display_text = f"ALERT! TRUCK HEIGHT {height_value:.1f}m EXCEEDS LIMIT"
                        color = (0, 0, 255)  # Red: Cannot go
                        thickness = 3
                        alert_led.on()  # Ensure LED is off
                        alert_buzzer.on()  # Ensure Buzzer is off
                else:
                    display_text = "Reading height..."
                    color = (0, 255, 255)  # Yellow: Processing
                    thickness = 2
                    alert_led.off()  # Ensure LED is off
                    alert_buzzer.off()  # Ensure Buzzer is off

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                cvzone.putTextRect(frame, display_text, (x1, y1 - 10), scale=1, thickness=1)

        cv2.imshow("RGB", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

except KeyboardInterrupt:
    print("Program terminated by user.")
except Exception as e:
    print(f"Unexpected error: {e}")

# Cleanup
alert_led.off()  # Ensure LED is off
alert_buzzer.off()  # Ensure Buzzer is off
cv2.destroyAllWindows()
picam2.close()
print("Program ended.")
