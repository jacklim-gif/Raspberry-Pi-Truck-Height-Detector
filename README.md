# 🚛 Raspberry Pi Truck Height Detector

## Overview
An IoT system for detecting overheight trucks on Raspberry Pi 5, using:  
- **YOLOv8** for vehicle detection  
- **Google Gemini** for OCR height extraction (e.g., "4.5m")  
- **LED (GPIO 26)** and **buzzer (GPIO 17)** for hardware alerts  
- Optional **HC-SR04 ultrasonic sensor** (GPIO 23/24) for precise distance checks  

Processes live camera feed in real-time, with **green boxes for safe heights** and **red for alerts** (> user limit, e.g., 3.5m).  
Great for road safety prototypes!  

**Demo:** YouTube Video (add yours)  
**Performance:** ~15 FPS on Pi 5  

---

## 🛠️ Hardware

- Raspberry Pi 5 (4GB+ RAM)  
- Pi Camera Module (v2/v3)  
- HC-SR04 Ultrasonic Sensor (optional)  
- LED + 220Ω resistor (GPIO 26)  
- Active Buzzer + resistor (GPIO 17)  
- Jumper wires & breadboard  

### Quick Wiring
- **HC-SR04:** VCC → Pin2 (5V), GND → Pin6, Trig → GPIO23 (Pin16), Echo → GPIO24 (Pin18) via 1kΩ/2kΩ divider  
- **LED:** GPIO26 (Pin37) → Resistor → Anode → Cathode → GND (Pin39)  
- **Buzzer:** GPIO17 (Pin11) → Resistor → (+) → (-) → GND (Pin9)  

---

## 📦 Installation & Usage

### Setup Pi
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install python3-opencv python3-libcamera python3-gpiozero python3-rpi.gpio -y
pip3 install ultralytics cvzone picamera2 langchain-google-genai numpy torch
