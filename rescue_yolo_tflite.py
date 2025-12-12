#!/usr/bin/env python3
"""
rescue_yolo_tflite.py
- MJPEG stream from ESP32-CAM
- Detect person using a TFLite YOLO nano model (generic parser)
- Decide STILL vs MOVING and display on a 16x2 I2C LCD
- Robust to common TFLite output variants (boxes+scores+classes OR single tensor exports)
"""

import cv2, time, numpy as np
from collections import deque
from smbus2 import SMBus
import argparse

try:
    import tflite_runtime.interpreter as tflite
except Exception:
    # fallback to tensorflow
    import tensorflow as tf
    tflite = tf.lite

# -------- CONFIG ----------
STREAM_URL = "http://192.168.4.1:81/stream"   # <<== change to your ESP32-CAM stream URL
TFLITE_MODEL = "yolov8n.tflite"               # <<== change if different
I2C_BUS = 1
I2C_ADDR = 0x27       # <<== change to your i2cdetect result
FRAME_W, FRAME_H = 320, 320  # adjust to model input (many YOLOs use 320 or 416)
FRAME_SKIP = 2
CONF_TH = 0.35
MOVEMENT_WINDOW = 30
MOVEMENT_PIX_THRESHOLD = 5
SMOOTH_LEN = 6
DISPLAY_UPDATE_RATE = 0.5
PERSON_CLASS_ID = 0  # many YOLO exports map person -> 0; adjust if your model uses COCO indexing (person=0 for some exports, =1 or 15 for others)
# --------------------------

# Minimal I2C LCD (PCF8574) driver (4-bit)
LCD_BACKLIGHT = 0x08
ENABLE = 0b00000100
LCD_CLR = 0x01
LCD_HOME = 0x02

class I2CLcd:
    def __init__(self, i2c_bus=1, address=0x27):
        self.bus = SMBus(i2c_bus)
        self.addr = address
        self.backlight = LCD_BACKLIGHT
        time.sleep(0.05)
        self._init_lcd()

    def _write_byte(self, data):
        self.bus.write_byte(self.addr, data | self.backlight)

    def _pulse(self, data):
        self._write_byte(data | ENABLE)
        time.sleep(0.0005)
        self._write_byte(data & ~ENABLE)
        time.sleep(0.0001)

    def _write4(self, nibble, mode=0):
        self._write_byte(nibble | mode)
        self._pulse(nibble | mode)

    def send_command(self, cmd):
        self._write4(cmd & 0xF0)
        self._write4((cmd << 4) & 0xF0)

    def send_data(self, data):
        self._write4(data & 0xF0, mode=0x01)
        self._write4((data << 4) & 0xF0, mode=0x01)

    def _init_lcd(self):
        self._write_byte(0x00)
        time.sleep(0.05)
        self._write4(0x30); time.sleep(0.005)
        self._write4(0x30); time.sleep(0.0002)
        self._write4(0x30); time.sleep(0.0002)
        self._write4(0x20); time.sleep(0.0002)
        self.send_command(0x20 | 0x08)
        self.send_command(0x08 | 0x04)
        self.clear()
        self.send_command(0x04 | 0x02)
        time.sleep(0.01)

    def clear(self):
        self.send_command(LCD_CLR); time.sleep(0.002)

    def write(self, text, line=0):
        if line == 0:
            addr = 0x80
        else:
            addr = 0xC0
        self.send_command(addr)
        for ch in text.ljust(16)[:16]:
            self.send_data(ord(ch))

# ---------------- TFLITE helpers ----------------
def build_interpreter(model_path):
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def preprocess_frame(frame, w, h):
    img = cv2.resize(frame, (w, h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return img

def run_tflite(interpreter, img_in):
    inp_details = interpreter.get_input_details()
    out_details = interpreter.get_output_details()
    # set input
    tensor = np.expand_dims(img_in, axis=0).astype(np.float32)
    interpreter.set_tensor(inp_details[0]['index'], tensor)
    interpreter.invoke()
    outputs = [interpreter.get_tensor(o['index']) for o in out_details]
    return outputs

def parse_tflite_outputs(outputs, conf_th, img_w, img_h):
    """
    Try to parse typical TFLite YOLO exports:
    - outputs may be [boxes, scores, classes, num] or a single array with [N,6] or others.
    Returns list of detections: (x1,y1,x2,y2,score,class)
    """
    dets = []
    # Common variant: outputs = [boxes, scores, classes, num]
    if len(outputs) >= 3:
        # Try to match shapes
        shapes = [o.shape for o in outputs]
        # locate a boxes-like output (N,4)
        boxes = None; scores = None; classes = None
        for o in outputs:
            if o.ndim == 3 and o.shape[2] == 4:
                boxes = o[0]
                continue
            if o.ndim == 2 and o.shape[1] == 4:
                boxes = o
                continue
            if o.ndim == 2 and o.shape[1] == 6:
                arr = o
                boxes = arr[:,0:4]
                scores = arr[:,4]
                classes = arr[:,5].astype(np.int32)
                continue
        # fallback: any (N,4)
        if boxes is None:
            for o in outputs:
                if o.ndim == 2 and o.shape[1] >= 4:
                    boxes = o[:,0:4]
                    if o.shape[1] >= 5:
                        scores = o[:,4]
                    if o.shape[1] >= 6:
                        classes = o[:,5].astype(np.int32)
                    break
        # try find scores/classes separately
        if scores is None:
            for o in outputs:
                if o.ndim == 2 and (o.shape[1] == 1 or o.shape[1] == 2):
                    scores = o[0].reshape(-1)
                    break
                if o.ndim == 1:
                    # single vector could be scores or classes
                    if scores is None:
                        scores = o
        if classes is None:
            for o in outputs:
                if o.ndim == 1 and o.dtype in (np.int32, np.int64):
                    classes = o.astype(np.int32)
                    break
        if boxes is None:
            return []
        boxes = np.array(boxes)
        if scores is None:
            scores = np.zeros(len(boxes))
        if classes is None:
            classes = np.zeros(len(boxes), dtype=np.int32)
        for i, b in enumerate(boxes):
            sc = float(scores[i])
            cl = int(classes[i])
            if sc < conf_th: continue
            # boxes either normalized y1,x1,y2,x2 or absolute
            if np.max(b) <= 1.01:
                y1,x1,y2,x2 = b
                x1i = int(x1 * img_w); y1i = int(y1 * img_h)
                x2i = int(x2 * img_w); y2i = int(y2 * img_h)
            else:
                # assume xyxy
                x1i, y1i, x2i, y2i = int(b[0]), int(b[1]), int(b[2]), int(b[3])
            dets.append((x1i,y1i,x2i,y2i,sc,cl))
        return dets

    # If single output [1,N,6] or [N,6] format
    for out in outputs:
        if out.ndim == 3 and out.shape[2] >= 5:
            arr = out[0]
            for row in arr:
                # row could be [x,y,w,h,score,class] or [y1,x1,y2,x2,score,class]
                if row[4] < conf_th: continue
                vals = row[:6]
                if np.max(vals[:4]) <= 1.01:
                    # normalized y1,x1,y2,x2
                    y1,x1,y2,x2 = vals[:4]
                    x1i = int(x1 * img_w); y1i = int(y1 * img_h)
                    x2i = int(x2 * img_w); y2i = int(y2 * img_h)
                else:
                    x1i,y1i,x2i,y2i = int(vals[0]), int(vals[1]), int(vals[2]), int(vals[3])
                dets.append((x1i,y1i,x2i,y2i,float(vals[4]),int(vals[5])))
            return dets

    # Nothing matched
    return []

# ---------------- image fallback ----------------
def contour_fallback(frame, small_w=FRAME_W, small_h=FRAME_H):
    frame_small = cv2.resize(frame, (small_w, small_h))
    gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.count_nonzero(th) > (th.size // 2):
        th = cv2.bitwise_not(th)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area > 700:
            pts = c.reshape(-1,2)
            centroid = pts.mean(axis=0).astype(int)
            cx,cy = int(centroid[0]), int(centroid[1])
            return cx,cy,0.35
    return small_w//2, small_h//2, 0.0

# ---------------- main ----------------
def main():
    print("Loading TFLite model:", TFLITE_MODEL)
    interpreter = build_interpreter(TFLITE_MODEL)
    input_details = interpreter.get_input_details()
    in_shape = input_details[0]['shape']
    in_h, in_w = int(in_shape[1]), int(in_shape[2])
    print("Model input:", in_w, "x", in_h)

    cap = cv2.VideoCapture(STREAM_URL)
    if not cap.isOpened():
        print("ERROR: Cannot open stream. Check STREAM_URL.")
        return

    # LCD init
    try:
        lcd = I2CLcd(I2C_BUS, I2C_ADDR)
        lcd.write("Rescue: TFLite", 0)
        lcd.write("Initializing...", 1)
    except Exception as e:
        print("LCD init error:", e)
        lcd = None

    smoothing = deque(maxlen=SMOOTH_LEN)
    movement_hist = deque(maxlen=MOVEMENT_WINDOW)
    last_display = 0.0
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("WARN: no frame, retrying")
            time.sleep(0.3)
            continue
        frame_id += 1
        if frame_id % FRAME_SKIP != 0:
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue

        # run inference
        img_in = preprocess_frame(frame, in_w, in_h)
        outputs = run_tflite(interpreter, img_in)
        dets = parse_tflite_outputs(outputs, CONF_TH, in_w, in_h)

        if dets:
            # filter persons
            persons = [d for d in dets if int(d[5]) == PERSON_CLASS_ID or (PERSON_CLASS_ID==0 and int(d[5])==0)]
            if persons:
                p = sorted(persons, key=lambda x: -x[4])[0]
                x1,y1,x2,y2,sc,cl = p
                cx = int((x1 + x2) / 2); cy = int((y1 + y2) / 2)
                conf = sc
                method = "YOLO-TFLITE"
            else:
                cx,cy,conf = contour_fallback(frame, FRAME_W, FRAME_H)
                method = "FALLBACK"
        else:
            cx,cy,conf = contour_fallback(frame, FRAME_W, FRAME_H)
            method = "FALLBACK"

        smoothing.append((cx,cy))
        avg = np.mean(np.array(smoothing), axis=0).astype(int)
        sx,sy = int(avg[0]), int(avg[1])

        movement_hist.append((sx,sy))
        if len(movement_hist) >= 2:
            pts = np.array(movement_hist)
            diffs = np.linalg.norm(pts - pts.mean(axis=0), axis=1)
            movement_metric = float(np.mean(diffs))
        else:
            movement_metric = 0.0

        if conf < 0.2:
            line1 = "No person"
            line2 = "Searching..."
        else:
            if movement_metric < MOVEMENT_PIX_THRESHOLD:
                line1 = "Person: ALIVE"
                line2 = "Status: STILL"
            else:
                line1 = "Person: ALIVE"
                line2 = "Status: MOVING"

        print(f"[{time.strftime('%H:%M:%S')}] {method} Conf:{conf:.2f} Move:{movement_metric:.2f} -> {line2}")

        ts = time.time()
        if lcd and (ts - last_display) > DISPLAY_UPDATE_RATE:
            try:
                lcd.write(line1[:16], 0)
                lcd.write(line2[:16], 1)
            except Exception as e:
                print("LCD write error:", e)
            last_display = ts

        # optional debug window
        cv2.imshow("RescueStream", cv2.resize(frame, (640,480)))
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
