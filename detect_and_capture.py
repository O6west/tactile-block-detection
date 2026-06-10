from ultralytics import YOLO
import cv2
import csv
import os
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# ── 설정 ──────────────────────────────────────────────
MODEL_PATH       = r"C:\Users\LucaL\tensorflow_finalproject\braille_best_s_v2.pt"
VIDEO_PATH       = "http://10.138.212.126:8080/video"  # IP Webcam 실시간
LOCATION         = "경기 안산시 상록구 사동 1621"
OUTPUT_DIR       = r"C:\Users\LucaL\tensorflow_finalproject\captures"
CSV_PATH         = r"C:\Users\LucaL\tensorflow_finalproject\detection_log.csv"
CONF_THRESHOLD   = 0.5
CAPTURE_INTERVAL = 3  # 동일 장면 중복 캡쳐 방지 (초)
# ──────────────────────────────────────────────────────

def draw_korean(img, text, pos, size=32, color=(0, 255, 0)):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", size)
    except:
        font = ImageFont.load_default()
    draw.text(pos, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)
print("클래스 목록:", model.names)

DAMAGED_CLASS_IDX = 0  # 'damage' 클래스
print(f"교체폐기 클래스 인덱스: {DAMAGED_CLASS_IDX} → '{model.names[DAMAGED_CLASS_IDX]}'")

with open(CSV_PATH, 'w', newline='', encoding='utf-8-sig') as f:
    csv.writer(f).writerow(['파일명', '탐지시각', '위치', '신뢰도', '탐지수'])

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise ConnectionError(f"스트림에 연결할 수 없습니다: {VIDEO_PATH}")

capture_count    = 0
prev_capture_sec = -CAPTURE_INTERVAL
frame_count      = 0

print("\n실시간 탐지 시작... (종료: q)\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("스트림 끊김 — 재연결 시도 중...")
        cap = cv2.VideoCapture(VIDEO_PATH)
        continue

    frame_count += 1
    current_sec = frame_count / 30.0
# YOLOv8 추론
    results = model(frame, conf=CONF_THRESHOLD, verbose=False)[0]
    damaged = [b for b in results.boxes if int(b.cls) == DAMAGED_CLASS_IDX]

    annotated = results.plot()
    annotated = draw_korean(annotated, f"위치: {LOCATION}", (10, 10))

    cv2.imshow("점자블록 탐지", annotated)

    if damaged and (current_sec - prev_capture_sec) >= CAPTURE_INTERVAL:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename  = f"capture_{capture_count+1:03d}_{timestamp}.jpg"
        cv2.imwrite(os.path.join(OUTPUT_DIR, filename), annotated)

        max_conf = max(float(b.conf) for b in damaged)
        with open(CSV_PATH, 'a', newline='', encoding='utf-8-sig') as f:
            csv.writer(f).writerow(
                [filename, timestamp, LOCATION, f"{max_conf:.3f}", len(damaged)]
            )

        capture_count    += 1
        prev_capture_sec  = current_sec
        print(f"[캡쳐 {capture_count:03d}] {filename}  신뢰도: {max_conf:.3f}  탐지수: {len(damaged)}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\n── 완료 ──────────────────────────────")
print(f"총 캡쳐: {capture_count}장  →  {OUTPUT_DIR}")
print(f"기록 파일: {CSV_PATH}")


# 1. YOLOv8 추론
results = model(frame, conf=0.5)[0]
damaged = [b for b in results.boxes if int(b.cls) == DAMAGED_CLASS_IDX]

# 2. 자동 캡쳐 (3초 중복 방지)
if damaged and (current_sec - prev_capture_sec) >= 3:
    cv2.imwrite(filename, annotated)

# 3. CSV 로그 기록
csv.writer(f).writerow([filename, timestamp, LOCATION, max_conf, len(damaged)])


