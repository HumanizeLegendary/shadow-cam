import cv2
import pyvirtualcam
import numpy as np
import onnxruntime as ort
import os
import sys
import time
import urllib.request

MODEL_FILENAME = "rvm_mobilenetv3_fp32.onnx"
MODEL_URL = "https://github.com/peterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp32.onnx"

def ensure_model(model_path):
    """
    Автоматически скачивает предобученную нейросетевую модель Robust Video Matting (RVM),
    если она отсутствует локально.
    """
    if os.path.exists(model_path) and os.path.getsize(model_path) > 10_000_000:
        return

    print(f"[ИНФО] Загрузка модели сегментации нейросети ({MODEL_FILENAME})...")
    req = urllib.request.Request(MODEL_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(model_path, "wb") as f:
        f.write(resp.read())
    print("[ИНФО] Модель успешно загружена!")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, MODEL_FILENAME)
    
    ensure_model(model_path)

    print("[ИНФО] Подключение к физической веб-камере...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ОШИБКА] Не удалось обнаружить или открыть камеру.")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0 or fps > 60:
        fps = 30

    print(f"[ИНФО] Камера захвачена: {width}x{height} @ {fps} FPS")

    # Инициализация модели в ONNX Runtime
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

    # Внутренние рекуррентные состояния памяти RVM
    r1 = np.zeros((1, 1, 1, 1), dtype=np.float32)
    r2 = np.zeros((1, 1, 1, 1), dtype=np.float32)
    r3 = np.zeros((1, 1, 1, 1), dtype=np.float32)
    r4 = np.zeros((1, 1, 1, 1), dtype=np.float32)
    downsample_ratio = np.array([0.25], dtype=np.float32)

    print("[ИНФО] Инициализация виртуальной веб-камеры...")
    try:
        with pyvirtualcam.Camera(width=width, height=height, fps=fps, fmt=pyvirtualcam.PixelFormat.BGR) as vcam:
            print(f"[УСПЕХ] Виртуальная камера запущена: '{vcam.device}'")
            print("[ИНФО] Трансляция активна. Силуэт человека заменяется на монолитный чёрный контур.")
            print("[ИНФО] Нажмите Ctrl+C для завершения.")

            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue

                frame = cv2.flip(frame, 1)

                # Подготовка кадра (RGB [0, 1])
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                src = np.transpose(rgb, (2, 0, 1))[np.newaxis, :, :, :]

                # Инференс нейросетевой альфа-маски человека
                _, pha, r1, r2, r3, r4 = session.run(None, {
                    'src': src,
                    'r1i': r1, 'r2i': r2, 'r3i': r3, 'r4i': r4,
                    'downsample_ratio': downsample_ratio
                })

                # Пороговая бинаризация и сглаживание краёв
                person_mask = (pha[0, 0] > 0.35).astype(np.float32)
                person_mask = cv2.GaussianBlur(person_mask, (5, 5), 0)[:, :, np.newaxis]

                # Заливка силуэта человека чёрным цветом при сохранении фона
                output_frame = (frame.astype(np.float32) * (1.0 - person_mask)).astype(np.uint8)

                vcam.send(output_frame)
                vcam.sleep_until_next_frame()

    except KeyboardInterrupt:
        print("\n[ИНФО] Остановка виртуальной камеры...")
    except Exception as e:
        print(f"[ОШИБКА] {e}")
    finally:
        cap.release()
        print("[ИНФО] Работа завершена.")

if __name__ == "__main__":
    main()
