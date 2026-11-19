import cv2
import pyvirtualcam
import numpy as np
import onnxruntime as ort
import os
import sys
import time
import urllib.request
import threading
import keyboard

MODEL_FILENAME = "rvm_mobilenetv3_fp32.onnx"
MODEL_URL = "https://github.com/peterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp32.onnx"

class AppState:
    def __init__(self):
        # Режимы: 'black' (черный силуэт), 'blur' (размытие силуэта), 'off' (оригинал без цензуры)
        self.mode = 'black'
        self.running = True
        self.lock = threading.Lock()

    def toggle_mode(self):
        with self.lock:
            modes = ['black', 'blur', 'off']
            curr_idx = modes.index(self.mode)
            self.mode = modes[(curr_idx + 1) % len(modes)]
            mode_names = {
                'black': 'Черный силуэт (Black Silhouette)',
                'blur': 'Размытие силуэта (Blur Silhouette)',
                'off': 'Цензура отключена (Passthrough / Off)'
            }
            print(f"\n[ГОРЯЧАЯ КЛАВИША] Режим переключен: {mode_names[self.mode]}")

    def set_black(self):
        with self.lock:
            self.mode = 'black'
            print("\n[ГОРЯЧАЯ КЛАВИША] Активирован режим: Черный силуэт")

    def set_blur(self):
        with self.lock:
            self.mode = 'blur'
            print("\n[ГОРЯЧАЯ КЛАВИША] Активирован режим: Размытие силуэта")

    def set_off(self):
        with self.lock:
            self.mode = 'off'
            print("\n[ГОРЯЧАЯ КЛАВИША] Активирован режим: Цензура выключена")

    def stop(self):
        with self.lock:
            self.running = False
            print("\n[ГОРЯЧАЯ КЛАВИША] Выход из программы...")

state = AppState()

def setup_hotkeys():
    try:
        # F8: Циклическое переключение режимов
        keyboard.add_hotkey('f8', state.toggle_mode)
        # F9: Сплошной черный силуэт
        keyboard.add_hotkey('f9', state.set_black)
        # F10: Размытие силуэта
        keyboard.add_hotkey('f10', state.set_blur)
        # F11: Выключение цензуры
        keyboard.add_hotkey('f11', state.set_off)
        # F12: Выход
        keyboard.add_hotkey('f12', state.stop)
    except Exception as e:
        print(f"[ВНИМАНИЕ] Не удалось зарегистрировать глобальные горячие клавиши: {e}")

def ensure_model(model_path):
    if os.path.exists(model_path) and os.path.getsize(model_path) > 10_000_000:
        return

    print(f"[ИНФО] Загрузка нейросетевой модели ({MODEL_FILENAME})...")
    req = urllib.request.Request(MODEL_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(model_path, "wb") as f:
        f.write(resp.read())
    print("[ИНФО] Модель успешно загружена!")

def print_banner():
    print("=" * 60)
    print("      SHADOW-CAM: ВИРТУАЛЬНАЯ КАМЕРА ЦЕНЗУРЫ СИЛУЭТА")
    print("=" * 60)
    print(" Управление горячими клавишами (работают в любых окнах):")
    print("  [F8]  - Смена режима по кругу (Черный -> Блюр -> Выкл)")
    print("  [F9]  - Включить ЧЕРНЫЙ силуэт")
    print("  [F10] - Включить РАЗМЫТИЕ (Blur) силуэта")
    print("  [F11] - ВРЕМЕННО ОТКЛЮЧИТЬ цензуру (Оригинал)")
    print("  [F12] или [Ctrl+C] - Завершение работы")
    print("=" * 60)

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, MODEL_FILENAME)
    
    ensure_model(model_path)
    print_banner()

    print("[ИНФО] Подключение к реальной веб-камере...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ОШИБКА] Не удалось открыть физическую камеру.")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0 or fps > 60:
        fps = 30

    print(f"[ИНФО] Параметры камеры: {width}x{height} @ {fps} FPS")

    # Инициализация модели в ONNX Runtime
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

    # Состояния нейросети
    r1 = np.zeros((1, 1, 1, 1), dtype=np.float32)
    r2 = np.zeros((1, 1, 1, 1), dtype=np.float32)
    r3 = np.zeros((1, 1, 1, 1), dtype=np.float32)
    r4 = np.zeros((1, 1, 1, 1), dtype=np.float32)
    downsample_ratio = np.array([0.25], dtype=np.float32)

    setup_hotkeys()

    print("[ИНФО] Запуск виртуальной веб-камеры в системе...")
    try:
        with pyvirtualcam.Camera(width=width, height=height, fps=fps, fmt=pyvirtualcam.PixelFormat.BGR) as vcam:
            print(f"[УСПЕХ] Виртуальная камера активна: '{vcam.device}'")
            print("[ИНФО] Трансляция началась. Нажмите горячие клавиши для смены режимов.")

            while state.running:
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue

                frame = cv2.flip(frame, 1)

                with state.lock:
                    curr_mode = state.mode

                if curr_mode == 'off':
                    # Без цензуры
                    vcam.send(frame)
                    vcam.sleep_until_next_frame()
                    continue

                # Подготовка кадра (RGB [0, 1]) для нейросети
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                src = np.transpose(rgb, (2, 0, 1))[np.newaxis, :, :, :]

                # Инференс нейросети RVM
                _, pha, r1, r2, r3, r4 = session.run(None, {
                    'src': src,
                    'r1i': r1, 'r2i': r2, 'r3i': r3, 'r4i': r4,
                    'downsample_ratio': downsample_ratio
                })

                # Маска силуэта человека
                person_mask = (pha[0, 0] > 0.35).astype(np.float32)
                person_mask = cv2.GaussianBlur(person_mask, (7, 7), 0)[:, :, np.newaxis]

                if curr_mode == 'black':
                    # Заливка монолитным черным цветом
                    output_frame = (frame.astype(np.float32) * (1.0 - person_mask)).astype(np.uint8)
                elif curr_mode == 'blur':
                    # Сильное размытие силуэта
                    blurred_all = cv2.GaussianBlur(frame, (51, 51), 30)
                    output_frame = (blurred_all.astype(np.float32) * person_mask + 
                                    frame.astype(np.float32) * (1.0 - person_mask)).astype(np.uint8)
                else:
                    output_frame = frame

                vcam.send(output_frame)
                vcam.sleep_until_next_frame()

    except KeyboardInterrupt:
        print("\n[ИНФО] Завершение работы...")
    except Exception as e:
        print(f"[ОШИБКА] {e}")
    finally:
        cap.release()
        print("[ИНФО] Камера освобождена.")

if __name__ == "__main__":
    main()
