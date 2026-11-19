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
import customtkinter as ctk
from PIL import Image

MODEL_FILENAME = "rvm_mobilenetv3_fp32.onnx"
MODEL_URL = "https://github.com/peterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp32.onnx"

# Предустановленные стили градиентов (BGR цвета)
GRADIENTS = {
    "Cyberpunk (Неон / Фиолетовый)": ((180, 20, 40), (25, 230, 240)),       # Purple -> Cyan
    "Sunset (Закат / Оранжево-розовый)": ((40, 30, 240), (180, 40, 240)),    # Orange -> Pink
    "Deep Space (Глубокий космос)": ((20, 10, 10), (140, 40, 80)),           # Dark -> Violet
    "Emerald (Изумруд)": ((40, 80, 20), (140, 230, 40)),                     # Dark green -> Bright green
    "Monochrome (Серый градиент)": ((20, 20, 20), (160, 160, 160)),          # Dark gray -> Light gray
    "Blood Moon (Кровавая луна)": ((10, 10, 20), (30, 30, 220))              # Black -> Red
}

class ShadowCamCore:
    def __init__(self):
        self.running = True
        self.mode = "gradient"  # "gradient", "black", "blur", "off"
        self.gradient_name = list(GRADIENTS.keys())[0]
        self.flip_horizontal = False  # По умолчанию не зеркалит (натуральный вид как в жизни)
        self.blur_strength = 45
        self.fast_mode = True  # Оптимизация инференса для 60 FPS
        self.current_fps = 0.0

        # Текущий кадр для GUI предпросмотра
        self.preview_frame = None
        self.lock = threading.Lock()

        # Градиентный кэш
        self._cached_grad = None
        self._cached_grad_size = (0, 0)
        self._cached_grad_name = None

    def get_gradient_canvas(self, height, width):
        if (self._cached_grad is not None and 
            self._cached_grad_size == (height, width) and 
            self._cached_grad_name == self.gradient_name):
            return self._cached_grad

        c1, c2 = GRADIENTS.get(self.gradient_name, GRADIENTS["Cyberpunk (Неон / Фиолетовый)"])
        # Линейный диагональный градиент
        x = np.linspace(0, 1, width, dtype=np.float32)[None, :]
        y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
        d = (x + y) / 2.0
        c1_arr = np.array(c1, dtype=np.float32)[None, None, :]
        c2_arr = np.array(c2, dtype=np.float32)[None, None, :]
        grad = ((1.0 - d[:, :, None]) * c1_arr + d[:, :, None] * c2_arr).astype(np.uint8)

        self._cached_grad = grad
        self._cached_grad_size = (height, width)
        self._cached_grad_name = self.gradient_name
        return grad


def ensure_model(model_path):
    if os.path.exists(model_path) and os.path.getsize(model_path) > 10_000_000:
        return
    print(f"[ИНФО] Загрузка нейросети RVM ({MODEL_FILENAME})...")
    req = urllib.request.Request(MODEL_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(model_path, "wb") as f:
        f.write(resp.read())
    print("[ИНФО] Модель загружена!")


def video_processing_worker(core, model_path):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ОШИБКА] Не удалось подключиться к физической веб-камере.")
        return

    # Задаем стабильное разрешение
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0 or fps > 60:
        fps = 30

    # Оптимизированный ONNX Runtime
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = max(2, os.cpu_count() // 2 if os.cpu_count() else 2)
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(model_path, sess_options=opts, providers=['CPUExecutionProvider'])

    # Рекуррентные состояния
    r1 = np.zeros((1, 1, 1, 1), dtype=np.float32)
    r2 = np.zeros((1, 1, 1, 1), dtype=np.float32)
    r3 = np.zeros((1, 1, 1, 1), dtype=np.float32)
    r4 = np.zeros((1, 1, 1, 1), dtype=np.float32)
    downsample_ratio = np.array([0.25], dtype=np.float32)

    # Черный холст
    black_canvas = np.zeros((height, width, 3), dtype=np.uint8)

    # Разрешение для быстрого нейросетевого инференса (снижает нагрузку на CPU в 4 раза)
    infer_w, infer_h = 480, 270

    try:
        with pyvirtualcam.Camera(width=width, height=height, fps=fps, fmt=pyvirtualcam.PixelFormat.BGR) as vcam:
            print(f"[УСПЕХ] Виртуальная камера запущена: {vcam.device}")

            frame_times = []
            while core.running:
                t0 = time.perf_counter()
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.005)
                    continue

                with core.lock:
                    flip_h = core.flip_horizontal
                    mode = core.mode
                    blur_k = core.blur_strength
                    fast = core.fast_mode

                # Зеркалирование (по умолчанию выключено — натуральный ракурс)
                if flip_h:
                    frame = cv2.flip(frame, 1)

                if mode == "off":
                    output_frame = frame
                else:
                    # Оптимизация: ресайз для мгновенного инференса без потери четкости силуэта
                    if fast:
                        small_frame = cv2.resize(frame, (infer_w, infer_h), interpolation=cv2.INTER_LINEAR)
                    else:
                        small_frame = frame

                    rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                    src = np.transpose(rgb, (2, 0, 1))[np.newaxis, :, :, :]

                    _, pha, r1, r2, r3, r4 = session.run(None, {
                        'src': src,
                        'r1i': r1, 'r2i': r2, 'r3i': r3, 'r4i': r4,
                        'downsample_ratio': downsample_ratio
                    })

                    alpha_small = pha[0, 0]
                    if fast:
                        # Возвращаем альфа-маску к исходному размеру кадра
                        alpha = cv2.resize(alpha_small, (width, height), interpolation=cv2.INTER_LINEAR)
                    else:
                        alpha = alpha_small

                    # Сглаженная бинаризация
                    mask = (alpha > 0.35).astype(np.float32)
                    mask = cv2.GaussianBlur(mask, (5, 5), 0)[:, :, np.newaxis]

                    if mode == "black":
                        # Монолитный черный силуэт
                        output_frame = (frame.astype(np.float32) * (1.0 - mask)).astype(np.uint8)
                    elif mode == "gradient":
                        # Стильный неоновый/градиентный силуэт
                        grad_canvas = core.get_gradient_canvas(height, width)
                        output_frame = (grad_canvas.astype(np.float32) * mask + 
                                        frame.astype(np.float32) * (1.0 - mask)).astype(np.uint8)
                    elif mode == "blur":
                        # Размытие силуэта
                        k = (blur_k // 2) * 2 + 1
                        blurred = cv2.GaussianBlur(frame, (k, k), 0)
                        output_frame = (blurred.astype(np.float32) * mask + 
                                        frame.astype(np.float32) * (1.0 - mask)).astype(np.uint8)
                    else:
                        output_frame = frame

                vcam.send(output_frame)
                vcam.sleep_until_next_frame()

                # Обновляем кадр для GUI превью (сжатый до 320x240 для экономии ресурсов)
                preview_small = cv2.resize(output_frame, (320, 240))
                preview_rgb = cv2.cvtColor(preview_small, cv2.COLOR_BGR2RGB)
                with core.lock:
                    core.preview_frame = preview_rgb

                # Расчет реального FPS
                t1 = time.perf_counter()
                frame_times.append(1.0 / max(1e-5, t1 - t0))
                if len(frame_times) > 20:
                    frame_times.pop(0)
                    core.current_fps = sum(frame_times) / len(frame_times)

    except Exception as e:
        print(f"[ОШИБКА ВИДЕОПОТОКА] {e}")
    finally:
        cap.release()


class ShadowCamGUI(ctk.CTk):
    def __init__(self, core):
        super().__init__()
        self.core = core

        self.title("Shadow Cam v2.0 — Privacy Virtual Camera")
        self.geometry("860x560")
        self.minsize(820, 520)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # Главный макет (Слева превью, справа панель настроек)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)
        self.grid_rowconfigure(0, weight=1)

        # ЛЕВАЯ ПАНЕЛЬ: Видео превью
        self.left_frame = ctk.CTkFrame(self, corner_radius=12)
        self.left_frame.grid(row=0, column=0, padx=15, pady=15, sticky="nsew")
        self.left_frame.grid_rowconfigure(1, weight=1)
        self.left_frame.grid_columnconfigure(0, weight=1)

        self.preview_title = ctk.CTkLabel(
            self.left_frame, 
            text="Превью виртуальной камеры (OBS Virtual Camera)", 
            font=ctk.CTkFont(size=15, weight="bold")
        )
        self.preview_title.grid(row=0, column=0, padx=10, pady=(12, 6))

        self.preview_canvas = ctk.CTkLabel(self.left_frame, text="", corner_radius=8)
        self.preview_canvas.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        self.fps_label = ctk.CTkLabel(
            self.left_frame, 
            text="FPS: -- | Загрузка: Низкая (CPU Boost ON)", 
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#38bdf8"
        )
        self.fps_label.grid(row=2, column=0, padx=10, pady=(4, 12))

        # ПРАВАЯ ПАНЕЛЬ: Управление и эффекты
        self.right_frame = ctk.CTkFrame(self, width=320, corner_radius=12)
        self.right_frame.grid(row=0, column=1, padx=(0, 15), pady=15, sticky="nsew")
        self.right_frame.grid_propagate(False)

        self.control_title = ctk.CTkLabel(
            self.right_frame, 
            text="Режим цензуры", 
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.control_title.pack(padx=15, pady=(16, 10), anchor="w")

        # Селектор режима
        self.mode_var = ctk.StringVar(value="Градиентный силуэт")
        self.mode_menu = ctk.CTkSegmentedButton(
            self.right_frame,
            values=["Градиент", "Чёрный", "Блюр", "Выкл"],
            command=self.on_mode_change
        )
        self.mode_menu.set("Градиент")
        self.mode_menu.pack(padx=15, pady=(0, 15), fill="x")

        # Выбор градиента
        self.grad_label = ctk.CTkLabel(self.right_frame, text="Цветовая палитра градиента:")
        self.grad_label.pack(padx=15, pady=(5, 4), anchor="w")

        self.grad_menu = ctk.CTkOptionMenu(
            self.right_frame,
            values=list(GRADIENTS.keys()),
            command=self.on_gradient_change
        )
        self.grad_menu.set(list(GRADIENTS.keys())[0])
        self.grad_menu.pack(padx=15, pady=(0, 15), fill="x")

        # Переключатель зеркалирования (не зеркалить по умолчанию)
        self.mirror_switch = ctk.CTkSwitch(
            self.right_frame,
            text="Отзеркалить видео (Flip X)",
            command=self.on_mirror_toggle
        )
        self.mirror_switch.pack(padx=15, pady=(5, 10), anchor="w")

        # Переключатель оптимизации FPS
        self.fast_switch = ctk.CTkSwitch(
            self.right_frame,
            text="Турбо-режим (Экономия CPU / 60 FPS)",
            command=self.on_fast_toggle
        )
        self.fast_switch.select()
        self.fast_switch.pack(padx=15, pady=(5, 15), anchor="w")

        # Слайдер силы размытия
        self.blur_label = ctk.CTkLabel(self.right_frame, text="Интенсивность блюра:")
        self.blur_label.pack(padx=15, pady=(5, 4), anchor="w")

        self.blur_slider = ctk.CTkSlider(
            self.right_frame,
            from_=15,
            to=95,
            number_of_steps=8,
            command=self.on_blur_change
        )
        self.blur_slider.set(45)
        self.blur_slider.pack(padx=15, pady=(0, 20), fill="x")

        # Информационный блок
        self.info_box = ctk.CTkTextbox(self.right_frame, height=130, corner_radius=8)
        self.info_box.pack(padx=15, pady=(0, 15), fill="both", expand=True)
        self.info_box.insert("1.0", (
            "Горячие клавиши (глобально):\n"
            "• F8  — Смена режима\n"
            "• F9  — Черный силуэт\n"
            "• F10 — Градиентный силуэт\n"
            "• F11 — Размытие / Блюр\n"
            "• F12 — Вкл/Выкл цензуру\n\n"
            "Устройство: OBS Virtual Camera"
        ))
        self.info_box.configure(state="disabled")

        # Запуск цикла обновления GUI превью
        self.update_preview()

    def on_mode_change(self, value):
        mapping = {
            "Градиент": "gradient",
            "Чёрный": "black",
            "Блюр": "blur",
            "Выкл": "off"
        }
        with self.core.lock:
            self.core.mode = mapping.get(value, "gradient")

    def on_gradient_change(self, value):
        with self.core.lock:
            self.core.gradient_name = value

    def on_mirror_toggle(self):
        with self.core.lock:
            self.core.flip_horizontal = self.mirror_switch.get() == 1

    def on_fast_toggle(self):
        with self.core.lock:
            self.core.fast_mode = self.fast_switch.get() == 1

    def on_blur_change(self, value):
        with self.core.lock:
            self.core.blur_strength = int(value)

    def update_preview(self):
        if not self.core.running:
            return

        with self.core.lock:
            img_arr = self.core.preview_frame
            fps_val = self.core.current_fps

        if img_arr is not None:
            pil_img = Image.fromarray(img_arr)
            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(460, 345))
            self.preview_canvas.configure(image=ctk_img)

        self.fps_label.configure(text=f"FPS: {fps_val:.1f} | Оптимизация: {'АКТИВНА (Турбо)' if self.core.fast_mode else 'Стандарт'}")
        self.after(33, self.update_preview)

    def on_close(self):
        with self.core.lock:
            self.core.running = False
        self.destroy()


def setup_global_hotkeys(core, gui):
    def toggle():
        with core.lock:
            modes = ["gradient", "black", "blur", "off"]
            idx = modes.index(core.mode)
            core.mode = modes[(idx + 1) % len(modes)]
            reverse_map = {
                "gradient": "Градиент",
                "black": "Чёрный",
                "blur": "Блюр",
                "off": "Выкл"
            }
            gui.mode_menu.set(reverse_map[core.mode])

    def set_black():
        with core.lock:
            core.mode = "black"
            gui.mode_menu.set("Чёрный")

    def set_gradient():
        with core.lock:
            core.mode = "gradient"
            gui.mode_menu.set("Градиент")

    def set_blur():
        with core.lock:
            core.mode = "blur"
            gui.mode_menu.set("Блюр")

    def set_off():
        with core.lock:
            core.mode = "off"
            gui.mode_menu.set("Выкл")

    try:
        keyboard.add_hotkey('f8', toggle)
        keyboard.add_hotkey('f9', set_black)
        keyboard.add_hotkey('f10', set_gradient)
        keyboard.add_hotkey('f11', set_blur)
        keyboard.add_hotkey('f12', set_off)
    except Exception as e:
        print(f"[ВНИМАНИЕ] Ошибка регистрации глобальных клавиш: {e}")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, MODEL_FILENAME)
    ensure_model(model_path)

    core = ShadowCamCore()

    # Запускаем обработку видео в отдельном потоке
    video_thread = threading.Thread(target=video_processing_worker, args=(core, model_path), daemon=True)
    video_thread.start()

    # Запускаем GUI
    app = ShadowCamGUI(core)
    setup_global_hotkeys(core, app)
    app.mainloop()

    # Завершение
    core.running = False
    video_thread.join(timeout=1.0)

if __name__ == "__main__":
    main()
