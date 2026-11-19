# Shadow Cam v2.0 — Privacy Virtual Camera with Modern GUI & Gradients

> [English](#english) | [Русский](#русский) | [Français](#français) | [Español](#español) | [中文](#中文) | [العربية](#العربية)

---

## Русский

**Shadow Cam v2.0** — виртуальная веб-камера с современным темным графическим интерфейсом (GUI), стильными неоновыми градиентами силуэта, оптимизацией FPS и естественным (не зеркальным) отображением.

### Ключевые возможности:
1. **Современный GUI (CustomTkinter)**:
   * Окно предпросмотра видео в реальном времени.
   * Удобные переключатели режимов, выпадающие списки и слайдер интенсивности размытия.
   * Счётчик FPS и монитор турбо-режима.
2. **Стильные неоновые градиенты**:
   * **Cyberpunk** (Неон / Фиолетовый-Циан).
   * **Sunset** (Оранжево-розовый закат).
   * **Deep Space** (Глубокий космос).
   * **Emerald** (Изумруд).
   * **Monochrome** (Серый градиент).
   * **Blood Moon** (Кровавая луна).
   * Классический **монолитный чёрный** силуэт и **размытие (Blur)**.
3. **Высокая оптимизация (Турбо-режим)**:
   * Снижает нагрузку на CPU в 4 раза за счёт масштабированного инференса нейросети и параллельных потоков ONNX Runtime.
   * Стабильные 30-60 FPS без лагов.
4. **Правильная ориентация камеры**:
   * Видеопоток по умолчанию **не зеркалится** (натуральный вид как в реальной жизни). По желанию можно включить зеркалирование тумблером в GUI.
5. **Глобальные горячие клавиши (бинды)**:
   * **`F8`** — Смена режима.
   * **`F9`** — Чёрный силуэт.
   * **`F10`** — Градиентный силуэт.
   * **`F11`** — Размытие силуэта.
   * **`F12`** — Выключить / включить цензуру.

### Запуск:
```bash
git clone https://github.com/HumanizeLegendary/shadow-cam.git
cd shadow-cam
pip install -r requirements.txt
python shadow_cam.py
```
В любых звонках (Discord, Zoom, Telegram, Google Meet) выберите камеру **OBS Virtual Camera**.

---

## English

**Shadow Cam v2.0** is an optimized real-time privacy virtual webcam featuring a dark modern GUI, aesthetic neon gradients, anti-aliasing, and low-latency Turbo Mode.

### Features:
- **Modern Dark GUI (CustomTkinter)** with real-time video preview.
- **Aesthetic Silhouette Gradients** (Cyberpunk, Sunset, Deep Space, Emerald, Monochrome, Blood Moon).
- **CPU Turbo Mode**: 4x lower CPU overhead via scaled neural inference.
- **Natural Orientation**: Camera does not mirror by default (toggleable in GUI).
- **Global Hotkeys**: `F8` to `F12` for instant switching across all windows.

### Usage:
```bash
pip install -r requirements.txt
python shadow_cam.py
```

---

## Français

**Shadow Cam v2.0** — Webcam virtuelle de confidentialité en temps réel avec interface graphique sombre, dégradés néon élégants et mode Turbo optimisé pour processeur.

---

## Español

**Shadow Cam v2.0** — Cámara virtual de privacidad en tiempo real con interfaz gráfica moderna, siluetas con degradados de color y optimización de rendimiento FPS.

---

## 中文

**Shadow Cam v2.0** — 具有现代暗黑风格 GUI、霓虹渐变剪影和低延迟 Turbo 模式的实时隐私虚拟摄像头。

---

## العربية

**Shadow Cam v2.0** — كاميرا ويب افتراضية للخصوصية مع واجهة مستخدم رسومية حديثة، تدرجات لونية، ومحرك فائق السرعة لتوفير استهلاك المعالج.
