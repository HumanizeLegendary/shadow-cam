# Shadow Cam — Виртуальная камера приватности в реальном времени

> [English](#english) | [Русский](#русский) | [Français](#français) | [Español](#español) | [中文](#中文) | [العربية](#العربية)

---

## Русский

**Shadow Cam** — приложение на Python, которое создаёт в системе виртуальную веб-камеру и в реальном времени заменяет силуэт человека монолитным чёрным контуром или размытием с помощью нейросетевой сегментации.

### Принцип работы
1. **Захват видеопотока**: Считывает кадры с физической веб-камеры через OpenCV.
2. **Сегментация силуэта**: Нейросеть **Robust Video Matting (RVM)** через ONNX Runtime в реальном времени находит контур человека (лицо, руки, тело, одежду).
3. **Обработка маски**: Сглаживает края (Anti-Aliasing) и закрашивает силуэт черным цветом, сохраняя видимым окружающий фон.
4. **Виртуальная камера**: Библиотека `pyvirtualcam` передает обработанный видеопоток в драйвер **OBS Virtual Camera**, делая его доступным в Telegram, Discord, Zoom, Google Meet и браузерах.

### Управление горячими клавишами (работают в любых окнах)
- **`F8`** — Переключение режимов по кругу (Чёрный силуэт -> Размытие -> Выкл).
- **`F9`** — Включить **чёрный силуэт**.
- **`F10`** — Включить **размытие (блюр)** силуэта.
- **`F11`** — Временно **отключить цензуру** (оригинальное видео).
- **`F12`** или **`Ctrl+C`** — Завершить работу приложения.

### Установка и запуск
1. Установите [OBS Studio](https://obsproject.com/) (для системного драйвера OBS Virtual Camera).
2. Склонируйте репозиторий и установите зависимости:
   ```bash
   git clone https://github.com/HumanizeLegendary/shadow-cam.git
   cd shadow-cam
   pip install -r requirements.txt
   ```
3. Запустите:
   ```bash
   python shadow_cam.py
   ```
   *(Веса нейросети скачаются автоматически при первом старте).*
4. В вашем приложении (Telegram, Zoom, Discord) выберите камеру **OBS Virtual Camera**.

---

## English

**Shadow Cam** is a real-time Python privacy virtual webcam that replaces your silhouette with a clean black contour or blur using deep learning segmentation.

### How It Works
1. **Video Ingestion**: Captures live frames from your physical webcam via OpenCV.
2. **Neural Matting**: Employs **Robust Video Matting (RVM)** via ONNX Runtime to segment the person (face, hands, body, and garments) in real time.
3. **Alpha Blending**: Soft Gaussian anti-aliasing fills the segmented figure with deep black while preserving the background.
4. **Virtual Camera Streaming**: Streams frames to **OBS Virtual Camera** using `pyvirtualcam` for Zoom, Discord, Telegram, Meet, and browsers.

### Hotkeys (Global)
- **`F8`**: Cycle modes (Black silhouette -> Blur -> Passthrough).
- **`F9`**: Force **Black Silhouette** mode.
- **`F10`**: Force **Blur Silhouette** mode.
- **`F11`**: **Disable censorship** (passthrough).
- **`F12`** / **`Ctrl+C`**: Quit.

### Quick Start
```bash
git clone https://github.com/HumanizeLegendary/shadow-cam.git
cd shadow-cam
pip install -r requirements.txt
python shadow_cam.py
```
Select **OBS Virtual Camera** in your video conferencing settings.

---

## Français

**Shadow Cam** est une webcam virtuelle de confidentialité en temps réel développée en Python qui remplace la silhouette humaine par une ombre noire ou un flou grâce à la segmentation par réseau de neurones.

### Raccourcis clavier globaux
- **`F8`** : Basculer entre les modes (Silhouette noire -> Flou -> Désactivé).
- **`F9`** : Activer la silhouette noire.
- **`F10`** : Activer le flou de la silhouette.
- **`F11`** : Désactiver temporairement la censure.
- **`F12`** / **`Ctrl+C`** : Quitter le programme.

### Installation
```bash
git clone https://github.com/HumanizeLegendary/shadow-cam.git
cd shadow-cam
pip install -r requirements.txt
python shadow_cam.py
```
Sélectionnez **OBS Virtual Camera** dans vos logiciels d'appel.

---

## Español

**Shadow Cam** es una cámara virtual en Python para privacidad en tiempo real que convierte la silueta del usuario en una sombra negra continua o desenfoque mediante segmentación neuronal.

### Atajos de teclado (Globales)
- **`F8`**: Cambiar modo de forma cíclica (Silueta negra -> Desenfoque -> Desactivado).
- **`F9`**: Activar silueta negra sólida.
- **`F10`**: Activar desenfoque (blur).
- **`F11`**: Desactivar censura (vídeo normal).
- **`F12`** / **`Ctrl+C`**: Salir de la aplicación.

### Instalación y uso
```bash
git clone https://github.com/HumanizeLegendary/shadow-cam.git
cd shadow-cam
pip install -r requirements.txt
python shadow_cam.py
```
Seleccione **OBS Virtual Camera** en sus aplicaciones de videollamada.

---

## 中文

**Shadow Cam** 是一款基于 Python 的实时隐私虚拟摄像头。利用神经网络抠图技术，将人像轮廓转化为纯黑色剪影或模糊效果，保护视频通话隐私。

### 全局快捷键
- **`F8`**：循环切换模式（黑色剪影 -> 模糊 -> 关闭审查）。
- **`F9`**：开启**纯黑剪影**模式。
- **`F10`**：开启**轮廓模糊**模式。
- **`F11`**：暂停审查（恢复原始画面）。
- **`F12`** 或 **`Ctrl+C`**：退出程序。

### 安装与运行
```bash
git clone https://github.com/HumanizeLegendary/shadow-cam.git
cd shadow-cam
pip install -r requirements.txt
python shadow_cam.py
```
在视频会议应用（如 Zoom、Discord、Telegram）中选择 **OBS Virtual Camera**。

---

## العربية

**Shadow Cam** هي كاميرا ويب افتراضية للخصوصية تعمل بلغة Python في الوقت الفعلي، حيث تستبدل صورة الشخص بظل أسود معتم أو تأثير ضبابي عبر التجزئة العصبية.

### مفاتيح الاختصار العامة
- **`F8`**: التبديل الدوري بين الأوضاع (ظل أسود -> ضبابي -> إيقاف).
- **`F9`**: تفعيل وضع **الظل الأسود**.
- **`F10`**: تفعيل وضع **التمويه (Blur)**.
- **`F11`**: **إيقاف الحجب** مؤقتاً (عرض الكاميرا العادية).
- **`F12`** أو **`Ctrl+C`**: إغلاق البرنامج.

### التثبيت والتشغيل
```bash
git clone https://github.com/HumanizeLegendary/shadow-cam.git
cd shadow-cam
pip install -r requirements.txt
python shadow_cam.py
```
اختر **OBS Virtual Camera** في إعدادات الكاميرا الخاصة بتطبيقك.
