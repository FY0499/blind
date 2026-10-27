# Smart Vision System for Assistive Guidance

A Python prototype exploring how computer vision can help visually impaired users understand nearby scenes. It combines object detection, OCR, relative depth estimation, Arabic translation, and speech-related utilities to turn visual information into descriptions.

## Problem it addresses

People with visual impairments may need support identifying objects and reading visible text in their surroundings. This project explores a pipeline that processes an image and generates a description that can be translated into Arabic for assistive use.

> **Status:** Prototype. The repository currently defines a `SmartVisionSystem` class; it does not provide a complete ready-to-run command-line or webcam application.



https://github.com/user-attachments/assets/2cac324e-a28c-4736-8496-2159f7d11dbb



## Features

- Object detection and OCR using Google Cloud Vision.
- Relative depth estimation with MiDaS DPT-Hybrid.
- Arabic translation using `deep-translator`.
- Detection-history and caching helpers.

## Demo

A verified demo image or recording is not currently included. After testing the pipeline, add a real screenshot or short GIF under `assets/demo.gif` and embed it here. Do not use a mockup as evidence of a working run.

## Technologies

Python, OpenCV, NumPy, PyTorch, MiDaS DPT-Hybrid, Google Cloud Vision, Pillow, deep-translator, and gTTS. See [requirements.txt](./requirements.txt) for the dependency list.

## Installation

Use a Python version compatible with the packages listed in `requirements.txt`.

```bash
git clone https://github.com/FY0499/blind.git
cd blind
python -m venv .venv
```

Windows:
```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS/Linux:
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration and usage

The class requires Google Cloud Vision credentials. Keep the service-account JSON outside the repository and configure the path securely.

```python
from blind import SmartVisionSystem

vision_system = SmartVisionSystem("PATH_TO_LOCAL_SERVICE_ACCOUNT.json")
```

This snippet only initializes the class; it is not a complete demo. Inspect the public methods in [blind.py](./blind.py) and create a separate entry-point script that passes an image to the intended processing method. Until that entry point is implemented and tested, `python blind.py` should not be expected to launch an interactive application.

## Safety and limitations

- Never commit credentials or API keys.
- Use consented, non-sensitive test images when sending data to cloud services.
- MiDaS depth is an estimate, not a safety-critical distance measurement.
- Runtime behavior depends on credentials, model downloads, internet access, and hardware.

## Requirements

Dependencies are listed in [requirements.txt](./requirements.txt).

## Future improvements

- Add and test a command-line or webcam demo.
- Include a real screenshot or short demo GIF.
- Add sample input/output and automated tests.
- Improve environment-based configuration and error handling.
