# Smart Vision System for Assistive Guidance

A Python prototype exploring how computer vision can help visually impaired users understand nearby scenes. It combines object detection, OCR, approximate depth estimation, Arabic translation, and speech-related utilities to turn visual information into descriptions.

> **Status:** Prototype. This repository's main file defines a `SmartVisionSystem` class; it does not currently provide a ready-to-run `main()` entry point. Follow the integration notes below rather than expecting `python blind.py` to launch a complete application.

## The problem it explores

People with visual impairments may need help identifying objects, reading visible text, and understanding where objects appear in a scene. This project explores a software pipeline that can describe visual content and provide Arabic output.

## Main features

- **Object detection:** sends images to Google Cloud Vision.
- **OCR:** extracts text from supported images.
- **Depth estimation:** uses MiDaS DPT-Hybrid to estimate relative scene depth.
- **Arabic translation:** translates generated descriptions using `deep-translator`.
- **Efficiency helpers:** tracks recent detections, caches scene information, and counts selected API calls.

## Demo

No verified screenshot or recorded demo is included yet. Add your own screenshot or a short GIF/video after testing the full pipeline with a non-sensitive sample image. Do not present a mockup as a real execution result.

Suggested location: `assets/demo.gif`

## Tech stack

Python · OpenCV · NumPy · PyTorch · MiDaS · Google Cloud Vision · Pillow · deep-translator · gTTS

## Setup

Use a Python environment compatible with the pinned packages in `requirements.txt`.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```

### Configure Google Cloud Vision

The constructor requires a path to a service-account JSON file or a JSON string containing service-account credentials:

```python
from blind import SmartVisionSystem

vision_system = SmartVisionSystem("PATH_TO_LOCAL_SERVICE_ACCOUNT.json")
```

Keep the credentials file outside the repository. The snippet above initializes the system; it is **not** a full webcam or command-line demo. Inspect the public methods in `blind.py` and build an application entry point that passes images to the appropriate processing method.

## Run

`python blind.py` currently defines the class but does not launch a complete interactive application. To make the project runnable from the command line, add a small `main()` function or a separate demo script that accepts an image path, initializes credentials securely, calls the intended processing method, and displays or speaks the result.

## Security and limitations

- Never commit service-account JSON files, API keys, or private credentials.
- Images sent to cloud services may contain sensitive information; use consented test images.
- MiDaS estimates relative depth; it does not provide a guaranteed, safety-critical distance measurement.
- Runtime behavior depends on credentials, model downloads, internet access, and available hardware.

## Roadmap

- [ ] Add a tested command-line or webcam entry point.
- [ ] Include a real screenshot or short demo GIF from a successful run.
- [ ] Add sample input/output and automated tests for image processing.
- [ ] Improve configuration with environment variables and clearer error handling.
- [ ] Document latency, API usage, and limitations on representative hardware.
