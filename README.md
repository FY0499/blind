# Smart Vision System for Assistive Guidance

A Python-based assistive computer-vision prototype designed to describe visual scenes. The `SmartVisionSystem` class combines object detection, OCR, depth estimation, Arabic translation, and speech-related components.

## Key capabilities
- Object detection and OCR through Google Cloud Vision.
- Relative depth estimation using MiDaS (DPT-Hybrid).
- Arabic translation of detected descriptions.
- Detection history and scene caching to reduce repeated announcements.
- API usage counters for object detection and OCR calls.

## Tech stack
Python, OpenCV, NumPy, PyTorch, MiDaS, Google Cloud Vision, Pillow, deep-translator, and gTTS.

## Setup
1. Create and activate a virtual environment.
2. Install dependencies: `pip install -r requirements.txt`
3. Configure Google Cloud Vision credentials using a local service-account JSON file or a credentials JSON string supplied to `SmartVisionSystem`.
4. Ensure network access is available to load the MiDaS model on first use.
5. Review [blind.py](./blind.py) before integrating the class into an application.

## Security and privacy
Never commit service-account JSON files, API keys, or private credentials. Images sent to cloud services may contain sensitive information; use consented test images and review the provider's data-handling terms.

## Limitations and status
This is a prototype implementation. Runtime behavior depends on credentials, network access, model downloads, and available hardware. Monocular depth estimates are approximate and must not be treated as safety-critical measurements. Validate the complete pipeline on your target machine before relying on it.
