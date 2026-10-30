# Smart Vision System for Assistive Guidance

A computer vision project that helps visually impaired users understand what is around them. The system analyzes images, detects nearby objects and visible text, estimates their relative distance, and provides descriptions in Arabic with audio guidance.

## How it works

The app sends an image to the system. It detects objects and text, estimates the scene depth, then prepares an Arabic description that can be delivered as audio.

## Demo

https://github.com/user-attachments/assets/2cac324e-a28c-4736-8496-2159f7d11dbb

## Features

- Object detection and text recognition using Google Cloud Vision.
- Relative depth estimation using MiDaS DPT-Hybrid.
- Arabic translation and audio output.
- Detection history and caching to reduce repeated processing.

## Technologies

Python, OpenCV, NumPy, PyTorch, MiDaS, Google Cloud Vision, Pillow, deep-translator, and gTTS.

## Setup

Install the packages from `requirements.txt`, configure Google Cloud Vision credentials, and initialize the system:

```python
from blind import SmartVisionSystem

vision_system = SmartVisionSystem("PATH_TO_SERVICE_ACCOUNT.json")
```

The project is connected to the app flow shown in the demo.
