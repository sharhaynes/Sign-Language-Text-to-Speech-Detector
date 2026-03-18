# 🤟 CaribSign

> **A real-time Barbadian Sign Language recognition system developed as a Major Research Project (COMP 3495) at the University of the West Indies, Cave Hill Campus.**

CaribSign is an AI-powered application that uses machine learning and computer vision to detect, recognise and classify Barbadian Sign Language hand gestures in real-time via a webcam. The system aims to bridge communication barriers faced by the deaf community in Barbados and the wider Caribbean, where accessible and culturally relevant assistive technologies remain limited.

---

## Demo

![CaribSign Demo](assets/demo.gif)

## 📋 Table of Contents

- [About the Project](#-about-the-project)
- [Acknowledgements](#-acknowledgement)
- [Features](#-features)
- [Technologies Used](#-technologies-used)
- [Getting Started](#-getting-started)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Results](#-results)
- [Contributors](#-contributors)
- [License](#-license)
- [Contact](#-contact)


---

## About the Project

This project was developed in response to the persistent communication barriers faced by deaf individuals in Barbados, where approximately **5,654 persons** live with some degree of hearing loss. Existing assistive technologies often lack cultural and linguistic relevance to the Caribbean context.

CaribSign addresses this by building a locally informed sign language recognition system that operates on widely available hardware.
---
## Acknowledgements

Portions of the data collection pipeline were adapted from the following tutorial and modified to suit the requirements of this project:

- [Hand Tracking Tutorial by Murtaza's Workshop](https://youtu.be/wa2ARoUUdU8) — used as a basis for the Hand Detector module and webcam capture logic
## Features

- Real-time hand gesture detection via webcam
- Supports both one-handed and two-handed signs
- Deep learning-based sign classification using a Keras model
- Tkinter GUI for easy interaction
- 21-point hand landmark tracking using MediaPipe's Hand Detector module

---

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| OpenCV | Real-time video capture and processing |
| TensorFlow / Keras (.h5) | Model integration and inference |
| Google Teachable Machine | Model training interface |
| MediaPipe | Hand landmark detection |
| Tkinter | GUI framework |

---

### Prerequisites

- Python 3.8 (you can use your own python version, but ensure the dependencies are compatible)
- pip
- A working webcam

### Installation

**1. Clone the repository**

```bash
git clone https://github.com/sharhaynes/Sign-Language-Text-to-Speech-Detector.git
cd Sign-Language-Text-to-Speech-Detector
```

**2. Install dependencies**

> ```bash
> pip install tensorflow keras opencv-python mediapipe numpy
> ```

**3. Collecting Your Own Signs**
>```bash
>python dataCollection2.py
> ```

**4. Run the GUI application**
> ```bash
> python sl_gui.py
> ```

---

## Dataset

- Collected using a laptop webcam following a structured data capture pipeline
- One-handed signs captured by pressing the **'s'** key
- Two-handed signs captured via an automatic timer
- Signs organised into labelled folders automatically upon closing the program

**Training Configuration:**

| Parameter | Value |
|---|---|
| Train / Test Split | 85% / 15% |
| Epochs | 60 |
| Batch Size | 16 |
| Learning Rate | 0.0001 |

---

##  Model Architecture

- Trained via **Google Teachable Machine** as a deep learning image classifier
- Exported as a **Keras (.h5)** model
- Integrated into the Python/Tkinter pipeline for real-time inference
- Hand landmarks extracted using **21 keypoints** per hand (joints and palm)

---

##  Results

> *(To be updated upon project completion)*

| Metric | Value |
|---|---|
| Accuracy | TBD |
| Latency | TBD |

---

##  Contributors

| Name | Role |
|---|---|
| T'Shara Haynes | Developer & Researcher |
| Dr. Adrian Als | Supervisor & Researcher |

**Institution:** University of the West Indies, Cave Hill Campus  
**Department:** Computer Science, Mathematics and Physics  
**Course:** COMP 3495 — Major Research Project in Computer Science  

---
##  License

This project is intended for academic purposes. Please contact the author before reuse.

---

##  Contact

For questions or inquiries, reach out at: `haynestshara0@gmail.com`
