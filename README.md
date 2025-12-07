# Animal Classifier API

A **FastAPI-based API** for classifying images of animals into 10 classes using a **MobileNetV2 deep learning model**.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset & Classes](#dataset--classes)
- [Installation](#installation)
- [Usage](#usage)
- [Tech Stack](#tech-stack)
- [Future Improvements](#future-improvements)

---

## Project Overview
This project demonstrates an end-to-end workflow of a computer vision application:

- Training a MobileNetV2-based deep learning model to classify 10 animal classes.
- Deploying the trained model via a **FastAPI API**.
- Allowing users to upload images and receive real-time predictions in JSON format.

The API returns the predicted class, confidence score, and probabilities for all classes.

---

## Features
- Predict 10 animal classes: `cane, cavallo, elefante, farfalla, gallina, gatto, mucca, pecora, ragno, scoiattolo`.
- Detailed JSON response with class probabilities.
- Single-image upload via API.
- Fast and efficient predictions.
- Easy to deploy and extend.

---

## Dataset & Classes
The model is trained to classify images into the following **10 classes**:

| Class Index | Class Name  |
|------------|-------------|
| 0          | cane        |
| 1          | cavallo     |
| 2          | elefante    |
| 3          | farfalla    |
| 4          | gallina     |
| 5          | gatto       |
| 6          | mucca       |
| 7          | pecora      |
| 8          | ragno       |
| 9          | scoiattolo  |

---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/animal-classifier-api.git
cd animal-classifier-api
````

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Make sure the **trained model weights** file is available in the project:

```
animals10_mobilenetv2_weights.h5
```

---

## Usage

Run the FastAPI server:

```bash
uvicorn app:app --reload
```

Open your browser or use **Postman/cURL** to interact with the API.

---
## Tech Stack

* Python 3
* TensorFlow / Keras
* MobileNetV2 Pretrained Model
* FastAPI
* Pillow & NumPy

---

## Future Improvements

* Add a **frontend web interface** to upload images via browser.
* Expand dataset for higher accuracy and more classes.
* Add support for **batch predictions**.
* Dockerize the API for easy deployment.

---

