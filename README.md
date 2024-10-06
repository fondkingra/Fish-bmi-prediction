# Fish BMI Calculator with YOLOv8

## Overview

The Fish BMI Calculator is a tool that utilizes the YOLOv8 deep learning model to detect fish in video footage. It computes and displays the height, width, and Body Mass Index (BMI) of each detected fish based on bounding box dimensions. The output is an annotated video that overlays these metrics on the detected fish.

## Features

- **Real-time Fish Detection:** Accurately identify fish using the YOLOv8 model.
- **Metric Calculation:** Calculate height, width, and BMI for each detected fish.
- **Annotated Output Video:** Visualize the calculated metrics directly on the video frames.

## Table of Contents

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Example Output](#example-output)
5. [License](#license)
6. [Acknowledgments](#acknowledgments)
7. [Contact](#contact)

## Requirements

Before running the application, ensure you have the following installed:

- **Python** (version 3.6 or higher)
- **OpenCV**
- **NumPy**
- **Ultralytics YOLO**

You can install the necessary libraries using pip:

```bash
pip install opencv-python numpy ultralytics
