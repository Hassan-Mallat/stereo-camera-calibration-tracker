# Multi-Camera Spatial Calibration & Stereo Tracking

This repository contains the software architecture and mathematical scripts developed during an R&D project for an educational behavioral analysis system. 

It implements a complete, object-oriented computer vision pipeline for spatial calibration, epipolar geometry application, and robust 3D triangulation using ChArUco markers, featuring a custom predictive system for occlusion handling.

## ⚠️ Data Privacy Warning (GDPR)
For confidentiality and privacy reasons, **no visual data (`.mp4` files, classroom images, faces, or annotated data) is included in this repository**. 
The scripts are provided bare. To run and test this pipeline, you must provide your own video feeds and place them in the appropriate input directories.

## 🛠️ Prerequisites and Installation

Ensure you have Python 3.x installed. Then, install the required mathematical and computer vision dependencies:

```bash
pip install -r requirements.txt
```

## 📂 Project Architecture

```text
 stereo-camera-calibration-tracker
 ┣ 📂 1_Data_Preparation
 ┃ ┣  charuco_generator.py        # Generates DICT_5X5_100 and DICT_4X4_50 boards
 ┃ ┗  video_extractor.py          # Frame extraction with gc memory optimization
 ┣ 📂 2_Calibration
 ┃ ┣  calibration_undistortion.py # Computes camera matrices and distortion coeffs
 ┃ ┗  debug_aruco_detection.py    # Visual diagnostic tool for rejected markers
 ┣ 📂 3_Stereo_Tracking
 ┃ ┗  stereo_tracker.py           # Core engine: solvePnP, Epipolar snapping, 3D Triangulation & Fallback Z
 ┣ 📂 vids                          # (Empty) Place your input .mp4 files here
 ┣  requirements.txt              # Project dependencies (OpenCV, NumPy)
 ┗  README.md

