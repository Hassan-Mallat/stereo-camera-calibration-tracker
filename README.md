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

## 📂 Project Architecture

```text
 stereo-camera-calibration-tracker
 ┣ 📂 phys-tools
 ┃ ┗  chArucoCreator.py           # Generates ChArUco and ArUco boards
 ┣ 📂 code
 ┃ ┣  calibration.py              # Computes matrices/distortion with gc memory optimization
 ┃ ┣  debugAruco.py               # Visual diagnostic tool for rejected markers
 ┃ ┣  optimized_stereo_tracker.py # Core engine: solvePnP, Epipolar snapping, 3D Triangulation
 ┃ ┣  prediction_triangulation.py # Fallback Z occlusion prediction
 ┃ ┗  video_extractor.py          # Frame extraction from video files
 ┣ 📂 vids                          # (Empty) Place your input .mp4 files here
 ┗  requirements.txt              # Project dependencies (OpenCV, NumPy)
   
##Module Guide:
phys-tools: Contains chArucoCreator.py to digitally generate the high-resolution ChArUco boards and markers ready for physical printing.

code: The main directory containing the entire computer vision pipeline:

Data Preparation: video_extractor.py extracts frames from your provided videos at a defined interval.

Calibration: calibration.py calculates the intrinsic parameters to neutralize lens distortion, utilizing the Python Garbage Collector (gc) to prevent memory leaks during heavy matrix processing. debugAruco.py acts as a visual diagnostic tool to verify marker detection quality.

Tracking & Triangulation: optimized_stereo_tracker.py is the core vectorized engine that computes extrinsic spatial positions and snaps raw pixels to Epipolar Lines for 3D triangulation. prediction_triangulation.py isolates the Fallback Z occlusion system to predict missing 2D coordinates using memorized heights.

## Author
Hassan Mallat

Computer Science & Mathematics Double Major - Université Claude Bernard Lyon 1

