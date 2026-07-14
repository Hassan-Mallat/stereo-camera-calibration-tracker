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
