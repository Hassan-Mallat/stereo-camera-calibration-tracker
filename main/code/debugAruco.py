import cv2 as cv
import numpy as np

def debug_aruco_detection(image_path):
    # Load the image
    img = cv.imread(image_path)
    if img is None:
        print("Error: Could not load image path.")
        return
        
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Define the EXACT dictionary you printed
    aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_5X5_100)
    parameters = cv.aruco.DetectorParameters()
    

    detector = cv.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, rejected = detector.detectMarkers(gray)

    # VISUALIZE THE RESULTS
    print(f"Detected Markers: {len(corners)}")
    print(f"Rejected Candidates: {len(rejected)}")

    # Draw Green lines around valid markers
    if len(corners) > 0:
        cv.aruco.drawDetectedMarkers(img, corners, ids, borderColor=(0, 255, 0))
    
    # Draw Red lines around rejected squares
    if len(rejected) > 0:
        cv.aruco.drawDetectedMarkers(img, rejected, borderColor=(0, 0, 255))

    # Shrink the image so it fits on your laptop screen to view
    h, w = img.shape[:2]
    img_resized = cv.resize(img, (int(w*0.6), int(h*0.6)))

    cv.imshow('ArUco Debugger', img_resized)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':

    debug_aruco_detection('frame_038.jpg')