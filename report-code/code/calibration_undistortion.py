
import os
import numpy as np
import matplotlib.pyplot as plt 
import cv2 as cv
import glob
import gc 

def calibrate(showPics=True):
    # 1. Reverse-Engineered Board Parameters
    squares_X = 6
    squares_Y = 8
    square_length = 0.09 #  (Exact scale doesn't matter for removing distortion, just ratio)
    marker_length = 0.07 #  
    
    # defining used dict
    dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_5X5_100)
    board = cv.aruco.CharucoBoard((squares_X, squares_Y), square_length, marker_length, dictionary)
    
    # Setup the charuco detector
    charucoDetector = cv.aruco.CharucoDetector(board)
    
    # Initialize lists to store data from all images
    allCharucoCorners = []
    allCharucoIds = []
    imageSize = None

    # Read images
    imgPathList = glob.glob('calibration_images_back_cam/*.jpg')
    print(f"Found {len(imgPathList)} images.")

    for curImgPath in imgPathList:
        imgBGR = cv.imread(curImgPath)
        imgGray = cv.cvtColor(imgBGR, cv.COLOR_BGR2GRAY)

        if imageSize is None:
            imageSize = imgGray.shape[::-1]

        # Detect the ChArUco Board
        charucoCorners, charucoIds, markerCorners, markerIds = charucoDetector.detectBoard(imgGray)

        # If we found at least 4 corners, save them
        if charucoCorners is not None and charucoIds is not None and len(charucoCorners) >= 4:
            allCharucoCorners.append(charucoCorners)
            allCharucoIds.append(charucoIds)
            
            if showPics:
                cv.aruco.drawDetectedCornersCharuco(imgBGR, charucoCorners, charucoIds, (0, 255, 0))
                cv.imshow('Charuco Board', imgBGR)
                cv.waitKey(500)
        
        # Force Python to clear RAM (optimization)
        del imgBGR
        del imgGray
        gc.collect()

    cv.destroyAllWindows()

    # Safety Check
    if len(allCharucoCorners) == 0:
        print("Error: Could not find ChArUco corners in any images. Try changing the dictionary to DICT_4X4_50.")
        return None, None

    # Format the data for the Calibration Function
    obj_points = []
    img_points = []
    for corners, ids in zip(allCharucoCorners, allCharucoIds):
        # Match the found corners in the image to their exact physical locations on the board
        objp, imgp = board.matchImagePoints(corners, ids)
        if objp is not None and len(objp) >= 4:
            obj_points.append(objp)
            img_points.append(imgp)

    # Calibrate
    repError, camMatrix, distCoeff, rvecs, tvecs = cv.calibrateCamera(obj_points, img_points, imageSize, None, None)
    
    print('\nCamera Matrix:\n\n', camMatrix)
    print("Reproj Error (Pixels): {:.4f}".format(repError))

    # Save Calibration parameters (used later)
    curFolder = os.path.dirname(os.path.abspath(__file__))
    paramPath = os.path.join(curFolder, 'back_calibration.npz')
    np.savez(paramPath,
             repError = repError,
             camMatrix = camMatrix,
             distCoeff = distCoeff,
             rvecs = rvecs,
             tvecs = tvecs)
    
    return camMatrix, distCoeff



def removeDistortion(cameraMatrix, distCoeff):
    #root = os.getcwd()
    imgPath = 'calibration_images_back_cam/frame_007.jpg'
    img = cv.imread(imgPath)

    h, w = img.shape[:2]
    newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, distCoeff, (w,h), 1, (w,h))

    # Undistort
    dst = cv.undistort(img, cameraMatrix, distCoeff, None, newCameraMatrix)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w] 
    cv.imwrite('backResult.png', dst)

    # Undistort with Remapping
    mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix, distCoeff, None, newCameraMatrix, (w,h), 5)
    dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

    # crop the image 
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv.imwrite('backResult.png', dst)

    plt.figure() 
    plt.subplot(121) 
    # Convert BGR to RGB for the original image
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB)) 
    plt.title('Original Distorted')
    #plt.imshow(img) 
    plt.subplot(122)
    # Convert BGR to RGB for the fixed image
    plt.imshow(cv.cvtColor(dst, cv.COLOR_BGR2RGB)) 
    plt.title('Undistorted (Fixed)')
    #plt.imshow(dst) 
    plt.show()


def runCalibration():
    calibrate(showPics=True)

def runRemoveDistortion():
    camMatrix, distCoeff = calibrate(showPics=False)
    removeDistortion(camMatrix, distCoeff)

if __name__ == '__main__':
    runCalibration()
    runRemoveDistortion()
