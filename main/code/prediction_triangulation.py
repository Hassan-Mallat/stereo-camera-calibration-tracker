import cv2 as cv
import numpy as np

def build_projection_matrix(calibration_file, image_path, target_id=2, marker_size_mm=500):
    # Calculates rvec/tvec from the background image and builds the P matrix.
    with np.load(calibration_file) as data:
        camMatrix = data['camMatrix']
        distCoeff = data['distCoeff']

    half_size = marker_size_mm / 2.0
    objp = np.array([
        [-half_size,  half_size, 0],
        [ half_size,  half_size, 0],
        [ half_size, -half_size, 0],
        [-half_size, -half_size, 0]
    ], dtype=np.float32)

    img = cv.imread(image_path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
    parameters = cv.aruco.DetectorParameters()
    detector = cv.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, _ = detector.detectMarkers(gray)

    # ID LOCKING
    if ids is None:
        raise ValueError(f"No ArUco markers detected in {image_path}!")

    # Flatten the IDs array to make it a simple 1D list
    ids_flat = ids.flatten()

    # Check if our target World Origin marker is actually in the image (the aruco marker, we locked to the front one)
    if target_id not in ids_flat:
        raise ValueError(f"CRITICAL ERROR: Target ID {target_id} not found... The camera only sees these IDs: {ids_flat}")

    # Find the exact index of our target marker
    best_marker_index = np.where(ids_flat == target_id)[0][0]

    # Lock onto its corners
    master_corners = corners[best_marker_index][0]

    _, rvec, tvec = cv.solvePnP(objp, master_corners, camMatrix, distCoeff)

    R, _ = cv.Rodrigues(rvec)
    Rt = np.hstack((R, tvec))
    P = camMatrix.dot(Rt)
    
    return P, rvec, tvec, camMatrix, distCoeff


def predict_cam2_from_cam1(u1, v1, Z_known, rvec1, tvec1, K1, dist1, rvec2, tvec2, K2, dist2):
    # Predicts Cam 2 pixel using Cam 1 pixel and known Z
    point_cam1 = np.array([[[float(u1), float(v1)]]])
    undistorted = cv.undistortPoints(point_cam1, K1, dist1)
    x_norm, y_norm = undistorted[0][0]
    
    ray_cam = np.array([[x_norm], [y_norm], [1.0]])
    
    R1, _ = cv.Rodrigues(rvec1)
    R1_inv = R1.T
    ray_world = R1_inv @ ray_cam
    
    C1 = -R1_inv @ tvec1
    
    d_z = ray_world[2][0]
    C1_z = C1[2][0]
    
    if abs(d_z) < 1e-6:
        print("Ray is perfectly horizontal, cannot intersect Z-plane.")
        return None
        
    t = (Z_known - C1_z) / d_z
    
    X = C1[0][0] + t * ray_world[0][0]
    Y = C1[1][0] + t * ray_world[1][0]
    
    point_3d = np.array([[[X, Y, Z_known]]], dtype=np.float32)
    
    pixels_cam2, _ = cv.projectPoints(point_3d, rvec2, tvec2, K2, dist2)
    
    u2 = int(pixels_cam2[0][0][0])
    v2 = int(pixels_cam2[0][0][1])
    
    return u2, v2

def get_fundamental_matrix(rvec1, tvec1, K1, rvec2, tvec2, K2):
    # Calculates the Fundamental Matrix linking Cam 1 and Cam 2
    R1, _ = cv.Rodrigues(rvec1)
    R2, _ = cv.Rodrigues(rvec2)
    
    R_rel = R2 @ R1.T
    T_rel = tvec2 - (R_rel @ tvec1)
    
    Tx = np.array([
        [0, -T_rel[2][0], T_rel[1][0]],
        [T_rel[2][0], 0, -T_rel[0][0]],
        [-T_rel[1][0], T_rel[0][0], 0]
    ])
    
    E = Tx @ R_rel
    
    K2_inv = np.linalg.inv(K2)
    K1_inv = np.linalg.inv(K1)
    F = K2_inv.T @ E @ K1_inv
    
    F = F / F[2, 2]
    return F



# ==========================================
# THE 2-CAMERA EXECUTION PIPELINE
# ==========================================

print("Initializing Camera Matrices (Front-Mid and Back)...")

# Both cameras will now exclusively lock onto Marker ID 0
P1, rvec1, tvec1, camMatrix1, distCoeff1 = build_projection_matrix(
    'mid_calibration.npz', 'locking_images_middle_cam/frame_001.jpg', 
    target_id=0, marker_size_mm=500
)

P2, rvec2, tvec2, camMatrix2, distCoeff2 = build_projection_matrix(
    'back_calibration.npz', 'locking_images_back_cam/frame_028.jpg', 
    target_id=0, marker_size_mm=500
)

# Check Camera 1's physical height
R1, _ = cv.Rodrigues(rvec1)
C1_z = (-R1.T @ tvec1)[2][0]
print(f"Mathematical Front Cam Height: {C1_z:.0f} mm") 
# This should print very close to 2430 mm (real life measurements i took myself)


# Feed the RAW AI Pixel Data
u_front = 1512.0 
v_front = 746.0  
u_back = 634.0   
v_back = 255.0   

point_front_cam = np.array([[[u_front, v_front]]], dtype=np.float32)
point_back_cam = np.array([[[u_back, v_back]]], dtype=np.float32)

# Mathematically flatten the pixels
undistorted_front = cv.undistortPoints(point_front_cam, camMatrix1, distCoeff1, P=camMatrix1)
undistorted_back = cv.undistortPoints(point_back_cam, camMatrix2, distCoeff2, P=camMatrix2)

# Epipolar Snapping
print("\nSnapping pixels to perfect Epipolar geometry...")
F = get_fundamental_matrix(rvec1, tvec1, camMatrix1, rvec2, tvec2, camMatrix2)

pts1_for_corr = undistorted_front.reshape(1, 1, 2)
pts2_for_corr = undistorted_back.reshape(1, 1, 2)

new_pts1, new_pts2 = cv.correctMatches(F, pts1_for_corr, pts2_for_corr)

print(f"Original Front Pixel: {pts1_for_corr[0][0]}")
print(f"Corrected Front Pixel: {new_pts1[0][0]}")
print(f"Original Back Pixel: {pts2_for_corr[0][0]}")
print(f"Corrected Back Pixel: {new_pts2[0][0]}")

# Triangulate using the CORRECTED coordinates
point_front_ready = new_pts1.reshape(2, 1)
point_back_ready = new_pts2.reshape(2, 1)

print("\nShooting perfectly aligned Epipolar Rays...")
points_4D = cv.triangulatePoints(P1, P2, point_front_ready, point_back_ready)
points_3D = points_4D[:3, :] / points_4D[3, :]

X = float(points_3D[0, 0])
Y = float(points_3D[1, 0])
Z = float(points_3D[2, 0])

print("\n--- True Triangulated Real-World Location ---")
print(f"X: {X:.0f} mm")
print(f"Y: {Y:.0f} mm")
print(f"Z: {Z:.0f} mm (Height from floor)")




# ==========================================
# TEST PREDICTION:
# ==========================================
print("\n--- Testing Single-Camera Fallback Prediction ---")

# 1- We simulate the Back Camera failing.
# 2- We use the True Z we just calculated (approx 691 mm).
Z_known = Z 

# 3- Predict where the desk should be in the blind Back Camera
predicted_u_back, predicted_v_back = predict_cam2_from_cam1(
    u_front, v_front, Z_known, 
    rvec1, tvec1, camMatrix1, distCoeff1,  # Cam 1 (Front) Data
    rvec2, tvec2, camMatrix2, distCoeff2   # Cam 2 (Back) Data
)

print(f"Prediction for Back Camera: (X: {predicted_u_back}, Y: {predicted_v_back})")
print(f"Actual Pixel in Back Camera: (X: {u_back}, Y: {v_back})")