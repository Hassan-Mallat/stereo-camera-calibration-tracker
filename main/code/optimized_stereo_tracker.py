import cv2 as cv
import numpy as np

class StereoTracker:
    def __init__(self, front_calib_npz, front_img, back_calib_npz, back_img, marker_size=500):
        print("Initializing Stereo Tracking Engine...")
        
        self.P1, self.rvec1, self.tvec1, self.K1, self.dist1 = self._build_matrix(
            front_calib_npz, front_img, marker_size, mode='largest'
        )
        self.P2, self.rvec2, self.tvec2, self.K2, self.dist2 = self._build_matrix(
            back_calib_npz, back_img, marker_size, mode='smallest'
        )
        
        self.F = self._get_fundamental_matrix(
            self.rvec1, self.tvec1, self.K1, 
            self.rvec2, self.tvec2, self.K2
        )

        # optimization: cache camera 1 world position
        R1, _ = cv.Rodrigues(self.rvec1) # cv.Rodrigues converts a 3D rotation vector to a 3 × 3 rotation matrix, or vice versa
        self.R1_inv = R1.T
        self.C1 = -self.R1_inv @ self.tvec1

        print("Engine Ready")


    def _build_matrix(self, calibration_file, image_path, marker_size_mm, mode):

        # This method calculates a camera's position relative to a physical ArUco marker,
        # effectively setting the "World Origin" (0,0,0) in the 3D space

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

        marker_areas = []
        for i in range(len(corners)):
            c = corners[i][0]
            area = np.linalg.norm(c[0] - c[1]) * np.linalg.norm(c[1] - c[2]) # calculating the Euclidean distance
            marker_areas.append((area, i))

        marker_areas.sort(key=lambda x: x[0], reverse=True)

        if mode == 'largest':
            best_marker_index = marker_areas[0][1] 
        elif mode == 'smallest':
            best_marker_index = marker_areas[-1][1] 
        else:
            best_marker_index = marker_areas[0][1]

        master_corners = corners[best_marker_index][0]

        _, rvec, tvec = cv.solvePnP(objp, master_corners, camMatrix, distCoeff) # 3D objp to the 2D master_corners
        R, _ = cv.Rodrigues(rvec)
        Rt = np.hstack((R, tvec))
        P = camMatrix.dot(Rt)
        
        return P, rvec, tvec, camMatrix, distCoeff

    def _get_fundamental_matrix(self, rvec1, tvec1, K1, rvec2, tvec2, K2):

        #This calculates the Fundamental matrix F, which ensures that a pixel in Camera 1
        # correctly corresponds to an epipolar line in Camera 2

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


    def _predict_fallback(self, points_cam1, Z_known):

        # This function is a mathematical safety net. If the back camera goes blind,
        # it uses the front camera and a known height (Z) to predict where the object should be
        
        # Format input to handle N amount of points instantly
        pts = np.array(points_cam1, dtype=np.float32).reshape(-1, 1, 2) # -1 means: "NumPy, calculate this number for me based on the amount of data I gave you"
        N = pts.shape[0] # Number of targets currently being tracked

        undistorted = cv.undistortPoints(pts, self.K1, self.dist1)
        
        # Build 3D ray matrix of shape (3, N) for bulk matrix multiplication
        '''
        The Goal: You need to shoot a 3D ray [X, Y, Z] for every single student.
        For 3 students, writing a for loop that runs 3 separate times isn't optimal.
        So we are bypassing the loop by building one giant block of memory
        '''
        ray_cam = np.ones((3, N), dtype=np.float32) # grid with 3 rows, and N columns filled with 1.0
        '''
        [ 1.0,  1.0,  1.0 ]  Row 0 (Will hold X values)
        [ 1.0,  1.0,  1.0 ]  Row 1 (Will hold Y values)
        [ 1.0,  1.0,  1.0 ]  Row 2 (Already holds Z values) in raycasting, your Z value must always start at 1.0
        '''
        ray_cam[0, :] = undistorted[:, 0, 0] # takes the X coordinates of all N students, and overwrite Row 0 of my grid
        ray_cam[1, :] = undistorted[:, 0, 1] # same for Y
        
        ray_world = self.R1_inv @ ray_cam 
        
        d_z = ray_world[2, :] 
        C1_z = self.C1[2][0]
        
        # Prevent division by zero if ray is perfectly horizontal
        d_z[np.abs(d_z) < 1e-6] = 1e-6 # safety net for Ray-Plane Intersection to prevent d_z = 0
            
        t = (Z_known - C1_z) / d_z 
        
        X = self.C1[0][0] + t * ray_world[0, :] # X and Y become arrays containing the exact 3D
        Y = self.C1[1][0] + t * ray_world[1, :] # physical coordinates of all N students simultaneously
        
        # Package bulk points for projection
        points_3d = np.zeros((N, 1, 3), dtype=np.float32)
        points_3d[:, 0, 0] = X # fill first column of that grid with x values 
        points_3d[:, 0, 1] = Y 
        points_3d[:, 0, 2] = Z_known
        
        pixels_cam2, _ = cv.projectPoints(points_3d, self.rvec2, self.tvec2, self.K2, self.dist2)
        
        # Return as a clean list of [u, v] pairs
        return pixels_cam2.reshape(-1, 2).astype(int).tolist()


    def process_student(self, points_front, points_back, known_z=None):

        """
        optimization 2: vectorization
        Accepts a single point [u, v] OR a list of multiple points [[u1,v1], [u2,v2], ...]
        Processes the entire array in a single C++ sweep without slow Python loops.
        """

        # MODE 1: Student visible in both cameras
        if points_front is not None and points_back is not None:
            # Force inputs into an (N, 1, 2) array structure
            pt_f = np.array(points_front, dtype=np.float32).reshape(-1, 1, 2)
            pt_b = np.array(points_back, dtype=np.float32).reshape(-1, 1, 2)
            N = pt_f.shape[0]

            undist_f = cv.undistortPoints(pt_f, self.K1, self.dist1, P=self.K1)
            undist_b = cv.undistortPoints(pt_b, self.K2, self.dist2, P=self.K2)

            pts1_for_corr = undist_f.reshape(1, N, 2)
            pts2_for_corr = undist_b.reshape(1, N, 2)

            new_pts1, new_pts2 = cv.correctMatches(self.F, pts1_for_corr, pts2_for_corr)

            pt_f_ready = new_pts1.reshape(2, N)
            pt_b_ready = new_pts2.reshape(2, N)

            # Triangulate all points simultaneously 
            points_4D = cv.triangulatePoints(self.P1, self.P2, pt_f_ready, pt_b_ready)
            points_3D = points_4D[:3, :] / points_4D[3, :] 

            return {
                "status": "triangulated",
                "coordinates": points_3D.T.tolist() # Returns list of [X, Y, Z]
            }

        # MODE 2: Predict Fallback
        elif points_front is not None and points_back is None and known_z is not None:
            predicted_pts = self._predict_fallback(points_front, known_z)
            return {
                "status": "predicted_cam2",
                "coordinates": predicted_pts # Returns list of [u, v]
            }
            
        # MODE 3: Target Lost
        else:
            return {"status": "target_lost"}