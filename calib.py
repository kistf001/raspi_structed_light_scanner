import cv2
import numpy as np
import image_load

def get_chassboard_Corner(img):

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 21, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

    for img_ in img:

        #
        gray = img_
        
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (7,6),None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            #img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)
        
    return np.array(objpoints), np.array(imgpoints)

def get_camera_matrix(objpoints,imgpoints,gray):

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray[0].shape[::-1],None,None)
    
    return ret, mtx, dist, rvecs, tvecs

def get_rotation_transpose(pts_l,pts_r,K_l,K_r):
    
    # Normalize for Esential Matrix calaculation
    pts_l_norm = cv2.undistortPoints(
        np.expand_dims(pts_l, axis=1), cameraMatrix=K_l, distCoeffs=None)
    pts_r_norm = cv2.undistortPoints(
        np.expand_dims(pts_r, axis=1), cameraMatrix=K_r, distCoeffs=None)

    #
    E, mask = cv2.findEssentialMat(
        pts_l_norm, pts_r_norm, 
        focal=1.0, pp=(0., 0.), method=cv2.RANSAC, prob=0.999, threshold=3.0
    )

    #
    points, R, t, mask = cv2.recoverPose(E, pts_l_norm, pts_r_norm)

    return R, t

if __name__ == "__main__":

    img = image_load.chassboard(0)
    objpoints,imgpoints = get_chassboard_Corner(img)
    ret, mtx, dist, rvecs, tvecs = get_camera_matrix(objpoints,imgpoints,img)
    print(ret, mtx, dist, rvecs, tvecs)

    img = image_load.chassboard(1)
    objpoints,imgpoints = get_chassboard_Corner(img)
    ret, mtx, dist, rvecs, tvecs = get_camera_matrix(objpoints,imgpoints,img)
    print(ret, mtx, dist, rvecs, tvecs)
