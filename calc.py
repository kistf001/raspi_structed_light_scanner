import cv2
import numpy as np

# 
# pts_l - set of n 2d points in left image. nx2 numpy float array
# pts_r - set of n 2d points in right image. nx2 numpy float array
# K_l - Left Camera matrix. 3x3 numpy float array
# K_r - Right Camera matrix. 3x3 numpy float array

#   # Normalize for Esential Matrix calaculation
#   pts_l_norm = cv2.undistortPoints(np.expand_dims(pts_l, axis=1), cameraMatrix=K_l, distCoeffs=None)
#   pts_r_norm = cv2.undistortPoints(np.expand_dims(pts_r, axis=1), cameraMatrix=K_r, distCoeffs=None)
#   
#   E, mask = cv2.findEssentialMat(
#       pts_l_norm, pts_r_norm, 
#       focal=1.0, pp=(0., 0.), method=cv2.RANSAC, prob=0.999, threshold=3.0
#   )
#   
#   points, R, t, mask = cv2.recoverPose(E, pts_l_norm, pts_r_norm)

def triangulate(R,t, pts_l,pts_r, K_l,K_r):

    M_r = np.hstack((R, t))
    M_l = np.hstack((np.eye(3, 3), np.zeros((3, 1))))

    P_l = np.dot(K_l,  M_l)
    P_r = np.dot(K_r,  M_r)

    point_4d_hom = cv2.triangulatePoints(
        P_l, P_r, 
        np.expand_dims(pts_l, axis=1), np.expand_dims(pts_r, axis=1)
    )
    point_4d = point_4d_hom / np.tile(point_4d_hom[-1, :], (4, 1))
    point_3d = point_4d[:3, :].T

    return point_3d