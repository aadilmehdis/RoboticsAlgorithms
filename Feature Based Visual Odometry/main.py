import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utilities import *
from load_data import *

if __name__ == "__main__": 

    # Initializing the Cumulative Transformation Matrix
    C = np.concatenate((np.eye(3), np.zeros((3,1))), axis = 1) 
    C = np.concatenate((C, np.array([[0, 0, 0, 1]])), axis = 0)

    # Opening the results file to write and writing the starting Pose
    f = open('results.txt','wb')
    np.savetxt(f,np.reshape(C[0:3,:], (1,12)))

    # Reading the image files directory
    dirFiles = os.listdir(os.path.join(DATA_PATH,'images/'))
    for i in range(len(dirFiles)):
        dirFiles[i] = dirFiles[i].split(".")[0]
    dirFiles.sort(key=float)
    for i in range(len(dirFiles)):
        dirFiles[i] = os.path.join(DATA_PATH,'images/',dirFiles[i] + '.png')

    # Reading images and extracting out the keypoints from the images and computing the matches in the consecutive image
    key_point_1 = np.zeros((800,150,3))
    key_point_2 = np.zeros((800,150,3))
    print("Extracting Keypoints and Descriptors ... ")
    for i in range(1,len(dirFiles)):
        img1 = cv2.imread(dirFiles[i-1])
        img2 = cv2.imread(dirFiles[i])

        kp1, kp2 = correspondingFeatureDetection(img1, img2)
        key_point_1[i-1] = kp1
        key_point_2[i-1] = kp2
    print("Done")

    # Running the Visual Odometry Pipeline
    for i in range(1, len(dirFiles)):
        print("Iteration {} ...".format(i))

        # Getting the Matched Keypoints
        kp1 = key_point_1[i-1]
        kp2 = key_point_2[i-1]

        # Getting the Normalization Matrix
        T1 = NormalizationMat(kp1)
        T2 = NormalizationMat(kp2)

        # Normalizing the image coordinates
        points1 = T1 @ kp1.T
        points2 = T2 @ kp2.T

        # Computing the Fundamental Matrix for the normalized points
        F = F_RANSAC(points1.T, points2.T, 0.005, 300)

        # Etraxting the Fundamental Matrix for the original points
        FundamentalMatrix = T2.T @ F @ T1

        # Estimating the Essential Matrix
        E = compute_essential_matrix(FundamentalMatrix, K)
        
        # Recovering the Pose of the camera from the Essential matrix 
        Transformation_info = cv2.recoverPose(E, (kp1[:,0:2]), (kp2[:,0:2]), K)
        Rotation = Transformation_info[1].T
        Translation = -Transformation_info[2]

        # Scaling the translation matrix as per the ground truth translation norm
        Translation = Translation * ground_truth_norm[i]

        # Constructing the Transformation Matrix for the current frame
        Transformation = np.concatenate((Rotation, Translation), axis = 1)
        Transformation = np.concatenate((Transformation, np.array([[0, 0, 0, 1]])), axis = 0)

        # Getting the Cumulative Transformation from the first frame onward
        C = C @ Transformation

        # Saving the resultant pose into the results file
        np.savetxt(f, np.reshape(C[0:3,:],(1,12)))

        print("Done")