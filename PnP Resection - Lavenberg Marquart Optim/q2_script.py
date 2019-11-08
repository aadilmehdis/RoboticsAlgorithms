from utilities_2 import *
import os
import shutil
import datetime
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
from config import *
import open3d as o3d
import matplotlib.pyplot as plt
from progress.bar import FillingSquaresBar

if __name__ == "__main__":

    if os.path.exists(OP2_DIR):
        shutil.rmtree(OP2_DIR)
    os.makedirs(OP2_DIR)
    os.makedirs(OP2_DIR+"/calculated_reconstructed_images")

    images_right     = load_images('./img2/')
    images_left     = load_images('./img3/')
    poses = get_poses('./poses.txt') 
    print('Calculating Pose for all 21 images')
    # print(len(images_left))
    for i in range(len(images_left) - 1):
        print(images_left[i].shape)
        color_map = extract_color(np.array([images_left[i]]))
        parallax_map, disp = create_parallax_map(np.array([images_left[i]]), np.array([images_right[i]]), numDisparities=144, blockSize=5, minDisparity = -39)
        B_matrix = get_baseline_matrix(B, K)
        print(parallax_map.shape)
        Point_Cloud, Point_Cloud_Colors, parallax_map = get_point_cloud(B_matrix, parallax_map, color_map, poses[i])
    
    
        p_cloud = Point_Cloud
        p_cloud_color = Point_Cloud_Colors

        mask = Point_Cloud[:,2] < 400
        Point_Cloud = Point_Cloud[mask,:]
        Point_Cloud_Colors = Point_Cloud_Colors[mask,:]
        parallax_map = parallax_map[mask,:]
        print(parallax_map.shape)

        mask = np.random.randint(0, parallax_map.shape[0],50)
        R, T = EstimatePose(parallax_map[mask,:], Point_Cloud[mask,:], K)
        R[:,0] /= np.linalg.norm(R[:,0])
        R[:,1] /= np.linalg.norm(R[:,1])
        R[:,2] /= np.linalg.norm(R[:,2])
        R = R
        T = np.array([T])
        T = T.T


        lr = 0.1
        points_considered_for_minimization = 5000
        P = Gauss_newton_minimization(np.hstack((parallax_map[105:points_considered_for_minimization,:2],np.ones((parallax_map[105:points_considered_for_minimization,:].shape[0],1)))), K, R, T, np.hstack((Point_Cloud[105:points_considered_for_minimization,:],np.ones((Point_Cloud[105:points_considered_for_minimization,:].shape[0],1)))), lr)
        R = P[:,:3]
        R[:,0] /= np.linalg.norm(R[:,0])
        R[:,1] /= np.linalg.norm(R[:,1])
        R[:,2] /= np.linalg.norm(R[:,2])

        T = - R.T @ np.array([P[:,3]]).T
        R = R.T
        print('Calculated [R | T] from camera to world:\n',np.hstack((R,T)))
        print('Actual [R | T] from camera to world:\n', np.reshape(poses[i],(3,4)))


        P = poses[1]
        R1 = P[:,:3]
        T1 = -1*(R1.T @ P[:,3])
        R1 = R1.T


        im = get_image(p_cloud, p_cloud_color, R1, T1, K, images_right[0].shape[1], images_right[0].shape[0], i)


    print('Q2 outputs stored in output_question_2 folder. Folder contains all reconstructed images from our estimated pose.')    
