# from utilities import *
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

def load_images(img_path):
    '''
        Function to load images into the main memory

        img_path    : Relative path to the image directory
        return      : numpy array of images present in that directory
                      sorted in the numerical order
    '''
    image_files_names = [name for name in os.listdir(img_path)]
    image_files_names = [name.split('.')[0] for name in os.listdir(img_path)]
    image_files_names.sort()
    image_files_names = [img_path+name+'.png' for name in image_files_names]
    
    images = []
    bar = FillingSquaresBar('Loading Images from {}'.format(img_path), max=len(image_files_names))
    for i in range(len(image_files_names)):
        image = cv2.imread(image_files_names[i])
        images.append(image)
        bar.next()
    bar.finish()

    images = np.array(images)
    return images

def extract_color(images):
    '''
        Function to get the RGB color values from the images

        images: np array of images
        return: np array of the color values extracted from the images

    '''
    color_map = [] 

    bar = FillingSquaresBar('Extracting Color Map', max=len(images))
    for k in range(len(images)):
        color_map.append([])
        image = cv2.cvtColor(images[k], cv2.COLOR_BGR2RGB)
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                color_map[k].append(image[y,x,:])
        bar.next()
    color_map = np.array(color_map)

    bar.finish()
    return color_map


def create_parallax_map(images_left, images_right):
    '''
        Return a parallax map given two stereo rectified images

        images_left: np array of the left stereo images
        images_left: np array of the right stereo images
        return:
    '''
    if len(images_left) != len(images_right):
        print("Error: #images_left must be equal to #images_right")
        return False

    window_size = 5
    minDisparity = -39
    numDisparities=144
    stereo = cv2.StereoSGBM_create(minDisparity=-39,
        numDisparities=144,
        blockSize=5,
        P1=8 * 3 * window_size ** 2,
        P2=64 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63,
        mode = 3
    )   
    
    
    disparity = []
    parallax_map = []

    bar = FillingSquaresBar('Extracting Disparity Map', max=len(images_left))
    for k in range(len(images_left)):
        im_right = cv2.cvtColor(images_right[k], cv2.COLOR_BGR2GRAY)
        im_left = cv2.cvtColor(images_left[k], cv2.COLOR_BGR2GRAY)
        disparity = stereo.compute(im_right,im_left).astype('float64')
        disparity = (disparity - minDisparity) / numDisparities
        
        parallax_map.append([])
        for y in range(disparity.shape[0]):    
            for x in range(disparity.shape[1]):    
                parallax_map[k].append([x, y,disparity[y,x],1])
        
        bar.next()
    
    parallax_map = np.array(parallax_map)

    bar.finish()
    return parallax_map, disparity

def get_baseline_matrix(B, K):
    '''
        Function to return the Baseline matrix

        B: Baseline of the stereo-camera
        K: Camera Calibration Matrix
        return: 4*4 Baseline matrix
    '''
    B_matrix = np.array([
        [1,     0,     0,    -K[0,2]],
        [0,     1,     0,    -K[1,2]],
        [0,     0,     0,     K[0,0]],
        [0,     0,  -1/B,          0],
    ])
    return B_matrix

def get_point_cloud(B_matrix, parallax_map, color_map, poses, mask_lower_bound=2):

    point_cloud = []
    point_cloud_colors = []

    bar = FillingSquaresBar('Generating Frame Point Cloud', max=len(parallax_map))
    for i in range(len(parallax_map)):

        p_map = parallax_map[i]
        mask = (p_map[:,2] > mask_lower_bound)
        p_map = p_map[mask,:]
        c_map = color_map[i]
        c_map = (c_map[mask,:] / 255.0).astype('float64')

        point_cloud_colors.append(c_map)
        
        point_cloud.append(B_matrix @ p_map.T)

        point_cloud[i] = point_cloud[i] / point_cloud[i][3]

        point_cloud[i] = poses[i] @ point_cloud[i]
        point_cloud[i] =  point_cloud[i].T

        bar.next()
    bar.finish()
    

    registered_point_cloud = point_cloud[0]
    registered_point_cloud_colors = point_cloud_colors[0]

    bar = FillingSquaresBar('Registering Global Point Cloud', max=len(point_cloud)-1)
    for i in range(1, len(point_cloud)):
        registered_point_cloud = np.concatenate((registered_point_cloud, point_cloud[i]), axis=0)
        registered_point_cloud_colors = np.concatenate((registered_point_cloud_colors, point_cloud_colors[i]), axis=0)
        bar.next()
    bar.finish()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(registered_point_cloud)
    pcd.colors = o3d.utility.Vector3dVector(registered_point_cloud_colors)


    o3d.io.write_point_cloud(OP3_DIR+"/point_cloud_{}.ply".format(datetime.datetime.now().strftime("%d-%b-%Y %H:%M:%S.%f")), pcd)

    return registered_point_cloud, registered_point_cloud_colors


def get_poses(path_to_poses):
    '''
        Function to read the poses, i.e. rotation and translation

        path_to_poses   : Relative path given to poses
        return          : N*3*4 Transformation Matrix 
    '''
    poses = np.loadtxt(path_to_poses)
    poses = poses.reshape((poses.shape[0], 3, 4)).astype('float64')
    return poses



if __name__ == "__main__":

    if os.path.exists(OP3_DIR):
        shutil.rmtree(OP3_DIR)
    os.makedirs(OP3_DIR)

    images_right     = load_images('./img2/')
    images_left     = load_images('./img3/')
    poses = get_poses('./poses.txt') 

    color_map = extract_color(images_left)

    parallax_map, disp = create_parallax_map(images_left, images_right)
    
    B_matrix = get_baseline_matrix(B, K)

    Point_Cloud, Point_Cloud_Colors = get_point_cloud(B_matrix, parallax_map, color_map, poses)
    print('Outputs saved in output_question_1_bonus')
