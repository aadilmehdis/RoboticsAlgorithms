import numpy as np
import matplotlib.pyplot as plt
import cv2
import os


def correspondingFeatureDetection(img1, img2):
    '''
        img1        : Image 1
        img2        : Image 2
        return      : Matched Keypoints from the 1st and 2nd image
    '''

    orb = cv2.ORB_create()
    Keypoints1, Descriptors1 = orb.detectAndCompute(img1, None)
    Keypoints2, Descriptors2 = orb.detectAndCompute(img2, None)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(Descriptors1,Descriptors2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    kp1_list = np.mat([])
    kp2_list = np.mat([])
    k = 0

    number_of_matches = 150

    for m in matches:
        img1Idx = m.queryIdx
        img2Idx = m.trainIdx

        (img1x, img1y) = Keypoints1[img1Idx].pt
        (img2x, img2y) = Keypoints2[img2Idx].pt

        if k == 0:
            kp1_list = [[img1x,img1y,1]]
            kp2_list = [[img2x,img2y,1]]
            k = 1
        else:
            kp1_list = np.append(kp1_list,[[img1x,img1y,1]],axis = 0)
            kp2_list = np.append(kp2_list,[[img2x,img2y,1]],axis = 0)
            k+=1
        if k == number_of_matches:
            break
    return kp1_list,kp2_list

def F_matrix(image_coords_1, image_coords_2):
    '''
    image_coords_1  : N*3 homogeneous coordinates of image pixels    
    image_coords_2  : N*3 homogeneous coordinates of image pixels
    return          : Fundamental Matrix of dimension 3*3
    '''
    
    A = np.zeros((len(image_coords_1),9))

    # Constructing A matrix of size N*9
    for i in range(len(image_coords_1)):
        A[i,:] = np.kron(image_coords_1[i,:], image_coords_2[i,:])
        
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    F = np.reshape(vh[8,:], (3,3))
    uf, sf, vhf = np.linalg.svd(F, full_matrices=True)
    
    F = uf @ np.diag(np.array([sf[0], sf[1], 0])) @ vhf

    return F

def F_RANSAC(image_points_1, image_points_2, threshold, n_iters):
    '''
        image_points_1  : N*3 matrix of a normalized 2D homogeneous of image 1 
        image_points_2  : N*3 matrix of a normalized 2D homogeneous of image 2
        threshold       : Inlier threshold
        n_iters         : Number of Iterations 
        return          : Fundamental Matrix of dimension 3*3
    '''

    n = 8 
    F = np.zeros((3,3))
    max_inliers = -9999999

    for i in range(n_iters):

        # Randomly sample 8 matching points from image 1 and image 2
        indices = np.random.choice(image_points_1.shape[0], n, replace=False)  
        matched_points_1 = image_points_1[indices] 
        matched_points_2 = image_points_2[indices]

        # Get the F matrix for the current iteration
        F_current = F_matrix(matched_points_1, matched_points_2)

        # Counting the number of inliers 
        number_current_inliers = 0
        for i in range(len(image_points_1)):
            error = np.abs(image_points_2[i,:] @ F_current @ image_points_1[i,:].T)
            if error < threshold:
                number_current_inliers += 1

        # Updating F matrix
        if number_current_inliers > max_inliers:
            max_inliers = number_current_inliers
            F = F_current
    return F


def NormalizationMat(image_coords):
    '''
        image_coords : N*3 matrix of image coordinates
        return       : 3*3 Scale Transformation matrix 
    '''

    mu = np.mean(image_coords,axis = 0)
    d = 0
    for i in range(len(image_coords)):
        d = d + np.sqrt((image_coords[i,0] - mu[0])**2 + (image_coords[i,1] - mu[1])**2)

    d = d/i
    T = np.mat([
        [1.44/d, 0, -1.44 * mu[0]/d], 
        [0, 1.44/d, -1.44 * mu[1]/d],
        [0,      0,               1]
        ])
    
    return T




def algebraic_triangulation(x1, x2, P1, P2):
    '''
        x1      : N*3 image coordinates of the 1st image
        x2      : N*3 image coordinates of the 2st image
        P1      : Projection matrix of the 1st image
        P2      : Projection matrix of the 2nd image
        return  : N*4 Triangulated world point
    '''

    X = np.zeros((len(x1),4))

    for i in range(len(x1)):
        J = np.zeros((4,4))
        J[:,0] = x1[i,0] * P1[2,:] - P1[0,:]
        J[:,1] = x1[i,1] * P1[2,:] - P1[1,:]
        J[:,2] = x2[i,0] * P2[2,:] - P2[0,:]
        J[:,3] = x2[i,1] * P2[2,:] - P2[1,:]

        u, s, vh = np.linalg.svd(J, full_matrices=False)
        X[i,:] = vh[3,:]
        X[i,:] = X[i,:] / X[i,3]
    
    return X
    

def decompose_essential_matrix(E, K, img_points1, img_points2, K_inverse):
    '''
        E           : 3*3 Essential Matrix
        K           : 3*3 Camera Calibration Matrix
        img_points1 : N*3 Image Coordinates of the 1st image
        img_points2 : N*3 Image Coordinates of the 2nd image
        K_inverse   : 3*3 Inverse of Camera Calibration Matrix
        return      : 3*3 Rotation and 3*1 Translation Parameters
    '''

    R1, R2, T = cv2.decomposeEssentialMat(E)

    P =  K @ np.concatenate((np.eye(3),np.zeros((3,1))),axis = 1)
    P1 = K @ np.concatenate((R1,T),axis=1)
    P2 = K @ np.concatenate((R1,-T),axis=1)
    P3 = K @ np.concatenate((R2,T),axis=1)
    P4 = K @ np.concatenate((R2,-T),axis=1)

    X_P1 = algebraic_triangulation(img_points1, img_points2, P, P1)
    X_P2 = algebraic_triangulation(img_points1, img_points2, P, P2)
    X_P3 = algebraic_triangulation(img_points1, img_points2, P, P3)
    X_P4 = algebraic_triangulation(img_points1, img_points2, P, P4)


    
    # Computing Image Coordinates for all the Triangulated Points in Image 1 and Image 2
    x1_1 = K_inverse @ P @ X_P1.T
    x2_1 = K_inverse @P1 @ X_P1.T

    x1_2 =K_inverse @ P @ X_P2.T
    x2_2 =K_inverse @ P2 @ X_P2.T

    x1_3 = K_inverse @ P @ X_P3.T
    x2_3 = K_inverse @ P3 @ X_P3.T

    x1_4 = K_inverse @ P @ X_P4.T
    x2_4 = K_inverse @ P4 @ X_P4.T

    # Computing the depth of all the reprojected image points
    d1_1 =  x1_1[2,:]
    d2_1 =  x2_1[2,:]
    score_1 = (d1_1 > 0) & (d2_1 > 0)
    score_1 = np.sum(score_1)

    d1_2 =  x1_2[2,:]
    d2_2 =  x2_2[2,:]
    score_2 = (d1_2 > 0) & (d2_2 > 0)
    score_2 = np.sum(score_2)

    d1_3 =  x1_3[2,:]
    d2_3 =  x2_3[2,:]
    score_3 = (d1_3 > 0) & (d2_3 > 0)
    score_3 = np.sum(score_3)

    d1_4 =  x1_4[2,:]
    d2_4 =  x2_4[2,:]
    score_4 = (d1_4 > 0) & (d2_4 > 0)
    score_4 = np.sum(score_4)

    # Selecting the Projection Matrix that gives the maximum number of reconstructed points in front of the camera
    index = np.argmax(np.array([score_1, score_2, score_3, score_4]))
    rotation = np.mat([])
    translation = np.mat([])
    if index == 0:
        rotation = R1
        translation = T
        Pactual = P1
    elif index == 1:
        rotation = R1
        translation = -T
        Pactual = P2
    elif index == 2:
        rotation = R2
        translation = T
        Pactual = P3        
    elif index == 3:
        rotation = R2
        translation = -T
        Pactual = P4

    return rotation, translation 



def compute_essential_matrix(F, K):
    '''
        F       : 3*3 Fundamental Matrix
        K       : 3*3 Camera Calibration Matrix
        return  : 3*3 Essential Matrix
    '''
    E = K.T @ F @ K
    u, s, vh = np.linalg.svd(E, full_matrices=True)
    E = u @ np.diag(np.array([1,1,0])) @ vh
    return E