import numpy as np
import matplotlib.pyplot as plt
import cv2
import open3d as o3d
from config import *

def load_images(img_path):
    '''
        img_path: Relative path to the image directory
        return: numpy array of images present in that directory sorted in the numerical order
    '''
    image_files_names = [name for name in os.listdir(img_path)]
    image_files_names = [name.split('.')[0] for name in os.listdir(img_path)]
    image_files_names.sort()
    image_files_names = [img_path+name+'.png' for name in image_files_names]
    
    images = []
    for i in range(len(image_files_names)):
        image = cv2.imread(image_files_names[i],cv2.IMREAD_COLOR)
        images.append(image)

    images = np.array(images)
    return images

def extract_color(images):
    color_map = [] 
    for k in range(len(images)):
        color_map.append([])
        image = cv2.cvtColor(images[k],cv2.COLOR_BGR2RGB)
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                color_map[k].append(image[y,x,:])
    color_map = np.array(color_map)
    return color_map    

def create_parallax_map(images_left, images_right, numDisparities, blockSize, minDisparity):
    '''
        Return a parallax map given two stereo rectified images
    '''
    if len(images_left) != len(images_right):
        print("Error: #images_left must be equal to #images_right")
        return False

    window_size = 5
    stereo = cv2.StereoSGBM_create(minDisparity=-39,
        numDisparities=144,             # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=5,
        P1=8 * 3 * window_size ** 2,    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        P2=64 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63
    )
    
    disparity = []
    parallax_map = []

    for k in range(len(images_left)):
        disparity = stereo.compute(images_right[k],images_left[k]).astype('float32')
        disparity = (disparity - minDisparity) / numDisparities
        
        parallax_map.append([])
        for y in range(images_left[k].shape[0]):    
            for x in range(images_left[k].shape[1]):    
            
                parallax_map[k].append([x, y,disparity[y,x],1])

    # print('disparity:', disparity.shape)
    
    parallax_map = np.array(parallax_map)

    return parallax_map, disparity

def get_baseline_matrix(B, K):
    B_matrix = np.array([
        [1, 0,   0,    -K[0,2]],
        [0, 1,   0,    -K[1,2]],
        [0, 0, 0 , K[0,0] ],
        [0, 0,  -1 / B,   0 ],
    ])
    return B_matrix

def compute_jacobian(x, K, R, T, X):
    '''
    x - Nx3 - homogeneous image coordinates
    K - 3x3 - camera intrinsic matrix
    R - 3x3 - rotation matrix matrix
    T - 3x1 - translation matrix
    X - Nx4 - world coordinates
    return: J - 3*Nx12 - Jacobian matrix 
    '''

    P = np.hstack((R, T))
    for i in range(X.shape[0]):
        
        Xtemp = P @ X[i,:].T
        Jtemp = np.array(
            [[K[0,0]*X[i,0], K[0,0]*X[i,1], K[0,0]*X[i,2], K[0,0]*1, 0, 0, 0, 0, 
            K[0,0]*X[i,0]*-Xtemp[0]/Xtemp[2], K[0,0]*X[i,1]*-Xtemp[0]/Xtemp[2], K[0,0]*X[i,2]*-Xtemp[0]/Xtemp[2], K[0,0]*-Xtemp[0]/Xtemp[2]],
        [0, 0, 0, 0,K[1,1]*X[i,0], K[1,1]*X[i,1], K[1,1]*X[i,2], K[1,1]*1, 
        K[1,1]*X[i,0]*-Xtemp[1]/Xtemp[2], K[1,1]*X[i,1]*-Xtemp[1]/Xtemp[2], K[1,1]*X[i,2]*-Xtemp[1]/Xtemp[2], K[1,1]*-Xtemp[1]/Xtemp[2]]]) / Xtemp[2]

        if i == 0:           
            J = Jtemp
        else:
            J = np.vstack((J,Jtemp))

    return J

def Gauss_newton_minimization(x, K, R, T, X, learning_rate):

    '''
    x - Nx3 - homogeneous image coordinates
    K - 3x3 - camera intrinsic matrix
    R - 3x3 - rotation matrix matrix
    T - 3x1 - translation matrix
    X - Nx4 - world coordinates
    learning_rate - 1x1 - Learning rate
    return: P - 3x4 - Pose matrix  P = [R | T] 
    P does not include intrinsic matrix
    '''
    P = np.hstack((R, T))
    N = X.shape[0]
    xpred = (K @ P @ X.T).T
    xpred = xpred.T
    xpred = xpred / xpred[2,:]
    xpred = xpred.T
    xpred = xpred[:,:2]

    invK = np.linalg.inv(K)
    Error = (x[:,:2] - xpred)
    Error = np.reshape(Error,(2*N,1))
    prevError = 100000000

    for i in range(100):

        J = compute_jacobian(x, K, R, T, X)
        deltaP = np.linalg.pinv(J.T @ J) @ J.T @ Error #12 x 12 x 12 x 2*N x 2*N x 1 = 12 x 1

        P = P + learning_rate * np.reshape(deltaP,(3,4))
        R = P[:,:3]
        T = np.array([P[:,3]])
        # print(T.shape)
        T = T.T
        xpred = (K @ P @ X.T).T
        xpred = xpred.T
        xpred = xpred / xpred[2,:]
        xpred = xpred.T
        xpred = xpred[:,:2]

        Error = (x[:,:2] - xpred)
        Error = np.reshape(Error,(2*N,1))
        
        print('Mean error',Error.T @ Error / N)
        if (i != 0) & ((prevError - Error.T @ Error / N) **2 < 0.0000001):
            break
        prevError = Error.T @ Error / N

    return P

def get_point_cloud(B_matrix, parallax_map, color_map, poses):
    point_cloud = []
    point_cloud_colors = []
    # print(parallax_map.shape)
    for i in range(len(parallax_map)):

        p_map = parallax_map[i]
        # print(p_map.shape)
        mask = (p_map[:,2] > 0.05) & (p_map[:,2] <= 30)
        p_map = p_map[mask,:]
        c_map = color_map[i]
        c_map = (c_map[mask,:] / 256.0).astype('float64')
        point_cloud_colors.append(c_map)
        
        point_cloud.append(B_matrix @ p_map.T)
        point_cloud[i] = point_cloud[i] / point_cloud[i][3]

        point_cloud[i] = poses @ point_cloud[i]
        point_cloud[i] =  point_cloud[i].T

    

    registered_point_cloud = point_cloud[0]
    registered_point_cloud_colors = point_cloud_colors[0]
    for i in range(1, len(point_cloud)):
        registered_point_cloud = np.concatenate((registered_point_cloud, point_cloud[i]), axis=0)
        registered_point_cloud_colors = np.concatenate((registered_point_cloud_colors, point_cloud_colors[i]), axis=0)

    # print('point_cloud_shape',registered_point_cloud.shape)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(registered_point_cloud)
    pcd.colors = o3d.utility.Vector3dVector(registered_point_cloud_colors)
    # o3d.io.write_point_cloud("point_cloud_1_frame.ply", pcd)


    # o3d.visualization.draw_geometries([pcd])
    return registered_point_cloud, registered_point_cloud_colors, p_map

def get_poses(path_to_poses):
    poses = np.loadtxt(path_to_poses)
    poses = poses.reshape((poses.shape[0], 3, 4))
    return poses

def EstimatePose(x, X, K):

    '''
    x - N x 3 - normalized point homogenous matrix
    X - N x 4 - normalized world points
    K - 3 x 3 - camera intrinsic matrix
    returns R, T
    '''
    N = x.shape[0]
    A = np.zeros((2*N,12))
    Temp = 0
    for i in range(x.shape[0]):
        
        A[Temp] = np.array([-X[i,0], -X[i,1], -X[i,2], -1, 0, 0, 0, 0,x[i,0]*X[i,0], x[i,0]*X[i,1], x[i,0]*X[i,2], x[i,0]])
        Temp = Temp + 1
        
        A[Temp] = np.array([0, 0, 0, 0,-X[i,0], -X[i,1], -X[i,2],  -1, x[i,1]*X[i,0], x[i,1]*X[i,1], x[i,1]*X[i,2], x[i,1]])
        Temp = Temp + 1

    # print(A.shape)
    U, D, Vt = np.linalg.svd(A)

    P = Vt[Vt.shape[0] - 1,:]
    # print(P)
    P = np.reshape(P.T,(3,4))
    # P = P / P[2,3]
    # print(P)
    P = np.linalg.inv(K) @ P
    R = P[:,:3]
    U, D, Vt = np.linalg.svd(R)
    R = U @ Vt
    T = P[:,3] / D[0]
    return R, T


def get_image(point_cloud, point_cloud_colors, R, T, K, im_width, im_height, i):
    dist_coeff = np.zeros((4, 1))
    im_points, _ = cv2.projectPoints(point_cloud, R, T, K, dist_coeff)
    im_points = im_points.reshape(-1, 2).astype(np.int)

    mask = (
        (0 <= im_points[:,0]) & (im_points[:, 0] < im_width) &
        (0 <= im_points[:, 1]) & (im_points[:, 1] < im_height)
    )


    im_points = im_points[mask]
    im_colors = point_cloud_colors*255.0
    im_colors = im_colors.astype('uint8')
    im_colors = im_colors[mask]
    print(im_colors.shape)

    image = np.zeros((im_height, im_width, 3), dtype=im_colors.dtype)
    image[im_points[:, 1], im_points[:, 0]] = im_colors
    # cv2.imwrite("img.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(OP2_DIR+"/calculated_reconstructed_images"+"/calculated_reconstructed_image_{}.png".format(i), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    return image


def correct_orientation(registered_point_cloud):
    c_mat = np.array([
        [-1,0,0],
        [0,1,0],
        [0,0,-1],
    ])

    registered_point_cloud = (c_mat @ registered_point_cloud.T).T
    return registered_point_cloud

if __name__ == "__main__":

    
    images_right     = np.array([cv2.imread('./img2/0000000461.png')])
    images_left     =  np.array([cv2.imread('./img3/0000000461.png')])
    print(images_left.shape)
    poses = get_poses('./poses.txt') 

    color_map = extract_color(images_left)
    parallax_map, disp = create_parallax_map(images_left, images_right, numDisparities=144, blockSize=5, minDisparity = -39)
    print(parallax_map.shape)
    B_matrix = get_baseline_matrix(B, K)
    Point_Cloud, Point_Cloud_Colors, parallax_map = get_point_cloud(B_matrix, parallax_map, color_map, poses)
    p_cloud = Point_Cloud
    p_cloud_color = Point_Cloud_Colors

    mask = Point_Cloud[:,2] < 100
    Point_Cloud = Point_Cloud[mask,:]
    Point_Cloud_Colors = Point_Cloud_Colors[mask,:]
    parallax_map = parallax_map[mask,:]

    

    R, T = EstimatePose(parallax_map[105:165,:], Point_Cloud[105:165,:], K)
    R[:,0] /= np.linalg.norm(R[:,0])
    R[:,1] /= np.linalg.norm(R[:,1])
    R[:,2] /= np.linalg.norm(R[:,2])
    R = R
    T = np.array([T])
    T = T.T


    lr = 1
    points_considered_for_minimization = 5000
    P = Gauss_newton_minimization(np.hstack((parallax_map[105:points_considered_for_minimization,:2],np.ones((parallax_map[105:points_considered_for_minimization,:].shape[0],1)))), K, R, T, np.hstack((Point_Cloud[105:points_considered_for_minimization,:],np.ones((Point_Cloud[105:points_considered_for_minimization,:].shape[0],1)))), lr)
    R = P[:,:3]
    R[:,0] /= np.linalg.norm(R[:,0])
    R[:,1] /= np.linalg.norm(R[:,1])
    R[:,2] /= np.linalg.norm(R[:,2])

    T = - R.T @ np.array([P[:,3]]).T
    R = R.T
    print('Calculated [R | T] from camera to world:\n',np.hstack((R,T)))
    print('Actual [R | T] from camera to world:\n', np.reshape(poses[0],(3,4)))


    P = poses[1]
    R1 = P[:,:3]
    T1 = -1*(R1.T @ P[:,3])
    R1 = R1.T


    im = get_image(p_cloud, p_cloud_color, R1, T1, K, images_right[0].shape[1], images_right[0].shape[0])


         