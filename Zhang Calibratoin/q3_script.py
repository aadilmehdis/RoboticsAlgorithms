import numpy as np
from numpy import linalg as LA
from matplotlib import pyplot as plt
import cv2

# K Matrix 
K = np.array([
    [406.952636, 0.00000000, 366.184147],
    [0.00000000, 405.671292, 244.705127],
    [0.00000000, 0.00000000, 1.00000000],
])
# Image pixel locations of the corners of the april tags (clock-wise from top-left to bottom-right) (left, right)
x = np.array([
    [284.56243896, 149.29254150],
    [373.93179321, 128.26719666],
    [387.53588867, 220.22708130],
    [281.29962158, 241.72782898],
    [428.86453247, 114.50731659],
    [524.76373291, 92.092185970],
    [568.36596680, 180.55757141],
    [453.60995483, 205.22370911],
])

# World coordinates of the corners of the april tags
X = np.array([
    [0.0000, 0.0000, 1],
    [0.1315, 0.0000, 1],
    [0.1315, 0.1315, 1],
    [0.0000, 0.1315, 1],
    [0.2105, 0.0000, 1],
    [0.3420, 0.0000, 1],
    [0.3420, 0.1315, 1],
    [0.2105, 0.1315, 1],
])

# Constructing the M matrix
M1 = np.append(np.append(-X,np.zeros(X.shape),axis=1), X*x[:,0,None], axis=1)
M2 = np.append(np.append(np.zeros(X.shape),-X,axis=1), X*x[:,1,None], axis=1)
M  = np.zeros((16,9))
for i in range(1,8):
    M[2*i,:] = M1[i,:]
    M[2*i+1,:] = M2[i,:]

# SVD of the M Matrix
u, s, vh = LA.svd(M)

# Taking the last row of vh and reshaping it to get the homography matrix
H = vh[-1,:].reshape((3,3))
print("Homography Matrix")
print(H)

# Rechecking the corner points
reprojected_points = np.matmul(H,X.T).T

# Normalizing the reprojected points
reprojected_points[:,0] = reprojected_points[:,0] / reprojected_points[:,2]
reprojected_points[:,1] = reprojected_points[:,1] / reprojected_points[:,2]
reprojected_points[:,2] = reprojected_points[:,2] / reprojected_points[:,2]

# Decomposing to find R and T
H1 = np.matmul(LA.inv(K),H)
H1 = H1 / LA.norm(H1[:,0])

# Finding the Orthogonal Matrix closest to (h1' h2' h1'*h2')
R = np.zeros((3,3))
R[:,0] = H1[:,0]
R[:,1] = H1[:,1]
R[:,2] = np.cross(H1[:,0],H1[:,1])

u_R, s_R, vh_R = LA.svd(R)
s_R = np.diag([1,1,LA.det(np.matmul(u_R, vh_R))])

# Rotation Matrix
R = np.matmul(np.matmul(u_R, s_R), vh_R)

# Translation Matrix
T = H1[:,2]
# T = H1[:,2] / LA.norm(H1[:,0])

FMN = np.zeros((3,3))
FMN[:,0] = R[:,0]
FMN[:,1] = R[:,1]
FMN[:,2] = T

FML = np.matmul(K,FMN)

# Rechecking the corner points
reprojected_points1 = np.matmul(FML,X.T).T

# Normalizing the reprojected points
reprojected_points1[:,0] = reprojected_points1[:,0] / reprojected_points1[:,2]
reprojected_points1[:,1] = reprojected_points1[:,1] / reprojected_points1[:,2]
reprojected_points1[:,2] = reprojected_points1[:,2] / reprojected_points1[:,2]

print("Reprojected points ")
print(reprojected_points)
print("Reprojected points after decomposition")
print(reprojected_points1)
print("Rotation Matrix")
print(R)

print("Translation Vector")
print(T)

print('[r1, r2, T]')

print(np.mat([R[:,0],R[:,1],T]))
FMP = np.zeros((3,4))
FMP[:,0] = np.array([1,0,0]).T
FMP[:,1] = np.array([0,1,0]).T
FMP[:,2] = np.array([0,0,1]).T
FMP[:,3] = T
P = np.matmul(K,np.matmul(R, FMP))
# print(np.matmul(P,np.array([0,0,0,1]).T))
img = plt.imread('image.png')
plt.imshow(img)
plt.scatter(reprojected_points[:,0],reprojected_points[:,1],s=100,c='r',marker='*')
plt.scatter(reprojected_points1[:,0],reprojected_points1[:,1],s=100,c='b',marker='*')
plt.show()

#some changes
