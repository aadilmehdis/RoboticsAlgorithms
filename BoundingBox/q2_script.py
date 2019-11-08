import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
x = [[835], [305], [1]]
K = np.mat(([7.2153e+02,0,6.0955e+02],[0,7.2153e+02,1.7285e+02],[0,0,1]))
B = np.dot(np.linalg.inv(K),x)

h = 1.65
groundNormal = [[0,1,0]]

WorldPoint = h*B/np.dot(groundNormal,B)

d2point = np.matmul(K,WorldPoint)
# print(WorldPoint)
# print(d2point)

carHeight = 1.38
carWidth = 1.51
carLength = 4.10


#5 degree rotation along y clockwise

Rot = np.mat(([np.cos(5*np.pi/180),0,-np.sin(5*np.pi/180)],[0,1,0],[ np.sin(5*np.pi/180),0,np.cos(5*np.pi/180)]))
 
# KITTI dataset y is downwards x is right and z is forward (into the image)
WorldPoint2 = np.matmul(Rot,[[0],[0],[carLength]]) + WorldPoint
WorldPoint3 = np.matmul(Rot,[[carWidth],[0],[0]]) + WorldPoint
WorldPoint4 = np.matmul(Rot,[[0],[-carHeight],[0]]) + WorldPoint
WorldPoint5 = np.matmul(Rot,[[carWidth],[-carHeight],[0]]) + WorldPoint
WorldPoint6 = np.matmul(Rot,[[0],[-carHeight],[carLength]]) + WorldPoint
WorldPoint7 = np.matmul(Rot,[[carWidth],[0],[carLength]]) + WorldPoint
WorldPoint8 = np.matmul(Rot,[[carWidth],[-carHeight],[carLength]]) + WorldPoint

WorldPoints = np.array((WorldPoint,WorldPoint2,WorldPoint3,WorldPoint4,WorldPoint5,WorldPoint6,WorldPoint7,WorldPoint8))

# print(K)
# print(WorldPoints[0])
# print(np.matmul(K,WorldPoint + WorldPoint2 ))
# print(np.matmul(K,WorldPoint + WorldPoint3 ))
# print(K[2][:])

pic = plt.imread("image.png")
plt.imshow(pic)
 
edges = np.array([1, 3, 5, 4, 1, 2, 6, 8, 7, 3, 5, 8, 6, 4, 6, 2, 7]) 
for i in range(8):
    d2point = np.matmul(K,WorldPoints[i])
    d2point = d2point/d2point[2]
    plt.plot(d2point[0],d2point[1],'bo')

# print(K)    
linemat = np.mat(((),(),()))
for i in range(edges.size):
    tmpmat = np.matmul(np.mat(K),np.mat(WorldPoints[edges[i] - 1]))
    tmpmat = tmpmat/tmpmat[2]
    linemat = np.hstack((linemat,tmpmat))

# print(linemat[0])    

plt.plot(np.transpose(linemat[0]),np.transpose(linemat[1]),linewidth="2",c='r')    

plt.show()
