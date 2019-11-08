import numpy as np

DATA_PATH = '../mr19-assignment2-data/'
K  = np.array([
    [7.215377e+02, 0.000000e+00, 6.095593e+02],
    [0.000000e+00, 7.215377e+02, 1.728540e+02],
    [0.000000e+00, 0.000000e+00, 1.000000e+00]
    ])

K_inverse = np.linalg.inv(K)

ground_truth_norm = np.loadtxt('ground-truth-norm.txt')

if __name__ == "__main__":
    f = open('ground-truth-norm.txt','wb')
    ground_truth = np.loadtxt('../mr19-assignment2-data/ground-truth.txt')
    ground_truth_translation = np.concatenate((np.concatenate((np.array([ground_truth[:,3]]).T, np.array([ground_truth[:,7]]).T), axis=1), np.array([ground_truth[:,11]]).T ), axis=1)
    for i in range(800, 0, -1):
        ground_truth_translation[i] = ground_truth_translation[i] - ground_truth_translation[i-1]
    ground_truth_norm = np.linalg.norm(ground_truth_translation, axis=1)
    np.savetxt(f, ground_truth_norm)

