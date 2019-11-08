import numpy as np
import matplotlib.pyplot as plt
import math

def normalize_angle(theta):
    for i in range(len(theta)):
        new_theta =  theta[i]
        while new_theta <= -math.pi:
            new_theta += 2*math.pi
        while new_theta > math.pi:
            new_theta -= 2*math.pi

        theta[i] = new_theta
    return theta


def plot_covariance_ellipse(X_est, P_est, color, label):
    X_est = X_est.T
    Pxy = P_est[0:2, 0:2]
    eigval, eigvec = np.linalg.eig(Pxy)

    if eigval[0] >= eigval[1]:
        BI = 0
        SI = 1
    else:
        BI = 1
        SI = 0

    t = np.arange(0, 2 * math.pi + 0.1, 0.1)
    a = math.sqrt(eigval[BI])
    b = math.sqrt(eigval[SI])
    x = [a * math.cos(it) for it in t]
    y = [b * math.sin(it) for it in t]
    angle = math.atan2(eigvec[BI, 1], eigvec[BI, 0])
    rot = np.array([[math.cos(angle), math.sin(angle)],
                    [-math.sin(angle), math.cos(angle)]])
    fx = rot @ (np.array([x, y]))
    px = np.array(fx[0, :] + X_est[0, 0]).flatten()
    py = np.array(fx[1, :] + X_est[1, 0]).flatten()
    plt.plot(px, py, ls="--", color=color, label=label)

def plot_data(landmark_pos, ground_truth, prediction_data, bot_error, temp_P, temp_P_1, time):
    plt.cla()
    plt.plot(ground_truth[:,0], ground_truth[:,1], color='chartreuse',label = 'ground truth')
    plt.plot(ground_truth[-1,0], ground_truth[-1,1], color='limegreen', marker="o", label='ground')

    plt.plot(prediction_data[:,0], prediction_data[:,1], color='lavenderblush', label = 'corrected path factoring Kalman gain')
    plt.plot(prediction_data[-1,0], prediction_data[-1,1], color='slateblue', marker="o", label='corrected factoring Kalman gain')

    # plt.plot(bot_error[:,0], bot_error[:,1], color='darkcyan',label = 'odometry path')    
    # plt.plot(bot_error[-1,0], bot_error[-1,1], color='paleturquoise', marker="o", label='odometry')

    plt.plot(landmark_pos[:,0], landmark_pos[:,1], 'rX')

    plot_covariance_ellipse(np.array([prediction_data[-1]]), temp_P, "cadetblue", "Covariance Ellipse - EKF Correction")
    # plot_covariance_ellipse(np.array([bot_error[-1]]), temp_P_1, "pink", "Covariance Ellipse - Normal Odometry")

    plt.legend()
    plt.title("EKF Localization | Time Step: {}".format(time))
    plt.axis("equal")
    plt.grid(True)
    plt.pause(0.001)
    if time > 12600:
        plt.savefig("./plot/OdometerVSGround")