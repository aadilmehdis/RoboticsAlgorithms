import numpy as np
import matplotlib.pyplot as plt
import math

plt.style.use('dark_background')

from utilities import normalize_angle, plot_covariance_ellipse, plot_data


class EKF_localizer():
    def __init__(self, X0, P0, num_landmarks, data):
        '''
            X_curr - 3x1 numpy vector - Current Position of the Robot
            P_curr - 3x3 numpy matrix - Current Covariance Matrix 
            X_l    - Locations of the landmarks
        '''
        self.X_curr         = X0
        self.X_curr_n       = X0
        self.P_curr         = P0
        self.P_curr_n       = P0
        self.num_landmarks  = num_landmarks
        self.X_l            = data['l']
        self.Q              = self.get_observation_noise(data)
        self.R              = self.get_control_noise(data)
        self.delta_time     = 0.1
        self.d              = data['d'][0,0]

    def get_control_noise(self, data):
        R_t = np.array([
            [data['v_var'][0][0],             0,                0],
            [            0, data['v_var'][0][0],                0],
            [            0,                0,data['om_var'][0][0]],
        ])  
        return R_t

    def get_observation_noise(self, data):
        Q_t_i = np.array([
            [data['r_var'][0][0],                0],
            [               0, data['b_var'][0][0]],
        ])

        Q_t = np.zeros((self.num_landmarks*2, self.num_landmarks*2))

        for i in range(self.num_landmarks):
            Q_t[2*i:2*i+2, 2*i:2*i+2] = Q_t_i

        return Q_t

    def get_motion_jacobian(self, X_t, u_t):
        G = np.eye(3)
        G[0,2] =  -self.delta_time * np.sin(X_t[2]) * u_t[0]
        G[1,2] =  self.delta_time * np.cos(X_t[2]) * u_t[0]
        
        return G

    def get_observation_jacobian(self, X_t, X_l, r_t):
        H = np.zeros((self.num_landmarks*2, 3))
        P = X_l[:,0] - X_t[0] - self.d*np.cos(X_t[2]) # x
        Q = X_l[:,1] - X_t[1] - self.d*np.sin(X_t[2]) # y
        D = P**2 + Q**2

        H[::2, 0]  =  -P * np.reciprocal(np.sqrt(D))
        H[::2, 1]  =  -Q * np.reciprocal(np.sqrt(D))
        H[::2, 2]  =   self.d * (P * np.sin(X_t[2]) - Q * np.cos(X_t[2])) * np.reciprocal(np.sqrt(D))
        H[1::2, 0] =  Q * np.reciprocal(D)
        H[1::2, 1] = -P * np.reciprocal(D)
        H[1::2, 2] = -self.d * (P * np.cos(X_t[2]) + Q * np.sin(X_t[2])) * np.reciprocal(D) - 1

        for i in range(len(r_t)):
            if r_t[i] == 0:
                H[2*i, :] = 0
                H[2*i+1, :] = 0

        return H

    def get_kalman_gain(self, P_pred, H_t):
        K_t = P_pred @ H_t.T @ np.linalg.pinv(H_t @ P_pred @ H_t.T + self.Q)
        return K_t

    def predict_motion(self, X_t, u_t):
        rotation_factor = np.array([
            [np.cos(X_t[2]), 0],
            [np.sin(X_t[2]), 0],
            [             0, 1],
        ])

        X_t_1 = X_t + self.delta_time * rotation_factor @ u_t.T
        return X_t_1

    def predict_observation(self, X_t, X_l, r_t):
        P = X_l[:,0].T - X_t[0] - self.d*np.cos(X_t[2])
        Q = X_l[:,1].T - X_t[1] - self.d*np.sin(X_t[2])

        r = np.sqrt(P**2 + Q**2)
        phi = np.arctan2(Q, P) - X_t[2]
        phi = normalize_angle(phi)

        h_t = np.concatenate((np.array([r]).T,np.array([phi]).T), axis=1)
        for i in range(len(r_t)):
            if r_t[i] == 0:
                h_t[i] = 0
        h_t = np.reshape(h_t, (self.num_landmarks*2, 1))
        return h_t
        

    def localize(self, u_t, r_t, b_t):
        
        # Prediction Step
        X_pred = self.predict_motion(self.X_curr, u_t)
        G_t = self.get_motion_jacobian(self.X_curr, u_t)        
        P_pred = G_t @ self.P_curr @ G_t.T + self.R

        # Computing the Kalman Gain
        H_t = self.get_observation_jacobian(self.X_curr, self.X_l, r_t)
        K_t = self.get_kalman_gain(P_pred, H_t)

        # Correction Step
        z_t = np.reshape(np.hstack((np.array([r_t]).T, np.array([b_t]).T)), (2*self.num_landmarks , 1))
        X_corr = X_pred + np.ravel( (K_t @ (z_t - self.predict_observation(X_pred, self.X_l, r_t))).T )
        P_corr = (np.eye(3) - K_t@H_t) @ P_pred

        # Updating Step
        self.X_curr = X_corr
        self.X_curr_n = self.predict_motion(self.X_curr_n, u_t)
        self.P_curr = P_corr
        self.P_curr_n = G_t @ self.P_curr_n @ G_t.T + self.R

        return X_corr, P_corr, self.X_curr_n, self.P_curr_n



if __name__ == '__main__':

    data = np.load('dataset.npz')

    ground_truth = np.hstack((np.hstack( (data['x_true'], data['y_true']) ) , data['th_true']))

    num_landmarks = 17
    P0 = np.array([
        [1, 0,   0],
        [0, 1,   0],
        [0, 0, 0.1],
    ])

    X0 = ground_truth[0]

    control_commands = np.concatenate(
        (data['v'], data['om']),
        axis = 1
    )

    e = EKF_localizer(X0, P0, num_landmarks, data)

    
    roboterr_data = np.array([])
    iterations = len(data['t'])
    for t in range(iterations):
        print("Iteration {}".format(t))

        u_t = control_commands[t]
        r_t = data['r'][t]
        b_t = data['b'][t]
        
        localization = e.localize(u_t, r_t, b_t)
        temp_corr = localization[0].T
        temp_pred = localization[2].T
        temp_P    = localization[1]
        temp_P_1    = localization[3]

        if t == 0:
            prediction = np.array([temp_corr])
            roboterr_data = np.array([temp_pred])
        else:
            prediction = np.vstack((prediction, np.array([temp_corr]))) 
            roboterr_data = np.vstack((roboterr_data, np.array([temp_pred]))) 


        plot_data(data['l'], ground_truth[:t+1,:], prediction, roboterr_data, temp_P, temp_P_1, t)

        print("Odometery Norm:",np.linalg.norm(ground_truth[t] - temp_pred))
        print("Corrected Norm:",np.linalg.norm(ground_truth[t] - temp_corr), '\n')

    mean_odometry_error = np.mean(np.sqrt((ground_truth[:iterations] - roboterr_data[:iterations])**2))
    mean_corrected_error = np.mean(np.sqrt((ground_truth[:iterations] - prediction[:iterations])**2))

    print("Mean Square Error in Odometry: {}".format(mean_odometry_error))
    print("Mean Square Error in EKF Correction: {}".format(mean_corrected_error))