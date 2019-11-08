# Declaring all Constants here§

# Importing Packages
import os
import numpy as np

# Base Directory Path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OP1_DIR = "./output_question_1"
OP2_DIR = "./output_question_2"
OP3_DIR = "./output_question_1_bonus"

# BaseLine Distance
B = 0.53790448812

# Camera Calibration Matrix
K = np.array([
        [7.070912e+02, 0.000000e+00, 6.018873e+02], 
        [0.000000e+00, 7.070912e+02, 1.831104e+02],
        [0.000000e+00, 0.000000e+00, 1.000000e+00],
])

print(BASE_DIR)