import numpy as np
import os
import pandas as pd

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import tensorflow.keras.backend as kb
import time
# import model_util
from tensorflow.keras.callbacks import CSVLogger
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def custom_loss(y_actual, y_pred):
    mask = kb.greater_equal(y_actual, 0)
    mask = tf.cast(mask, tf.float32)
    custom_loss = tf.math.reduce_sum(
        kb.square(mask*(y_actual-y_pred)))/tf.math.reduce_sum(mask)
    return custom_loss

def custom_loss_for_ramp(y_actual, y_pred):
    # Create a mask where `y_actual` is not equal to -100
    mask = kb.not_equal(y_actual, -100)
    # Cast the mask to float32
    mask = tf.cast(mask, tf.float32)
    # Use the mask to ignore the values where `y_actual` is -100
    masked_squared_error = kb.square(mask * (y_actual - y_pred))
    # Calculate the sum of the squared errors where the mask is True
    numerator = tf.math.reduce_sum(masked_squared_error)
    # Calculate the sum of the mask values (essentially the count of non-masked elements)
    denominator = tf.math.reduce_sum(mask)
    # To avoid division by zero, add a small constant to the denominator
    denominator = tf.where(tf.equal(denominator, 0), tf.constant(1, dtype=tf.float32), denominator)
    # Compute the mean squared error while ignoring the masked values
    custom_loss_value = numerator / denominator
    return custom_loss_value

def manipulate_ss_col(ss_col):
    flag = 0  # Flags: 0 -> looking for first positive, 1 -> searching for 1, 2 -> holding 1, 3 -> setting to -1, 4 -> holding 0
    counter = 0
    first_positive_found = False  # Indicator for the first positive value

    for i in range(len(ss_col)):
        if not first_positive_found:
            if ss_col[i] > 0:
                first_positive_found = True
                flag = 1  # Start looking for the next 1
            else:
                ss_col[i] = -1  # Set to -1 until the first positive value is found
        else:
            if flag == 1 and ss_col[i] == 1:
                # When 1 is found, start holding at 1
                flag = 2
                counter = 1
            elif flag == 2:
                # Hold 1 for 15 points
                if counter < 15:
                    ss_col[i] = 1
                    counter += 1
                else:
                    # Then go to -1 for the next 10 points
                    flag = 3
                    counter = 1
                    ss_col[i] = -1
            elif flag == 3:
                # Hold -1 for 10 points
                if counter < 10:
                    ss_col[i] = -1
                    counter += 1
                else:
                    # Then go back to 0 and start looking for 1 again
                    flag = 4
                    counter = 0
            elif flag == 4 and ss_col[i] != 0:
                # As soon as the value starts increasing, start looking for 1 again
                flag = 1
    
    return ss_col

class CustomError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


def get_files_to_use(root_folder, subject_nums, sides, trial_nums):
    subject_strings = convert_subject_nums_to_strings(subject_nums)
    trial_strings = convert_trial_nums_to_strings(trial_nums)
    files_to_use = []
    for subject_string in subject_strings:
        subject_folder = os.path.join(root_folder, subject_string)
        filenames = os.listdir(subject_folder)
        for side in sides:
            for trial_string in trial_strings:
                for f in filenames:
                    if side in f and trial_string in f:
                        files_to_use.append(os.path.join(subject_folder, f))
    return(files_to_use)


def convert_trial_nums_to_strings(list_of_trial_nums: int):
    mystrings = []
    for num in list_of_trial_nums:
        if num < 10:
            mystrings.append('T0' + str(num))
        else:
            mystrings.append('T' + str(num))
    return mystrings


def convert_subject_nums_to_strings(list_of_subject_nums: int):
    mystrings = []
    for num in list_of_subject_nums:
        if num < 10:
            mystrings.append('S0' + str(num))
        else:
            mystrings.append('S' + str(num))
    return mystrings



def load_file(myfile):
    stance_phase_to_get = 'TM_Stance_Phase'
    df = pd.read_csv(myfile, usecols=['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y',
                     'gyro_z', 'ankle_angle', 'ankle_velocity', 'Ramp', 'Velocity', 'TM_Is_Stance_Phase', stance_phase_to_get])
    
    # first_heel_strike_index_found = False
    
    # for i in range(len(df)):
    #     if df['TM_Is_Stance_Phase'][i] == 1 and not first_heel_strike_index_found:
    #         first_heel_strike_index, first_heel_strike_index_found = i, True
    #         break
    
    # if (not first_heel_strike_index_found): raise CustomError('Something is Wrong with the Left & Right File!!!')
    
    # df = df[first_heel_strike_index:df.index[-1]]
    # if 'LEFT' in myfile:
    #     df.insert(8, 'is_left', value=1)
    # else:
    #     df.insert(8, 'is_left', value=0)
    return df.values