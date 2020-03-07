# -*- coding: utf-8 -*-

# Mount Google Drive
"""

from google.colab import drive
drive.mount('/content/drive')

"""# Import Libraries"""

from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Conv2D, Flatten, Dense, Activation, MaxPooling2D, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
from keras import optimizers
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
import plot_confusion_matrix
from plot_confusion_matrix import plot_confusion_matrix
import numpy as np

"""# Load Dataset"""

def read_data(directory):
    # =============================================================================
    #  This function gets a directory name and return all images in it concatenated
    #  to each other
    # =============================================================================
    data_list = glob(os.path.join('.', 'Data' + directory + '*.png'))
    data = np.asarray([cv2.imread(img,0) for img in data_list])
    return data


# read data from directory
x_train = read_data('\Train\Raw\\')
y_train = read_data('\Train\Seg\\')
x_test = read_data('\Test\Raw\\')
y_test = read_data('\Test\Seg\\')

# Change the shape to (n_clss)x(Height)x(Weight)x(channels)
x_train = (np.expand_dims(x_train, axis=3)).astype('float')
x_test = (np.expand_dims(x_test, axis=3)).astype('float')

# Change labels to categorical
y_train = (to_categorical(y_train)).astype('float')
y_test = (to_categorical(y_test)).astype('float')

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

"""# Define Hyperparameters"""

num_of_clss = 10      # number of classes
lr =          0.0001  # learning rate
beta_1 =      0.9     # beta 1 - for adam optimizer
beta_2 =      0.99    # beta 2 - for adam optimizer
epsilon =     1e-8    # epsilon - for adam optimizer
epochs =      70      # number of epochs
bs =          32      # batch size