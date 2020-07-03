import pandas as pd
import numpy as np
import cv2
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose
from tensorflow.keras import backend as K
from tensorflow.keras.layers import MaxPooling2D, GlobalMaxPool2D
from tensorflow.keras.layers import concatenate, add, multiply
from tensorflow.keras.layers import Lambda, UpSampling2D, Cropping2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from tensorflow.keras.layers import Lambda, RepeatVector, Reshape
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import jaccard_score
import tensorflow
from numpy.random import seed
from tensorflow.keras.preprocessing import image
import math
import json
import random

def jaccardIndex(predMask, trueMask):
    predMask[predMask>0] = 1
    trueMask[trueMask>0] = 1
    
    pMask = np.ravel(predMask)
    tMask = np.ravel(trueMask)
    return jaccard_score(tMask, pMask)


def diceCoefficient(predMask, trueMask):
    J = jaccardIndex(predMask, trueMask)
    D = (2*J)/(1+J)
    return D

def getJaccardDiceCofficientList(jaccard_dict,dice_dict,indlist):
    jaccard_lst=[]
    dice_lst=[]
    
    for i in indlist:
        jaccard_lst.append(jaccard_dict[i])
        dice_lst.append(dice_dict[i])
        
    return jaccard_lst,dice_lst
