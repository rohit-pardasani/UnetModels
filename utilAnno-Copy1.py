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

def strToContourPts(arrlist):
    json_str = json.loads(arrlist) 
    x=json_str['all_points_x']
    y=json_str['all_points_y']
    
    N = len(x)
    #display(N)
    C = np.zeros((N,2), dtype=np.int32)
    #display(C)
    C[:,0]=x
    C[:,1]=y
    #display(C)
    
    return C


def annoToBothMasks(dfS,dfN,nameIndS,path,fanno):
    df=pd.read_csv(fanno,header=0)
    for i in range(dfN):
        fname = path + df.loc[dfS+i,'filename']
        img = cv2.imread(fname,0)
        height,width = np.shape(img)
        pts = strToContourPts(df.loc[dfS+i,'region_shape_attributes'])
        
        mask = np.zeros((height, width), np.uint8)
        cv2.polylines(mask, [pts], isClosed = True, color=(255,255,255),thickness=2)
        fsave = path+'mask_'+str(nameIndS+1+i)+'.png'
        cv2.imwrite(fsave,mask)
        
        fmask = np.zeros((height, width), np.uint8)
        cv2.fillPoly(fmask, [pts], color=(255,255,255))
        fsave = path+'slice_'+str(nameIndS+1+i)+'_fmask.png'
        cv2.imwrite(fsave,fmask)



def annoToContourMask(dfS,dfN,nameIndS,path,fanno):
    df=pd.read_csv(fanno,header=0)
    for i in range(dfN):
        fname = path + df.loc[dfS+i,'filename']
        img = cv2.imread(fname,0)
        height,width = np.shape(img)
        pts = strToContourPts(df.loc[dfS+i,'region_shape_attributes'])
        mask = np.zeros((height, width), np.uint8)
        cv2.polylines(mask, [pts], isClosed = True, color=(255,255,255),thickness=2)
        fsave = path+'mask_'+str(nameIndS+1+i)+'.png'
        cv2.imwrite(fsave,mask)

def annoToFilledMask(dfS,dfN,nameIndS,height,width,path,fanno):
    df=pd.read_csv(fanno,header=0)
    for i in range(dfN):
        fname = path + df.loc[dfS+i,'filename']
        img = cv2.imread(fname,0)
        height,width = np.shape(img)
        pts = strToContourPts(df.loc[dfS+i,'region_shape_attributes'])
        mask = np.zeros((height, width), np.uint8)
        cv2.fillPoly(mask, [pts], color=(255,255,255))
        fsave = path+'slice_'+str(nameIndS+1+i)+'_fmask.png'
        cv2.imwrite(fsave,mask)

def createMappingFile(N,trainPath):
    #trainPathhead = '../GEData/Fractional Limb Data/Data50_DirectPng/'
    #trainPathlimb = '../GEData/Fractional Limb Data/Data75/'
    flst = []
    mlst = []
    clst = []

    '''
    flst = glob.glob(trainPathhead+'*HC.png')
    flst.sort()
    mlst = glob.glob(trainPathhead+'*_fmask.png')
    mlst.sort()
    clst = glob.glob(trainPathhead+'*_Annotation.png')
    clst.sort()
    #print(mlst)
    #print(clst)
    '''

    for i in range(1,N+1,1):
        flst.append(trainPath+'slice_'+str(i)+'.png')
        mlst.append(trainPath+'slice_'+str(i)+'_fmask.png')
        clst.append(trainPath+'mask_'+str(i)+'.png')    
    df = pd.DataFrame(list(zip(flst, mlst,clst)), columns =['slice', 'mask','contour'])
    #
    #
    h = []
    w = []
    maxValF = []
    maxValM = []
    maxValC = []
    minValF = []
    minValM = []
    minValC = []
    typeImg = []


    for i in range(len(df)):
        imgF = cv2.imread(df.loc[i,'slice'],0)
        imgM = cv2.imread(df.loc[i,'mask'],0)
        imgC = cv2.imread(df.loc[i,'contour'],0)
        if(np.shape(imgF) != np.shape(imgM) or np.shape(imgM) != np.shape(imgC) or np.shape(imgF) != np.shape(imgC)):
            print('Unequal image sizes for ' + str(i))
            break
        h.append(np.shape(imgF)[0])
        w.append(np.shape(imgF)[1])
        maxValF.append(np.amax(imgF))
        maxValM.append(np.amax(imgM)) 
        maxValC.append(np.amax(imgC)) 
        minValF.append(np.amin(imgF)) 
        minValM.append(np.amin(imgM))
        minValC.append(np.amin(imgC))
        if(type(imgF[0,0]) != type(imgM[0,0]) or type(imgF[0,0]) != type(imgC[0,0]) or type(imgC[0,0]) != type(imgM[0,0])):
            print('Image type issue for ' + str(i))
            break
        typeImg.append(type(imgF[0,0]))

    df['height'] = h
    df['width'] = w
    df['maxSlice'] = maxValF
    df['minSlice'] = minValF
    df['maxMask'] = maxValM
    df['minMask'] = minValM
    df['maxContour'] = maxValC
    df['minContour'] = minValC
    df['typeImg'] = typeImg

    df.to_csv('Data100_Direct_FetalLimb.csv',index=False)