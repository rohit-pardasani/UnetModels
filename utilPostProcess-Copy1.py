import cv2
import numpy as np
import os
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import jaccard_score

import utilMetrics
    

def postProcessImage(image):
    
    # findcontours
    _,contours, hierarchy = cv2.findContours(image =image , mode = cv2.RETR_TREE,method = cv2.CHAIN_APPROX_SIMPLE)

    # create an empty mask
    new_image = np.zeros(image.shape[:2],dtype=np.uint8)
    
    c = max(contours, key = cv2.contourArea)
    
    x,y = c.T
    # Convert from numpy arrays to normal arrays
    x = x.tolist()[0]
    y = y.tolist()[0]
    
    # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splprep.html
    tck, u = splprep([x,y], u=None, s=1.0, per=1)
    
    # https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.linspace.html
    u_new = np.linspace(u.min(), u.max(), 25)
    
    # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splev.html
    x_new, y_new = splev(u_new, tck, der=0)
    
    # Convert it back to numpy format for opencv to be able to display it
    res_array = [[[int(i[0]), int(i[1])]] for i in zip(x_new,y_new)]
    smoothened=np.asarray(res_array, dtype=np.int32)
    cv2.drawContours(new_image,[smoothened], 0, (255), -1)
    
    return new_image


def postProcessing(input_folder,output_folder):
    img_list=os.listdir(input_folder)
    # Load images one by one 
    for x in img_list:
        img_num=x.split('_')[1]
        image = cv2.imread(input_folder+x, 0) 
        new_image=postProcessImage(image)

        fsave = output_folder+'smoothened_'+img_num
        cv2.imwrite(fsave,new_image)

        #plt.subplots(1,2)
        fig,ax=plt.subplots(nrows=1,ncols=2)
        ax[0].imshow(image)
        ax[1].imshow(new_image)

        ax[0].set_title("Thresholded Image")
        ax[1].set_title("Biggest contour smoothened")

        plt.show()

def calculateAUC(indlist_train,csvFile,saveFolder,modelName):
        
    
    df = pd.read_csv(csvFile)
    
    cnt=0
    for i in indlist_train:
        y_img = cv2.imread(df.loc[i,'mask'],0)

        name=df.loc[i,'slice'].split('/')[-1].split('_')[-1]
        y_img = y_img/255.0
        y_img = np.around(y_img,0)
        if(cnt==0):
            flaty = np.ravel(y_img)
        else:
            flaty = np.concatenate([flaty, np.ravel(y_img)])
        fsave = saveFolder + '/'+ 'all_smoothened/' +  'smoothened_' + name 
        p_img = cv2.imread(fsave,0)
        p_img = p_img/255.0
        if(cnt==0):
            flatp = np.ravel(p_img)
        else:
            flatp = np.concatenate([flatp, np.ravel(p_img)])
        flatp = np.array(flatp,dtype=np.float32)  

        aucGraphSave=saveFolder + '/' + modelName+ 'smoothened_AUC_plot.png'
        cnt+=1

    # calculate roc curve

    fpr, tpr, thresholds = roc_curve(flaty, flatp)
    auc = roc_auc_score(flaty, flatp)
    print(auc)
    plt.plot(fpr, tpr)
    diff = tpr - fpr
    ind = np.argmax(diff)
    T = thresholds[ind]
    print('Total Points: ' + str(len(flaty)))
    print('Positive Points: ' + str(sum(flaty)))
    print('Optimal Threshold: '+ str(T))

    print('Sensitivity at threshold: ' + str(tpr[ind]))
    print('Specificity at threshold: ' + str(1 - fpr[ind]))
    plt.title('ROC curve')
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.savefig(aucGraphSave)
    plt.show()



    plt.plot(thresholds,diff)
    plt.title('threshold vs tpr-fpr')
    plt.xlabel('thresholds')
    plt.ylabel('tpr - fpr')
    plt.show()

def saveOverlayedImages(indlist_pred,csvFile,saveFolder):
    
    df = pd.read_csv(csvFile)
    
    # Check if save folder exists, in not create it
    if(not os.path.isdir(saveFolder + '/'+ 'all_overlayed_final')):
        os.mkdir(saveFolder + '/'+ 'all_overlayed_final')

    for i in indlist_pred:
        name=df.loc[i,'slice'].split('/')[-1].split('_')[-1]
        A = cv2.imread(df.loc[i,'contour'],0)
        fthresh = saveFolder + '/' +  "all_smoothened/"+'smoothened_' + name
        B = cv2.imread(fthresh,0)
        (H,W) = np.shape(A)
        Z = np.zeros((H, W, 3), np.uint8)
        for yc in range(H):
            for xc in range(W):
                if(A[yc,xc] == 255):
                    Z[yc,xc,0] = 0
                    Z[yc,xc,1] = 0
                    Z[yc,xc,2] = 255
                if(A[yc,xc] == 0 and B[yc,xc]==255):
                    Z[yc,xc,0] = 0
                    Z[yc,xc,1] = 255
                    Z[yc,xc,2] = 255
          
        fwrite = saveFolder + '/' +  "all_overlayed_final/"+'overlayed_final_' + name
        cv2.imwrite(fwrite, Z)   

def calculateFinalJaccardDice(indlist_pred,indlist_train,indlist_val,indlist_test,csvFile,saveFolder):
    
    df=pd.read_csv(csvFile)
    
    jaccard_dict = {}
    dice_dict = {}

    # Calculate Jaccard Index and Dice Coefficient
    for i in indlist_pred:
        name=df.loc[i,'slice'].split('/')[-1].split('_')[-1]
        trueMask = cv2.imread(df.loc[i,'mask'],0)
        fthresh = saveFolder + '/' +  "all_smoothened/"+'smoothened_' + name

        predMask = cv2.imread(fthresh,0)
        #display(trueMask.dtype)
        #display(predMask.dtype)
        J = utilMetrics.jaccardIndex(predMask, trueMask)
        D = utilMetrics.diceCoefficient(predMask, trueMask)
        jaccard_dict[i]=J
        dice_dict[i]=D
    
    jlist_train,dlist_train = utilMetrics.getJaccardDiceCofficientList(jaccard_dict,dice_dict,indlist_train)
    jlist_val,dlist_val = utilMetrics.getJaccardDiceCofficientList(jaccard_dict,dice_dict,indlist_val)
    jlist_test,dlist_test = utilMetrics.getJaccardDiceCofficientList(jaccard_dict,dice_dict,indlist_test)
    jlist_pred,dlist_pred = utilMetrics.getJaccardDiceCofficientList(jaccard_dict,dice_dict,indlist_pred)
    
    
    l=[0]*6
    l[0]=np.mean(jlist_train)
    l[1]=np.mean(jlist_val)
    l[2]=np.mean(jlist_test)

    l[3]=np.mean(dlist_train)
    l[4]=np.mean(dlist_val)
    l[5]=np.mean(dlist_test)
    
    return l

