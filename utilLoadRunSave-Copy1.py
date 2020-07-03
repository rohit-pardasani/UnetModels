import pandas as pd
import numpy as np
import cv2
import random
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
from tensorflow.keras.preprocessing import image
import math
import json
import random
import os
import utilMetrics

def globalSeedSetter(seed_value):
    os.environ['PYTHONHASHSEED']=str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tensorflow.random.set_seed(seed_value)
    session_conf = tensorflow.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tensorflow.compat.v1.Session(graph=tensorflow.compat.v1.get_default_graph(), config=session_conf)
    tensorflow.compat.v1.keras.backend.set_session(sess)

def CustomImgGenNZ(indlst,df,H,W,onlyX=False,shuffle=False,BATCH_SIZE=16):
    L = len(indlst)
    while True:
        if(shuffle):
            random.shuffle(indlst)
            print(indlst)
        ii = 0 # Current image index
        left = L
        while left>0:
            BL = min(BATCH_SIZE,left)
            X_BATCH = np.zeros((BL,H,W,1),dtype=np.float32)
            Y_BATCH = np.zeros((BL,H,W,1),dtype=np.float32)
            for bi in range(BL):
                imgIdx = indlst[ii]
                img=cv2.imread(df.loc[imgIdx,'slice'],0)
                img = cv2.resize(img, (H, W), interpolation = cv2.INTER_CUBIC)
                img = np.float32(img)/255
                img = np.reshape(img, (H,W,1))
                X_BATCH[bi,:,:,:] = img
                
                if(not onlyX):
                    msk=cv2.imread(df.loc[imgIdx,'mask'],0)
                    msk = cv2.resize(msk, (H, W), interpolation = cv2.INTER_CUBIC)
                    msk = np.float32(msk)/255
                    msk = np.reshape(msk, (H,W,1))        
                    Y_BATCH[bi,:,:,:] = msk
                
                ii+=1
                
            left = left - BL
            if(not onlyX):
                yield (X_BATCH,Y_BATCH)
            else:
                yield X_BATCH

                        

def trainGen(modelName,csvFile,saveFolder,
                    indlist_train,indlist_val,indlist_test,indlist_pred,H,W,
                  model,reportCsv,min_lr=0.00001,epochs=200,LR_patience=20,LR_factor=0.1,stop_patience=100,
                    batch_size=2,retrainFlag=False):
    
    N_TRAIN=len(indlist_train)
    N_VALIDATE=len(indlist_val)
    N_TEST=len(indlist_test)
    N_PRED=len(indlist_pred)
    
    t_steps = math.ceil(N_TRAIN/batch_size)
    v_steps = math.ceil(N_VALIDATE/batch_size)
    tt_steps = math.ceil(N_TEST/batch_size)
    pred_steps= math.ceil(N_PRED/batch_size)

    
    df = pd.read_csv(csvFile)
    ur = pd.read_csv(reportCsv)

    # Check if save folder exists, in not create it
    if(not os.path.isdir(saveFolder)):
        os.mkdir(saveFolder)
        
    # Save location
    modelSave = saveFolder + '/' + modelName + '.h5'
    trainGraphSave=saveFolder + '/' + modelName+ '_training_plot.png'
    aucGraphSave=saveFolder + '/' + modelName+ '_AUC_plot.png'
    
    train_generator=CustomImgGenNZ(indlist_train,df,H,W,onlyX=False,shuffle=True,BATCH_SIZE=batch_size)
    val_generator=CustomImgGenNZ(indlist_val,df,H,W,onlyX=False,shuffle=True,BATCH_SIZE=batch_size)
    
    #compile model 
    model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"]) 
    model.summary()
    
    #compile model 
    callbacks = [EarlyStopping(patience=stop_patience, verbose=1),
                 ReduceLROnPlateau(factor=LR_factor, patience=LR_patience, min_lr=min_lr, verbose=1),
                 ModelCheckpoint(modelSave, verbose=1, save_best_only=True, save_weights_only=False)]
    
    #compile model 
    results = model.fit_generator(train_generator, steps_per_epoch=t_steps,  epochs=epochs,use_multiprocessing=False, 
                                  workers=0,validation_data=val_generator,validation_steps=v_steps,callbacks=callbacks, shuffle=False)
    
    # Save model Training curve

    plt.figure(figsize=(8, 8))
    plt.title("Learning curve")
    plt.plot(results.history["loss"], label="loss")
    plt.plot(results.history["val_loss"], label="val_loss")
    plt.plot( np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("log_loss")
    plt.legend()
    plt.savefig(trainGraphSave)
    plt.show()
    
    plist=[None] * 30
    
    plist[0]=model.count_params()
    plist[1]=N_PRED
    plist[2]=N_TRAIN
    plist[3]=N_VALIDATE
    plist[4]=N_TEST
    plist[5]=min_lr
    plist[6]=epochs
    plist[7]=LR_patience
    plist[8]=LR_factor
    plist[9]=stop_patience
    plist[10]=batch_size
    plist[11]=np.argmin(results.history["val_loss"])
    plist[12]=np.min(results.history["loss"])
    plist[13]=np.min(results.history["val_loss"])
    plist[14]=len(results.history["val_loss"])
    plist[15]=results.history["loss"][-1]
    plist[16]=results.history["val_loss"][-1]
    plist[18]=trainGraphSave
    plist[19]=modelSave
    
    ur[modelName]=plist
    ur.to_csv(reportCsv,index=False)
    
    ur=pd.read_csv(reportCsv)
    print(ur)
    
    
    return results
    

def testAndSaveReport(csvFile,saveFolder,modelName,indlist_train,indlist_val,indlist_test,indlist_pred,H,W,model,
                      reportCsv,batch_size=32):
    
    N_TRAIN=len(indlist_train)
    N_VALIDATE=len(indlist_val)
    N_TEST=len(indlist_test)
    N_PRED=len(indlist_pred)
    
    pred_steps= math.ceil(N_PRED/batch_size)
    
    df = pd.read_csv(csvFile)
    ur = pd.read_csv(reportCsv)

    
    pred_generator=CustomImgGenNZ(indlist_pred,df,H,W,onlyX=False,shuffle=False,BATCH_SIZE=batch_size)
    pred_X_generator=CustomImgGenNZ(indlist_pred,df,H,W,onlyX=True,shuffle=False,BATCH_SIZE=batch_size)
    
    pred_loss=model.evaluate_generator(pred_generator,steps = pred_steps,use_multiprocessing=False,workers=0)
    pred_Y_predict = model.predict_generator(pred_X_generator,steps=pred_steps,use_multiprocessing=False,workers=0)
    
    # Check if save folder exists, in not create it
    if(not os.path.isdir(saveFolder + '/'+ 'all_predicted')):
        os.mkdir(saveFolder + '/'+ 'all_predicted')
    
    
    for i in indlist_pred:
        name=df.loc[i,'slice'].split('/')[-1].split('_')[-1]
        fsave = saveFolder + '/'+ 'all_predicted/' + 'pred_' + name 
        p_img = pred_Y_predict[i,:,:,0]
        Ha = int(df.loc[i,'height'])
        Wa = int(df.loc[i,'width'])
        p_img = cv2.resize(p_img, (Wa,Ha), interpolation = cv2.INTER_CUBIC)
        p_img = p_img*255
        p_img = np.around(p_img,0)
        p_img[p_img>255] = 255
        p_img[p_img<0] = 0
        p_img = np.array(p_img,dtype=np.int32)
        cv2.imwrite(fsave,p_img)
    
    
    #Create AUC and find the best threshold
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
        fsave = saveFolder + '/'+ 'all_predicted/' +  'pred_' + name 
        p_img = cv2.imread(fsave,0)
        p_img = p_img/255.0
        if(cnt==0):
            flatp = np.ravel(p_img)
        else:
            flatp = np.concatenate([flatp, np.ravel(p_img)])
        flatp = np.array(flatp,dtype=np.float32)  

        aucGraphSave=saveFolder + '/' + modelName+ '_AUC_plot.png'
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
    
    
    # Check if save folder exists, in not create it
    if(not os.path.isdir(saveFolder + '/'+ 'all_thresholded')):
        os.mkdir(saveFolder + '/'+ 'all_thresholded')
        
    # Then save those thresholded images with sensitivity and specificity 
    for i in indlist_pred:
        name=df.loc[i,'slice'].split('/')[-1].split('_')[-1]
        fpred = saveFolder + '/' + "all_predicted/"+ 'pred_' + name
        p_img = cv2.imread(fpred,0)
        p_img = p_img/255.0
        p_img[p_img>T] = 255
        p_img[p_img<=T] = 0
        p_img = np.array(p_img,dtype=np.int32)
        fthresh = saveFolder + '/' +  "all_thresholded/"+'thresh_' + name
        cv2.imwrite(fthresh,p_img)

    
    # Check if save folder exists, in not create it
    if(not os.path.isdir(saveFolder + '/'+ 'all_overlayed')):
        os.mkdir(saveFolder + '/'+ 'all_overlayed')

    # Save overlayed images
    for i in indlist_pred:
        name=df.loc[i,'slice'].split('/')[-1].split('_')[-1]
        A = cv2.imread(df.loc[i,'contour'],0)
        fthresh = saveFolder + '/' +  "all_thresholded/"+'thresh_' + name
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
        fwrite = saveFolder + '/' +  "all_overlayed/"+'overlayed_' + name
        cv2.imwrite(fwrite, Z)   
        
    
    jaccard_dict = {}
    dice_dict = {}



    # Calculate Jaccard Index and Dice Coefficient
    for i in indlist_pred:
        name=df.loc[i,'slice'].split('/')[-1].split('_')[-1]
        trueMask = cv2.imread(df.loc[i,'mask'],0)
        fthresh = saveFolder + '/' +  "all_thresholded/"+'thresh_' + name

        predMask = cv2.imread(fthresh,0)
        #display(trueMask.dtype)
        #display(predMask.dtype)
        J = utilMetrics.jaccardIndex(predMask, trueMask)
        D = utilMetrics.diceCoefficient(predMask, trueMask)
        jaccard_dict[i]=J
        dice_dict[i]=D

    #print(jaccard_dict)
    #print(dice_dict)

    jlist_train,dlist_train = utilMetrics.getJaccardDiceCofficientList(jaccard_dict,dice_dict,indlist_train)
    jlist_val,dlist_val = utilMetrics.getJaccardDiceCofficientList(jaccard_dict,dice_dict,indlist_val)
    jlist_test,dlist_test = utilMetrics.getJaccardDiceCofficientList(jaccard_dict,dice_dict,indlist_test)
    jlist_pred,dlist_pred = utilMetrics.getJaccardDiceCofficientList(jaccard_dict,dice_dict,indlist_pred)
    
    plist=list(ur[modelName])

    plist[17]=pred_loss[0]
    plist[20]=T
    plist[21]=tpr[ind]
    plist[22]=1 - fpr[ind]
    plist[23]=aucGraphSave
    
    plist[24]=np.mean(jlist_train)
    plist[25]=np.mean(jlist_val)
    plist[26]=np.mean(jlist_test)

    plist[27]=np.mean(dlist_train)
    plist[28]=np.mean(dlist_val)
    plist[29]=np.mean(dlist_test)
    
    ur[modelName]=plist
    ur.to_csv(reportCsv,index=False)
    