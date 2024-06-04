# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 11:03:07 2020

@author: Elvis
"""
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from scipy import stats
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from itertools import combinations 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import random
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE

def UpdatePartitionIndices(partition):
    begin = 0
    for i in range(partition.size):
        end = partition[i][0].size + begin
        partition[i] = [np.array(range(begin,end))]
        begin = partition[i][0].max()+1
    return partition

def SFFS(decimDataset,decimPartition,decimLabels,imageLabels,classifier,pixelDataset,pixelPartition):
    stopSearch = False
    maxFeatures = 5 
    featureBox = []
    testedFeatures = []
    optimalAUC = [0]*maxFeatures
    optimalFeatures = [0]*maxFeatures
    k = 0
    while stopSearch == False:
        # FORWARD STEP
        featureBox, maxAUC, testedFeatures = SFS(featureBox,testedFeatures,decimDataset,decimPartition,decimLabels,imageLabels,
                                                 classifier,pixelDataset,pixelPartition)
        k += 1
        while k > 2: 
            # BACKWARD STEP
            bestSubset, bestSubsetAUC, testedFeatures = SBS(featureBox,testedFeatures,decimDataset,decimPartition,
                                                     decimLabels,imageLabels,classifier,pixelDataset,pixelPartition)
            if bestSubsetAUC > optimalAUC[k-2]: # If improvement found
                featureBox = bestSubset
                maxAUC = bestSubsetAUC
                k -= 1   # Update k
            else:
                break
        optimalFeatures[k-1] = featureBox[:]
        optimalAUC[k-1] = maxAUC
        if k == maxFeatures:
            stopSearch = True # Stop if all features have been tested
            indxBestFeatures = optimalAUC.index(max(optimalAUC[0:maxFeatures-1])) # Find the max AUC
            bestFeatures = optimalFeatures[indxBestFeatures]
    return bestFeatures

def SFS(featureBox,testedFeatures,decimDataset,decimPartition,decimLabels,imageLabels,classifier,pixelDataset,pixelPartition):
    Nf = decimDataset.shape[1]
    sampleIndx = np.arange(imageLabels.size)
    testIndx = [np.arange(11),np.arange(11,22),np.arange(22,33),np.arange(33,44),np.arange(44,55),np.arange(55,66)]
    print('SFS Size:',len(featureBox)+1)     
    featPerf = [0]*Nf   
    for j in range(Nf): # Evaluate each feature
        rocAUC = [0]*len(testIndx)
        if j not in featureBox:
            featureBox.append(j)
            testedFeatures.append(featureBox[:])
            for fold in range(len(testIndx)):
                trainIndx = np.setdiff1d(sampleIndx,testIndx[fold])
                rows = [val for s1 in decimPartition[trainIndx] for s2 in s1 for val in s2]
                train_decimData = stats.zscore(decimDataset[rows,:])
                mu = np.mean(decimDataset[rows,:],0)
                sigma = np.std(decimDataset[rows,:],0)
                trainDecimLabels = decimLabels[rows]
                test_originalPartition = pixelPartition[testIndx[fold]]
                rows = [val for s1 in test_originalPartition for s2 in s1 for val in s2]
                test_originalData = (pixelDataset[rows,:] - mu)/sigma
                test_originalPartition = UpdatePartitionIndices(test_originalPartition)
                testImageLabels = imageLabels[testIndx[fold]]  
                if classifier == 'LDA':
                    model = LinearDiscriminantAnalysis()
                elif classifier == 'QDA':
                    model = QuadraticDiscriminantAnalysis()
                elif classifier == 'SVM':
                    model = LinearSVC(dual=False,C=train_decimData.shape[0])
                    model = CalibratedClassifierCV(model)
                else: # LOGREG
                    model = LogisticRegression()
                model.fit(train_decimData[:,featureBox],trainDecimLabels)
                ##### Apply Model to Validation Set #####
                ppMaps = model.predict_proba(test_originalData[:,featureBox])
                cancerProbs = ppMaps[:,1]
                ##### ROC AUC #####
                cancerScores = np.zeros((testImageLabels.size,1))
                for i in range(testImageLabels.size):
                    cancerScores[i] = np.mean(cancerProbs[test_originalPartition[i][0]]**2)
                rocAUC[fold] = roc_auc_score(testImageLabels,cancerScores)
            featureBox.remove(j)
            featPerf[j] = round(np.median(rocAUC),2)     
    indxBestFeat = featPerf.index(max(featPerf)) # Find the max AUC
    featureBox.append(indxBestFeat) # Select features that maximize AUC
    maxAUC = featPerf[indxBestFeat]
    return featureBox, maxAUC, testedFeatures

def SBS(featureBox,testedFeatures,decimDataset,decimPartition,decimLabels,imageLabels,classifier,pixelDataset,pixelPartition):
    sampleIndx = np.arange(imageLabels.size)
    testIndx = [np.arange(11),np.arange(11,22),np.arange(22,33),np.arange(33,44),np.arange(44,55),np.arange(55,66)]
    k = len(featureBox)
    subsetPerf = [0]*k
    subsets = [0]*k
    print('SBS Size:', k-1)
    for j in range(k):
        featSubset = np.setdiff1d(featureBox,featureBox[j]).tolist()
        subsets[j] = featSubset
        if featSubset not in testedFeatures:
            rocAUC = [0]*len(testIndx)
            testedFeatures.append(featSubset[:])
            for fold in range(len(testIndx)):
                trainIndx = np.setdiff1d(sampleIndx,testIndx[fold])
                rows = [val for s1 in decimPartition[trainIndx] for s2 in s1 for val in s2]
                train_decimData = stats.zscore(decimDataset[rows,:])
                mu = np.mean(decimDataset[rows,:],0)
                sigma = np.std(decimDataset[rows,:],0)
                trainDecimLabels = decimLabels[rows]
                test_originalPartition = pixelPartition[testIndx[fold]]
                rows = [val for s1 in test_originalPartition for s2 in s1 for val in s2]
                test_originalData = (pixelDataset[rows,:] - mu)/sigma
                test_originalPartition = UpdatePartitionIndices(test_originalPartition)
                testImageLabels = imageLabels[testIndx[fold]]  
                if classifier == 'LDA':
                    model = LinearDiscriminantAnalysis()
                elif classifier == 'QDA':
                    model = QuadraticDiscriminantAnalysis()
                elif classifier == 'SVM':
                    model = LinearSVC(dual=False,C=train_decimData.shape[0])
                    model = CalibratedClassifierCV(model)
                else: # LOGREG
                    model = LogisticRegression()
                model.fit(train_decimData[:,featSubset],trainDecimLabels)
                ##### Apply Model to Validation Set #####
                ppMaps = model.predict_proba(test_originalData[:,featSubset])
                cancerProbs = ppMaps[:,1]
                ##### ROC AUC #####
                cancerScores = np.zeros((testImageLabels.size,1))
                for i in range(testImageLabels.size):
                    cancerScores[i] = np.mean(cancerProbs[test_originalPartition[i][0]]**2)
                rocAUC[fold] = roc_auc_score(testImageLabels,cancerScores)
            subsetPerf[j] = (round(np.median(rocAUC),2))    
    bestSubsetAUC = max(subsetPerf)
    indxBestSubset = subsetPerf.index(bestSubsetAUC) # Find index of the max AUC
    bestSubset = subsets[indxBestSubset]
    return bestSubset, bestSubsetAUC, testedFeatures    
    
def SeqFwdSearchAUC(decimDataset,decimPartition,decimLabels,imageLabels,classifier,pixelDataset,pixelPartition,splitData):
    stopSearch = False
    k, maxFeatures = 1, 3 #Limited by Sample Size
    featureBox = []
    Nf = decimDataset.shape[1]
    optimalAUC = [0]*maxFeatures
    sampleIndx = np.arange(imageLabels.size)
    bestAUC = -1
    if splitData == True:
    #################### K-FOLD CV #############################
        testIndx = [np.arange(11),np.arange(11,22),np.arange(22,33),np.arange(33,44),np.arange(44,55),np.arange(55,66)]
    #################### Split training/testing #############################
        #indxBEN = np.where(imageLabels==0)[0].tolist()
        #indxSCC = np.where(imageLabels==1)[0].tolist()
        #testIndx = random.sample(indxBEN,5)+random.sample(indxSCC,5)
        #trainIndx = np.setdiff1d(indxBEN+indxSCC,testIndx)
        #rows = [val for s1 in decimPartition[trainIndx] for s2 in s1 for val in s2]
        #train_decimData = stats.zscore(decimDataset[rows,:])
        #mu = np.mean(decimDataset[rows,:],0)
        #sigma = np.std(decimDataset[rows,:],0)
        #trainDecimLabels = decimLabels[rows]
        #train_originalPartition = pixelPartition[trainIndx]
        #rows = [val for s1 in train_originalPartition for s2 in s1 for val in s2]
        #train_originalData = pixelDataset[rows,:]
        #train_originalPartition = UpdatePartitionIndices(train_originalPartition)     
        #test_originalPartition = pixelPartition[testIndx]
        #rows = [val for s1 in test_originalPartition for s2 in s1 for val in s2]
        #test_originalData = (pixelDataset[rows,:] - mu)/sigma
        #test_originalPartition = UpdatePartitionIndices(test_originalPartition)
        #trainImageLabels = imageLabels[trainIndx]
        #testImageLabels = imageLabels[testIndx]
    #########################################################################
    else:
        train_decimData = decimDataset
        trainDecimLabels = decimLabels
        test_originalPartition = pixelPartition
        test_originalData = pixelDataset
        testImageLabels = imageLabels
    while stopSearch == False:
        print('SFS Size:',k)     
        featPerf = [0]*Nf   
        for j in range(Nf): # Evaluate each feature
                rocAUC = [0]*len(testIndx)
                if j not in featureBox:
                    featureBox.append(j)
                    for fold in range(len(testIndx)):
                        trainIndx = np.setdiff1d(sampleIndx,testIndx[fold])
                        rows = [val for s1 in decimPartition[trainIndx] for s2 in s1 for val in s2]
                        train_decimData = stats.zscore(decimDataset[rows,:])
                        mu = np.mean(decimDataset[rows,:],0)
                        sigma = np.std(decimDataset[rows,:],0)
                        trainDecimLabels = decimLabels[rows]
                        test_originalPartition = pixelPartition[testIndx[fold]]
                        rows = [val for s1 in test_originalPartition for s2 in s1 for val in s2]
                        test_originalData = (pixelDataset[rows,:] - mu)/sigma
                        test_originalPartition = UpdatePartitionIndices(test_originalPartition)
                        testImageLabels = imageLabels[testIndx[fold]]  
                        if classifier == 'LDA':
                            model = LinearDiscriminantAnalysis()
                        elif classifier == 'QDA':
                            model = QuadraticDiscriminantAnalysis()
                        elif classifier == 'SVM':
                            model = LinearSVC(dual=False,C=train_decimData.shape[0])
                            model = CalibratedClassifierCV(model)
                        else: # LOGREG
                            model = LogisticRegression()
                        model.fit(train_decimData[:,featureBox],trainDecimLabels)
                        ##### Apply Model to Validation Set #####
                        ppMaps = model.predict_proba(test_originalData[:,featureBox])
                        cancerProbs = ppMaps[:,1]
                        ##### ROC AUC #####
                        cancerScores = np.zeros((testImageLabels.size,1))
                        for i in range(testImageLabels.size):
                            cancerScores[i] = np.mean(cancerProbs[test_originalPartition[i][0]]**2)
                        rocAUC[fold] = roc_auc_score(testImageLabels,cancerScores)
                    featureBox.remove(j)
                    featPerf[j] = round(np.median(rocAUC),2)     
        indxBestFeat = featPerf.index(max(featPerf)) # Find the max AUC
        featureBox.append(indxBestFeat) # Select features that maximize AUC
        maxAUC = featPerf[indxBestFeat]
        optimalAUC[k-1] = maxAUC
        if (optimalAUC[k-1] - bestAUC) >= 0.03:
            bestFeatures = featureBox[0:k]
            bestAUC = optimalAUC[k-1]
        if k == maxFeatures:
            stopSearch = True # Stop if all features have been tested
            #bestAUC = max(optimalAUC)
            #indxBestFeatures = optimalAUC.index(bestAUC) # Find the max AUC
            #bestFeatures = featureBox[0:indxBestFeatures+1]
        else:       
            k += 1 
    return bestFeatures, bestAUC

def TrainingValidation(allIndx,validationIndx,trainLabels,train_decimPartition,train_decimData,
                       train_originalPartition,train_originalData,classifier,fold_indx):
    trainingIndx = np.setdiff1d(allIndx,validationIndx)
    trueImageLabels = trainLabels[trainingIndx]
    # Training Set 
    rows = [val for s1 in train_decimPartition[trainingIndx] for s2 in s1 for val in s2]
    trainDecimDataset = stats.zscore(train_decimData[rows,0:-1]) # Normalize Training Dataset
    mu = np.mean(train_decimData[rows,0:-1],0)
    sigma = np.std(train_decimData[rows,0:-1],0)
    trainDecimLabels = train_decimData[rows,-1]  
    rows = [val for s1 in train_originalPartition[trainingIndx] for s2 in s1 for val in s2]
    trainPixelDataset = (train_originalData[rows,0:-1]- mu)/sigma # Normalize Training Dataset      
    # Update Partition Indices 
    trainDecimPartition = UpdatePartitionIndices(train_decimPartition[trainingIndx])
    trainPixelPartition = UpdatePartitionIndices(train_originalPartition[trainingIndx])  
    testPixelPartition = UpdatePartitionIndices(train_originalPartition[validationIndx])
    # Validation Set
    rows = [val for s1 in train_originalPartition[validationIndx] for s2 in s1 for val in s2]
    testPixelDataset = (train_originalData[rows,0:-1] - mu)/sigma           
    # Feature Selection
    selectedFeatures, bestAUC, featureBox = SeqFwdSearchAUC(trainDecimDataset,trainDecimPartition,trainDecimLabels,
                                                            trueImageLabels,classifier,trainPixelDataset,trainPixelPartition)
    selectedFeatures.sort()
    # Retrain Classifier
    if classifier == 'LDA':
        model = LinearDiscriminantAnalysis()
    elif classifier == 'QDA':
        model = QuadraticDiscriminantAnalysis()
    elif classifier == 'SVM':
        model = LinearSVC(dual=False,C=trainDecimDataset.shape[0])
        model = CalibratedClassifierCV(model)
    else: # LOGREG
        model = LogisticRegression()
    model.fit(trainDecimDataset[:,selectedFeatures],trainDecimLabels)
    # Pick ROC Threshold
    ppMaps = model.predict_proba(trainPixelDataset[:,selectedFeatures])
    cancerProbs = ppMaps[:,1]
    cancerScores = [0]*trainPixelPartition.size
    for z in range(trainPixelPartition.size):
        cancerScores[z] = sum(cancerProbs[trainPixelPartition[z][0]]**2)/trainPixelPartition[z][0].size
    X,Y,T = roc_curve(trueImageLabels,cancerScores,drop_intermediate=False)
    maxY = max(Y[(X>0) & (X<=0.3)])
    minX = min(X[Y==maxY])
    indx = (X==minX) & (Y==maxY)
    #distance = (Y-X).tolist()
    #indx = distance.index(max(distance)) # ROC Point closest to the upper left corner      
    ################### Classify VALIDATION Samples #######################
    ppMaps = model.predict_proba(testPixelDataset[:,selectedFeatures]) 
    cancerProbs = ppMaps[:,1]
    cancerScores = [0]*testPixelPartition.size
    for z in range(testPixelPartition.size):
        cancerScores[z] = sum(cancerProbs[testPixelPartition[z][0]]**2)/testPixelPartition[z][0].size
    predSampleLabels = (cancerScores >= T[indx][0])*1
    roc_auc = roc_auc_score(trainLabels[validationIndx],cancerScores)
    # Confusion Matrix (Image Classification)
    C = confusion_matrix(trainLabels[validationIndx],predSampleLabels)
    return selectedFeatures, T[indx][0], cancerProbs, roc_auc, cancerScores, predSampleLabels,\
        C[1,1]/sum(C[1,:]), C[0,0]/sum(C[0,:]), (C[1,1] + C[0,0])/sum(C.flatten()), fold_indx

def CFS(Rcf,Rff):
    featureBox = []
    stopSearch = False
    Nf = Rcf.size
    bestMerit = [0]*Nf
    z = 0
    prevMerit = -1
    while stopSearch == False:
        merit = [0]*Nf
        for j in range(Nf): 
            avgFFcorr = 0
            if j not in featureBox:
                featureBox.append(j)
                k = len(featureBox)
                if k > 1:
                    comb = list(combinations(featureBox,2))
                    for c in comb:
                        avgFFcorr += Rff[c]
                    avgFFcorr = avgFFcorr/len(comb)
                avgCFcorr = np.mean(Rcf[featureBox])
                merit[j] = round((k*avgCFcorr)/np.sqrt(k + k*(k-1)*avgFFcorr),2)
                featureBox.remove(j)
        indxBestFeat = merit.index(max(merit)) # Find the max Merit 
        bestMerit[z] = merit[indxBestFeat]
        if bestMerit[z] < prevMerit: 
            stopSearch = True
        else:
            featureBox.append(indxBestFeat) # Select features that maximize Merit
            prevMerit = bestMerit[z]
            z+=1    
    return featureBox       

def EXS(decimDataset,decimPartition,decimLabels,imageLabels,classifier,pixelDataset,pixelPartition):
    Nf = decimDataset.shape[1]
    f = list(range(Nf))
    if Nf > 3:Nf = 3 # Combinations up to Nf features
    #comb = sum([list(map(list, combinations(f, n))) for n in range(1,Nf+1)], [])
    #featPerf = [0]*len(comb) 
    sampleIndx = np.arange(imageLabels.size)
    testIndx = [np.arange(11),np.arange(11,22),np.arange(22,33),np.arange(33,44),np.arange(44,55),np.arange(55,66)]
    maxAUC = -1
    for n in range(1,Nf+1):
        comb = sum([list(map(list, combinations(f, n)))], [])
        featPerf = [0]*len(comb)
        for j, featComb in enumerate(comb): # Evaluate each feature combination
                print('EXS Combination:',j+1) 
                rocAUC = [0]*len(testIndx)
                for fold in range(len(testIndx)):
                    trainIndx = np.setdiff1d(sampleIndx,testIndx[fold])
                    rows = [val for s1 in decimPartition[trainIndx] for s2 in s1 for val in s2]
                    train_decimData = stats.zscore(decimDataset[rows,:])
                    mu = np.mean(decimDataset[rows,:],0)
                    sigma = np.std(decimDataset[rows,:],0)
                    trainDecimLabels = decimLabels[rows]
                    test_originalPartition = pixelPartition[testIndx[fold]]
                    rows = [val for s1 in test_originalPartition for s2 in s1 for val in s2]
                    test_originalData = (pixelDataset[rows,:] - mu)/sigma
                    test_originalPartition = UpdatePartitionIndices(test_originalPartition)
                    testImageLabels = imageLabels[testIndx[fold]]  
                    if classifier == 'LDA':
                        model = LinearDiscriminantAnalysis()
                    elif classifier == 'QDA':
                        model = QuadraticDiscriminantAnalysis()
                    elif classifier == 'SVM':
                        model = LinearSVC(dual=False,C=0.1,class_weight='balanced')
                        model = CalibratedClassifierCV(model)
                    else: # LOGREG
                        model = LogisticRegression()
                    model.fit(train_decimData[:,featComb],trainDecimLabels)
                    ##### Apply Model to Validation Set #####
                    ppMaps = model.predict_proba(test_originalData[:,featComb])
                    cancerProbs = ppMaps[:,1]
                    ##### ROC AUC #####
                    cancerScores = np.zeros((testImageLabels.size,1))
                    for i in range(testImageLabels.size):
                        cancerScores[i] = np.mean(cancerProbs[test_originalPartition[i][0]]**2)
                    rocAUC[fold] = roc_auc_score(testImageLabels,cancerScores)
                featPerf[j] = round(np.median(rocAUC),2)   
        indxBestComb = featPerf.index(max(featPerf)) # Find the max AUC
        bestAUC = featPerf[indxBestComb]
        if (bestAUC - maxAUC) >= 0.03:
            bestFeatures = comb[indxBestComb]
            maxAUC = bestAUC
    return bestFeatures, maxAUC
                
def LOOCV(allIndx,validationIndx,trainLabels,train_decimPartition,train_decimData,
                       train_originalPartition,train_originalData,classifier,fold_indx):   
    trainingIndx = np.setdiff1d(allIndx,validationIndx)
    trueImageLabels = trainLabels[trainingIndx]
    # Training Set 
    rows = [val for s1 in train_decimPartition[trainingIndx] for s2 in s1 for val in s2]
    trainDecimDataset = stats.zscore(train_decimData[rows,0:-1]) # Normalize Training Dataset
    mu = np.mean(train_decimData[rows,0:-1],0)
    sigma = np.std(train_decimData[rows,0:-1],0)
    trainDecimLabels = train_decimData[rows,-1]  
    rows = [val for s1 in train_originalPartition[trainingIndx] for s2 in s1 for val in s2]
    trainPixelDataset = (train_originalData[rows,0:-1]- mu)/sigma # Normalize Training Dataset      
    # Update Partition Indices 
    trainDecimPartition = UpdatePartitionIndices(train_decimPartition[trainingIndx])
    trainPixelPartition = UpdatePartitionIndices(train_originalPartition[trainingIndx])  
    # Validation Set
    rows = [val for s1 in train_originalPartition[validationIndx] for val in s1]
    testPixelDataset = (train_originalData[rows,0:-1] - mu)/sigma           

    # Correlation Filter
    Rff = abs(np.corrcoef(trainDecimDataset,rowvar=False)) # Feature-Feature Correlation Coefficients
    Rcf = np.zeros((trainDecimDataset.shape[1],1)) # Feature-Class Correlation Coefficients
    for i in range(trainDecimDataset.shape[1]):
        Rcf[i] = abs(np.corrcoef(trainDecimDataset[:,i],trainDecimLabels,rowvar=False)[0,1])
    featuresCFS = CFS(Rcf,Rff)   
    trainDecimDataset = trainDecimDataset[:,featuresCFS]
    trainPixelDataset = trainPixelDataset[:,featuresCFS]
    testPixelDataset = testPixelDataset[:,featuresCFS]
    
    # FEATURE SELECTION
   #selectedFeatures, bestAUC = SeqFwdSearchAUC(trainDecimDataset,trainDecimPartition,trainDecimLabels,
                                                           #trueImageLabels,classifier,trainPixelDataset,trainPixelPartition,True)
    selectedFeatures, bestAUC = EXS(trainDecimDataset,trainDecimPartition,trainDecimLabels,
                                                            trueImageLabels,classifier,trainPixelDataset,trainPixelPartition)
    #selectedFeatures = SFFS(trainDecimDataset,trainDecimPartition,trainDecimLabels,trueImageLabels,classifier,
    #                        trainPixelDataset,trainPixelPartition)
    selectedFeatures.sort()
    # Retrain Classifier
    if classifier == 'LDA':
        model = LinearDiscriminantAnalysis()
    elif classifier == 'QDA':
        model = QuadraticDiscriminantAnalysis()
    elif classifier == 'SVM':
        model = LinearSVC(dual=False,C=0.1,class_weight='balanced')
        model = CalibratedClassifierCV(model)
    else: # LOGREG
        model = LogisticRegression()
    model.fit(trainDecimDataset[:,selectedFeatures],trainDecimLabels)
    # Pick ROC Threshold
    ppMaps = model.predict_proba(trainPixelDataset[:,selectedFeatures])
    cancerProbs = ppMaps[:,1]
    cancerScores = [0]*trainPixelPartition.size
    for z in range(trainPixelPartition.size):
        cancerScores[z] = np.mean(cancerProbs[trainPixelPartition[z][0]]**2)
    X,Y,T = roc_curve(trueImageLabels,cancerScores,drop_intermediate=False)
    maxY = max(Y[(X>0) & (X<=0.3)])
    minX = min(X[Y==maxY])
    indx = (X==minX) & (Y==maxY)
    ################### Classify VALIDATION Sample #######################
    ppMaps = model.predict_proba(testPixelDataset[:,selectedFeatures]) 
    ppValidation = ppMaps[:,1]
    cancerScore = np.mean(ppValidation**2)
    predSampleLabel = (cancerScore >= T[indx][0])*1   
    selectedFeatures = np.array(featuresCFS)[selectedFeatures].tolist()
    return selectedFeatures, ppValidation, predSampleLabel, bestAUC, featuresCFS, fold_indx
        
def L1_SVM_LOOCV(allIndx,validationIndx,trainLabels,train_decimPartition,train_decimData,
                       train_originalPartition,train_originalData,param):   
    print("FOLD:",validationIndx+1)
    trainingIndx = np.setdiff1d(allIndx,validationIndx)
    trueImageLabels = trainLabels[trainingIndx]
    
    # TRAINING 
    rows = [val for s1 in train_decimPartition[trainingIndx] for s2 in s1 for val in s2]
    trainDecimDataset = train_decimData[rows,:-1] 
    trainDecimLabels = train_decimData[rows,-1]  
    rows = [val for s1 in train_originalPartition[trainingIndx] for s2 in s1 for val in s2]
    trainPixelDataset = train_originalData[rows,:-1]   
    # Update Partition Indices 
    #trainDecimPartition = UpdatePartitionIndices(train_decimPartition[trainingIndx])  
    trainPixelPartition = UpdatePartitionIndices(train_originalPartition[trainingIndx])  
    
    # TESTING
    rows = [val for s1 in train_originalPartition[validationIndx] for val in s1]
    testPixelDataset = train_originalData[rows,:-1] 
    '''
    # K-FOLD CV SPLITS 
    newTrainIndx = np.arange(trainingIndx.size)
    nValSamples = 9 # For 5-Fold-CV
    lims = np.arange(newTrainIndx.size+1,step=nValSamples)
    cvSplits = [list(range(lims[i],lims[i+1])) for i in range(lims.size-1)]
    cvSplits = cvSplits + [list(range(cvSplits[-1][-1]+1,newTrainIndx.size)) + [0]]

    #indxBEN = np.where(trueImageLabels==0)[0].tolist()
    #indxCAN = np.where(trueImageLabels==1)[0].tolist()
    #random.seed(8)
    #cvSplits = [random.sample(indxBEN,7)+random.sample(indxCAN,7) for i in range(5)]
    
    decimRowIndx = np.arange(trainDecimDataset.shape[0])
    trainValPairs = [0]*len(cvSplits)
    for i, fold in enumerate(cvSplits):
         valRows = [val for s1 in trainDecimPartition[fold] for s2 in s1 for val in s2]
         trainValPairs[i] = (np.setdiff1d(decimRowIndx,valRows),np.array(valRows))
    '''
    '''
    # PIPELINE
    model = Pipeline([('scaler',StandardScaler()),('clf',RandomForestClassifier(n_estimators=param[0],max_depth=param[1],
                                                                    criterion='gini',bootstrap=True,class_weight='balanced'))])     
    model.fit(trainDecimDataset,trainDecimLabels)
    importance = model['clf'].feature_importances_
    importance = np.round(importance,2)  
    selectedFeatures = np.argsort(importance)[::-1][:3].tolist() # Sort in descending order and pick top features
    selectedFeatures.sort()
    model.steps[1] = ('clf',RandomForestClassifier(n_estimators=param[0],max_depth=param[1],criterion='gini',
                                                   bootstrap=True,class_weight='balanced',max_features=3))
    model.fit(trainDecimDataset[:,selectedFeatures],trainDecimLabels)
    '''
    
    model = Pipeline([('scaler', StandardScaler()), 
                      ('clf',LinearSVC(penalty='l1',dual=False,C=param,class_weight='balanced',max_iter=150000,tol=1e-4))])     
    model.fit(trainDecimDataset,trainDecimLabels)
    weights_norm = abs(model['clf'].coef_)/np.sum(abs(model['clf'].coef_)) # Take absolute value of weights and normalize
    weights_norm = np.round(weights_norm,2).ravel() # Round to 2 decimals
    selectedFeatures = np.argsort(weights_norm)[::-1][:4].tolist() # Sort in descending order and pick top features
    selectedFeatures.sort()
    model.steps[1] = ('clf',CalibratedClassifierCV(model['clf']))
    model.fit(trainDecimDataset[:,selectedFeatures],trainDecimLabels)
    '''
    classifier = LinearSVC(penalty='l1',dual=False,C=param,class_weight='balanced',max_iter=50000,tol=1e-4)
    model = Pipeline([('scaler', StandardScaler()), 
                      ('rfe',RFE(estimator=classifier, n_features_to_select=8)),('clf',CalibratedClassifierCV(classifier))]) 
    model.fit(trainDecimDataset,trainDecimLabels)
    selectedFeatures = np.where(model['rfe'].support_==True)[0].tolist()
    '''
    # ROC THRESHOLD
    cancerProbs = model.predict_proba(trainPixelDataset[:,selectedFeatures])[:,1]
    cancerScores = [0]*trainPixelPartition.size
    for z in range(trainPixelPartition.size):
        cancerScores[z] = np.mean(cancerProbs[trainPixelPartition[z][0]])
    X,Y,T = roc_curve(trueImageLabels,cancerScores,drop_intermediate=False)
    maxY = max(Y[(X>0) & (X<=0.3)])
    minX = min(X[Y==maxY])
    indx = (X==minX) & (Y==maxY)
    
    ################### Classify VALIDATION Sample #######################
    ppValidation = model.predict_proba(testPixelDataset[:,selectedFeatures])[:,1]
    cancerScore = np.mean(ppValidation)
    predSampleLabel = (cancerScore >= T[indx][0])*1   
    return ppValidation, predSampleLabel, selectedFeatures, validationIndx

def Image_L1_SVM_LOOCV(allIndx,validationIndx,trainLabels,trainDataset,CC,TOL):   
    print("FOLD:",validationIndx+1)
    trainIndx = np.setdiff1d(allIndx,validationIndx)
    trainData = trainDataset[trainIndx,:]
    trueImageLabels = trainLabels[trainIndx]
    testData = trainDataset[validationIndx,:]
    '''
    # K-FOLD CV SPLITS
    nValSamples = 9
    lims = np.arange(trainData.shape[0]+1,step=nValSamples)
    cvSplits = [list(range(lims[i],lims[i+1])) for i in range(lims.size-1)]
    cvSplits = cvSplits + [list(range(cvSplits[-1][-1]+1,trainData.shape[0])) + [0]]
    
    indxBEN = np.where(trueImageLabels==0)[0].tolist()
    indxCAN = np.where(trueImageLabels==1)[0].tolist()
    random.seed(8)
    cvSplits = [random.sample(indxBEN,7)+random.sample(indxCAN,7) for i in range(5)]
    trainDataIndx = np.arange(trainData.shape[0])
    trainValPairs = [(np.setdiff1d(trainDataIndx,cvSplits[i]),np.array(cvSplits[i])) for i in range(len(cvSplits))]
    '''
    # PIPELINE
    model = Pipeline([('scaler', StandardScaler()), 
                              ('clf',LinearSVC(penalty='l1',dual=False,C=CC,class_weight='balanced', max_iter=100000,tol=TOL))])
    model.fit(trainData,trueImageLabels)
    weights_norm = abs(model['clf'].coef_)/np.sum(abs(model['clf'].coef_)) # Take absolute value of weights and normalize
    weights_norm = np.round(weights_norm,3).ravel() # Round to 3 decimals
    selectedFeatures = np.argsort(weights_norm)[::-1][:3].tolist() # Sort in descending order and pick top features
    selectedFeatures.sort()
    model.steps[1] = ('clf',CalibratedClassifierCV(model['clf']))
    model.fit(trainData[:,selectedFeatures],trueImageLabels)
    weights_norm = [0]
    
    # Pick ROC Threshold
    cancerScores = model.predict_proba(trainData[:,selectedFeatures])[:,1]
    X,Y,T = roc_curve(trueImageLabels,cancerScores,drop_intermediate=False)
    maxY = max(Y[(X>0) & (X<=0.3)])
    minX = min(X[Y==maxY])
    indx = (X==minX) & (Y==maxY)
    ################### Classify VALIDATION Sample #######################
    testCancerScore = model.predict_proba(testData[selectedFeatures].reshape(1,-1))[0][1]
    testLabel = (testCancerScore >= T[indx][0])*1 
    return weights_norm, testLabel, selectedFeatures, validationIndx

def EnsembleLOOCV(allIndx,testIndx,trueLabels,decimPartition_model1,decimData_model1,originalData_model1,originalPartition,
                  dataset_model2,w1):  
    
    print("FOLD:",testIndx+1)
    trainIndx = np.setdiff1d(allIndx,testIndx)
    trueImageLabels = trueLabels[trainIndx]
    
    """ MODEL 1 """
    # TRAINING 
    rows = [val for s1 in decimPartition_model1[trainIndx] for s2 in s1 for val in s2]
    trainDecimDataset = decimData_model1[rows,:-1] 
    trainDecimLabels = decimData_model1[rows,-1]  
    rows = [val for s1 in originalPartition[trainIndx] for s2 in s1 for val in s2]
    trainPixelDataset = originalData_model1[rows,:-1]     
    # Update Partition Indices 
    #trainDecimPartition = UpdatePartitionIndices(decimPartition_model1[trainIndx])  
    trainPixelPartition = UpdatePartitionIndices(originalPartition[trainIndx])  
    
    # TESTING
    rows = [val for s1 in originalPartition[testIndx] for val in s1]
    testData_model1 = originalData_model1[rows,:-1] 
        
    # PIPELINE
    model1 = Pipeline([('scaler', StandardScaler()), 
                      ('clf',LinearSVC(penalty='l1',dual=False,C=0.1,class_weight='balanced',max_iter=40000,tol=1e-2))])    
    model1.fit(trainDecimDataset,trainDecimLabels)
    weights_norm = abs(model1['clf'].coef_)/np.sum(abs(model1['clf'].coef_)) # Take absolute value of weights and normalize
    weights_norm_model1 = np.round(weights_norm,3).ravel() # Round to 3 decimals
    model1.steps[1] = ('clf',CalibratedClassifierCV(model1['clf']))
    model1.fit(trainDecimDataset,trainDecimLabels)
    pp = model1.predict_proba(trainPixelDataset)[:,1]
    scores_model1 = np.zeros(trueImageLabels.size)
    for z in range(trueImageLabels.size):
        scores_model1[z] = np.mean(pp[trainPixelPartition[z][0]])  
        
    """ MODEL 2 """
    # TRAINING 
    trainData = dataset_model2[trainIndx,:]
    
    # TESTING
    testData_model2 = dataset_model2[testIndx,:] 
    
    # PIPELINE
    model2 = Pipeline([('scaler', StandardScaler()), 
                              ('clf',LinearSVC(penalty='l1',dual=False,C=1.1,class_weight='balanced', max_iter=100000,tol=1e-2))])
    model2.fit(trainData,trueImageLabels)
    weights_norm = abs(model2['clf'].coef_)/np.sum(abs(model2['clf'].coef_)) # Take absolute value of weights and normalize
    weights_norm_model2 = np.round(weights_norm,3).ravel() # Round to 3 decimals
    model2.steps[1] = ('clf',CalibratedClassifierCV(model2['clf']))
    model2.fit(trainData,trueImageLabels)  
    scores_model2 = model2.predict_proba(trainData)[:,1]
    
    """ COMBINE PP SCORES AND PICK THRESHOLD """
    weighted_scores = w1*scores_model1 + (1-w1)*scores_model2
    X,Y,T = roc_curve(trueImageLabels,weighted_scores,drop_intermediate=False)
    maxY = max(Y[(X>0) & (X<=0.3)])
    minX = min(X[Y==maxY])
    indx = (X==minX) & (Y==maxY)
    T_opt = T[indx][0]
    
    """ CLASSIFY TEST SET """
    test_score1 = np.mean(model1.predict_proba(testData_model1)[:,1]) 
    test_score2 = model2.predict_proba(testData_model2.reshape(1,-1))[0][1] 
    test_weighted_score = w1*test_score1 + (1-w1)*test_score2
    predSampleLabel = (test_weighted_score >= T_opt)*1
    return T_opt, predSampleLabel, weights_norm_model1, weights_norm_model2, testIndx     

def EnsembleLOOCV_V2(allIndx,testIndx,trueLabels,decimPartition_model1,decimData_model1,originalData_model1,
                     decimPartition_model2,decimData_model2,originalData_model2,originalPartition):  
    
    print("FOLD:",testIndx+1)
    trainIndx = np.setdiff1d(allIndx,testIndx) # Indices of complete training set
    trainImageLabels = trueLabels[trainIndx] # Labels of complete training set
    ######### HYPERPARAMETER OPTIMIZATION ####################################################################################
    nFolds = trainIndx.size
    C_param1, C_param2 = [0.1,0.3,0.5], [0.01,0.1,1]
    predLabels_model1 = np.array([0]*nFolds)
    predLabels_model2 = np.array([0]*nFolds)
    sen_model1 = [0]*len(C_param1)
    sen_model2 = [0]*len(C_param2)
    features_model1 = np.zeros((nFolds,len(C_param1))).tolist()
    features_model2 = np.zeros((nFolds,len(C_param2))).tolist()
    for k in range(len(C_param1)):
        for i, valIndx in enumerate(trainIndx): # LOOCV
            trainIndx2 = np.setdiff1d(trainIndx,valIndx) 
            trainImageLabels2 = trueLabels[trainIndx2] 
            """ MODEL 1 """
            # TRAINING 
            rows = [val for s1 in decimPartition_model1[trainIndx2] for s2 in s1 for val in s2]
            trainDecimDataset = decimData_model1[rows,:-1] 
            trainDecimLabels = decimData_model1[rows,-1]  
            rows = [val for s1 in originalPartition[trainIndx2] for s2 in s1 for val in s2]
            trainPixelDataset = originalData_model1[rows,:-1]     
            trainPixelPartition = UpdatePartitionIndices(originalPartition[trainIndx2])  # Update Partition Indices
            # TESTING
            rows = [val for s1 in originalPartition[valIndx] for val in s1]
            testData_model1 = originalData_model1[rows,:-1] 
            # PIPELINE
            model1 = Pipeline([('scaler', StandardScaler()), 
                              ('clf',LinearSVC(penalty='l1',dual=False,C=C_param1[k],class_weight='balanced',
                                               max_iter=40000,tol=1e-2))])    
            model1.fit(trainDecimDataset,trainDecimLabels)
            weights_norm_model1 = abs(model1['clf'].coef_)/np.sum(abs(model1['clf'].coef_)) # Take absolute value of weights and normalize
            weights_norm_model1 = np.round(weights_norm_model1,3).ravel() # Round to 3 decimals
            selectedFeatures = np.argsort(weights_norm_model1)[::-1][:3].tolist() # Sort in descending order and pick top features
            selectedFeatures.sort()
            features_model1[i][k] = selectedFeatures
            model1.steps[1] = ('clf',CalibratedClassifierCV(model1['clf']))
            model1.fit(trainDecimDataset[:,selectedFeatures],trainDecimLabels)
            pp = model1.predict_proba(trainPixelDataset[:,selectedFeatures])[:,1]
            scores_model1 = np.zeros(trainImageLabels2.size)
            for z in range(trainImageLabels2.size):
                scores_model1[z] = np.mean(pp[trainPixelPartition[z][0]])  
            # ROC THRESHOLD
            X,Y,T = roc_curve(trainImageLabels2,scores_model1,drop_intermediate=False)
            maxY = max(Y[(X>0) & (X<=0.3)])
            minX = min(X[Y==maxY])
            indx = (X==minX) & (Y==maxY)
            T_opt = T[indx][0]
            # CLASSIFY OUT OF FOLD SAMPLE
            score = np.mean(model1.predict_proba(testData_model1[:,selectedFeatures])[:,1])
            predLabels_model1[i] = (score >= T_opt)*1
            """ MODEL 2 """
            # TRAINING 
            rows = [val for s1 in decimPartition_model2[trainIndx2] for s2 in s1 for val in s2]
            trainDecimDataset = decimData_model2[rows,:-1] 
            trainDecimLabels = decimData_model2[rows,-1]  
            rows = [val for s1 in originalPartition[trainIndx2] for s2 in s1 for val in s2]
            trainPixelDataset = originalData_model2[rows,:-1]     
            trainPixelPartition = UpdatePartitionIndices(originalPartition[trainIndx2])  # Update Partition Indices
            # TESTING
            rows = [val for s1 in originalPartition[valIndx] for val in s1]
            testData_model2 = originalData_model2[rows,:-1] 
            '''
            # TRAINING 
            trainData = dataset_model2[trainIndx2,:]
            # TESTING
            testData_model2 = dataset_model2[valIndx,:] 
            '''
            # PIPELINE
            model2 = Pipeline([('scaler', StandardScaler()), 
                                      ('clf',LinearSVC(penalty='l1',dual=False,C=C_param2[k],class_weight='balanced',
                                                       max_iter=100000,tol=1e-2))])
            model2.fit(trainDecimDataset,trainDecimLabels)
            weights_norm_model2 = abs(model2['clf'].coef_)/np.sum(abs(model2['clf'].coef_)) # Take absolute value of weights and normalize
            weights_norm_model2 = np.round(weights_norm_model2,3).ravel() # Round to 3 decimals
            selectedFeatures = np.argsort(weights_norm_model2)[::-1][:3].tolist() # Sort in descending order and pick top features
            selectedFeatures.sort()
            features_model2[i][k] = selectedFeatures
            model2.steps[1] = ('clf',CalibratedClassifierCV(model2['clf']))
            model2.fit(trainDecimDataset[:,selectedFeatures],trainDecimLabels)  
            pp = model2.predict_proba(trainPixelDataset[:,selectedFeatures])[:,1]
            scores_model2 = np.zeros(trainImageLabels2.size)
            for z in range(trainImageLabels2.size):
                scores_model2[z] = np.mean(pp[trainPixelPartition[z][0]])
            # ROC THRESHOLD
            X,Y,T = roc_curve(trainImageLabels2,scores_model2,drop_intermediate=False)
            maxY = max(Y[(X>0) & (X<=0.3)])
            minX = min(X[Y==maxY])
            indx = (X==minX) & (Y==maxY)
            T_opt = T[indx][0]
            # CLASSIFY OUT OF FOLD SAMPLE
            score = np.mean(model2.predict_proba(testData_model2[:,selectedFeatures])[:,1])
            predLabels_model2[i] = (score >= T_opt)*1    
        sen_model1[k] = round(recall_score(trainImageLabels,predLabels_model1),2)
        sen_model2[k] = round(recall_score(trainImageLabels,predLabels_model2),2)
    """ OPTIMAL HYPERPARAMETERS and MOST FREQUENT FEATURES """
    index1 = sen_model1.index(max(sen_model1))
    index2 = sen_model2.index(max(sen_model2))
    opt_C1 = C_param1[index1]
    opt_C2 = C_param2[index2]
    features_model1 = [vals for l1 in features_model1 for vals in l1[index1]]
    selectedFeatures1 = [t[0] for t in Counter(features_model1).most_common()[:3]] # Three most frequent features
    features_model2 = [vals for l1 in features_model2 for vals in l1[index2]]
    selectedFeatures2 = [t[0] for t in Counter(features_model2).most_common()[:3]] # Three most frequent features
    ######### ENSEMBLE WEIGHT OPTIMIZATION ####################################################################################
    train_scores1 = np.zeros((nFolds-1,nFolds))
    train_scores2 = np.zeros((nFolds-1,nFolds))
    test_scores1 = np.zeros((nFolds,1))
    test_scores2 = np.zeros((nFolds,1))
    trainLabels = np.zeros((nFolds-1,nFolds))
    for i, valIndx in enumerate(trainIndx):# LOOCV
        trainIndx2 = np.setdiff1d(trainIndx,valIndx) 
        trainImageLabels2 = trueLabels[trainIndx2] 
        trainLabels[:,i] = trainImageLabels2
        """ MODEL 1 """
        # TRAINING 
        rows = [val for s1 in decimPartition_model1[trainIndx2] for s2 in s1 for val in s2]
        trainDecimDataset = decimData_model1[rows,:-1] 
        trainDecimLabels = decimData_model1[rows,-1]  
        rows = [val for s1 in originalPartition[trainIndx2] for s2 in s1 for val in s2]
        trainPixelDataset = originalData_model1[rows,:-1]     
        trainPixelPartition = UpdatePartitionIndices(originalPartition[trainIndx2])  # Update Partition Indices
        # TESTING
        rows = [val for s1 in originalPartition[valIndx] for val in s1]
        testData_model1 = originalData_model1[rows,:-1] 
        # PIPELINE
        model1 = Pipeline([('scaler', StandardScaler()), 
                          ('clf',LinearSVC(penalty='l1',dual=False,C=opt_C1,class_weight='balanced',
                                           max_iter=40000,tol=1e-2))])    
        model1.steps[1] = ('clf',CalibratedClassifierCV(model1['clf']))
        model1.fit(trainDecimDataset[:,selectedFeatures1],trainDecimLabels)
        pp = model1.predict_proba(trainPixelDataset[:,selectedFeatures1])[:,1]
        for z in range(trainImageLabels2.size):
            train_scores1[z,i] = np.mean(pp[trainPixelPartition[z][0]])  
        # CLASSIFY OUT OF FOLD SAMPLE
        test_scores1[i] = np.mean(model1.predict_proba(testData_model1[:,selectedFeatures1])[:,1])
        """ MODEL 2 """
        # TRAINING 
        rows = [val for s1 in decimPartition_model2[trainIndx2] for s2 in s1 for val in s2]
        trainDecimDataset = decimData_model2[rows,:-1] 
        trainDecimLabels = decimData_model2[rows,-1]  
        rows = [val for s1 in originalPartition[trainIndx2] for s2 in s1 for val in s2]
        trainPixelDataset = originalData_model2[rows,:-1]     
        trainPixelPartition = UpdatePartitionIndices(originalPartition[trainIndx2])  # Update Partition Indices
        # TESTING
        rows = [val for s1 in originalPartition[valIndx] for val in s1]
        testData_model2 = originalData_model2[rows,:-1]
        '''
        # TRAINING 
        trainData = dataset_model2[trainIndx2,:]
        # TESTING
        testData_model2 = dataset_model2[valIndx,:] 
        '''
        # PIPELINE
        model2 = Pipeline([('scaler', StandardScaler()), 
                                  ('clf',LinearSVC(penalty='l1',dual=False,C=opt_C2,class_weight='balanced',
                                                   max_iter=100000,tol=1e-2))])
        model2.steps[1] = ('clf',CalibratedClassifierCV(model2['clf']))
        model2.fit(trainDecimDataset[:,selectedFeatures2],trainDecimLabels)  
        pp = model2.predict_proba(trainPixelDataset[:,selectedFeatures2])[:,1]
        for z in range(trainImageLabels2.size):
            train_scores2[z,i] = np.mean(pp[trainPixelPartition[z][0]])
        # CLASSIFY OUT OF FOLD SAMPLE
        test_scores2[i] = np.mean(model2.predict_proba(testData_model2[:,selectedFeatures2])[:,1])
    """ WEIGHT OPTIMIZATION """
    w1 = np.arange(0.05,1,0.05) # Normalized weights between 0 and 1
    f1_w1 = [0]*w1.size
    testLabels = [0]*nFolds
    for i, w in enumerate(w1):
        for j in range(nFolds):
            weighted_scores = w*train_scores1[:,j] + (1-w)*train_scores2[:,j]
            # ROC THRESHOLD
            X,Y,T = roc_curve(trainLabels[:,j],weighted_scores,drop_intermediate=False)
            maxY = max(Y[(X>0) & (X<=0.3)])
            minX = min(X[Y==maxY])
            indx = (X==minX) & (Y==maxY)
            T_opt = T[indx][0]
            test_weighted_score = w*test_scores1[j] + (1-w)*test_scores2[j]
            testLabels[j] = (test_weighted_score >= T_opt)*1
        f1_w1[i] = round(f1_score(trainImageLabels,testLabels),2)
    opt_w1 = w1[f1_w1.index(max(f1_w1))]
    ######### TEST SET CLASSIFICATION #################################################################################### 
    """ MODEL 1 """
    # TRAINING 
    rows = [val for s1 in decimPartition_model1[trainIndx] for s2 in s1 for val in s2]
    trainDecimDataset = decimData_model1[rows,:-1] 
    trainDecimLabels = decimData_model1[rows,-1]  
    rows = [val for s1 in originalPartition[trainIndx] for s2 in s1 for val in s2]
    trainPixelDataset = originalData_model1[rows,:-1]     
    trainPixelPartition = UpdatePartitionIndices(originalPartition[trainIndx])  # Update Partition Indices
    # TESTING
    rows = [val for s1 in originalPartition[testIndx] for val in s1]
    testData_model1 = originalData_model1[rows,:-1] 
    # PIPELINE
    model1 = Pipeline([('scaler', StandardScaler()), 
                      ('clf',LinearSVC(penalty='l1',dual=False,C=opt_C1,class_weight='balanced',
                                       max_iter=40000,tol=1e-2))])    
    model1.steps[1] = ('clf',CalibratedClassifierCV(model1['clf']))
    model1.fit(trainDecimDataset[:,selectedFeatures1],trainDecimLabels)
    pp = model1.predict_proba(trainPixelDataset[:,selectedFeatures1])[:,1]
    scores_model1 = np.zeros(trainImageLabels.size)
    for z in range(trainImageLabels.size):
        scores_model1[z] = np.mean(pp[trainPixelPartition[z][0]])  
    """ MODEL 2 """
    # TRAINING 
    rows = [val for s1 in decimPartition_model2[trainIndx] for s2 in s1 for val in s2]
    trainDecimDataset = decimData_model2[rows,:-1] 
    trainDecimLabels = decimData_model2[rows,-1]  
    rows = [val for s1 in originalPartition[trainIndx] for s2 in s1 for val in s2]
    trainPixelDataset = originalData_model2[rows,:-1]     
    trainPixelPartition = UpdatePartitionIndices(originalPartition[trainIndx])  # Update Partition Indices
    # TESTING
    rows = [val for s1 in originalPartition[testIndx] for val in s1]
    testData_model2 = originalData_model2[rows,:-1]
    '''
    # TRAINING 
    trainData = dataset_model2[trainIndx,:]
    # TESTING
    testData_model2 = dataset_model2[testIndx,:] 
    '''
    # PIPELINE
    model2 = Pipeline([('scaler', StandardScaler()), 
                              ('clf',LinearSVC(penalty='l1',dual=False,C=opt_C2,class_weight='balanced', 
                                               max_iter=100000,tol=1e-2))])
    model2.steps[1] = ('clf',CalibratedClassifierCV(model2['clf']))
    model2.fit(trainDecimDataset[:,selectedFeatures2],trainDecimLabels)  
    pp = model2.predict_proba(trainPixelDataset[:,selectedFeatures2])[:,1]
    scores_model2 = np.zeros(trainImageLabels.size)
    for z in range(trainImageLabels.size):
        scores_model2[z] = np.mean(pp[trainPixelPartition[z][0]])
    """ COMBINE PP SCORES AND PICK THRESHOLD """
    weighted_scores = opt_w1*scores_model1 + (1-opt_w1)*scores_model2
    X,Y,T = roc_curve(trainImageLabels,weighted_scores,drop_intermediate=False)
    maxY = max(Y[(X>0) & (X<=0.3)])
    minX = min(X[Y==maxY])
    indx = (X==minX) & (Y==maxY)
    T_opt = T[indx][0]
    """ CLASSIFY TEST SET """
    test_score1 = np.mean(model1.predict_proba(testData_model1[:,selectedFeatures1])[:,1]) 
    test_score2 = np.mean(model2.predict_proba(testData_model2[:,selectedFeatures2])[:,1])
    test_weighted_score = opt_w1*test_score1 + (1-opt_w1)*test_score2
    predTestLabel = (test_weighted_score >= T_opt)*1
    return predTestLabel,selectedFeatures1,selectedFeatures2,opt_C1,opt_C2,opt_w1,testIndx     

def L1_SVM_KFOLDCV(allIndx,validationIndx,trainLabels,train_decimPartition,train_decimData,
                       train_originalPartition,train_originalData,param,foldIndx):   
    print("FOLD:",foldIndx+1)
    trainingIndx = np.setdiff1d(allIndx,validationIndx)
    trueImageLabels = trainLabels[trainingIndx]
    
    # TRAINING 
    rows = [val for s1 in train_decimPartition[trainingIndx] for s2 in s1 for val in s2]
    trainDecimDataset = train_decimData[rows,:-1] 
    trainDecimLabels = train_decimData[rows,-1]  
    rows = [val for s1 in train_originalPartition[trainingIndx] for s2 in s1 for val in s2]
    trainPixelDataset = train_originalData[rows,:-1]   
    # Update Partition Indices   
    trainPixelPartition = UpdatePartitionIndices(train_originalPartition[trainingIndx])  
    testPixelPartition = UpdatePartitionIndices(train_originalPartition[validationIndx])
    
    # TESTING
    rows = [val for s1 in train_originalPartition[validationIndx] for s2 in s1 for val in s2]
    testPixelDataset = train_originalData[rows,:-1] 
    
    # PIPELINE
    '''
    model = Pipeline([('scaler',StandardScaler()),('clf',RandomForestClassifier(n_estimators=param[0],max_depth=param[1],
                                                                    criterion='entropy',bootstrap=True,class_weight='balanced'))])     
    model.fit(trainDecimDataset,trainDecimLabels)
    importance = model['clf'].feature_importances_
    importance = np.round(importance,2)  
    selectedFeatures = np.argsort(importance)[::-1][:3].tolist() # Sort in descending order and pick top features
    selectedFeatures.sort()
    #model.steps[1] = ('clf',RandomForestClassifier(n_estimators=param[0],max_depth=param[1],criterion='entropy',
     #                                              bootstrap=True,class_weight='balanced',max_features=3))
    model.fit(trainDecimDataset[:,selectedFeatures],trainDecimLabels)
    '''
    model = Pipeline([('scaler', StandardScaler()), 
                      ('clf',LinearSVC(penalty='l1',dual=False,C=param,class_weight='balanced',max_iter=50000,tol=1e-4))])     
    model.fit(trainDecimDataset,trainDecimLabels)
    weights_norm = abs(model['clf'].coef_)/np.sum(abs(model['clf'].coef_)) # Take absolute value of weights and normalize
    weights_norm = np.round(weights_norm,2).ravel() # Round to 2 decimals
    #selectedFeatures = np.argsort(weights_norm)[::-1][:3].tolist() # Sort in descending order and pick top features
    #selectedFeatures.sort()
    model.steps[1] = ('clf',CalibratedClassifierCV(model['clf']))
    model.fit(trainDecimDataset,trainDecimLabels)
    
    # ROC THRESHOLD
    cancerProbs = model.predict_proba(trainPixelDataset)[:,1]
    cancerScores = [0]*trainPixelPartition.size
    for z in range(trainPixelPartition.size):
        cancerScores[z] = np.mean(cancerProbs[trainPixelPartition[z][0]])
    X,Y,T = roc_curve(trueImageLabels,cancerScores,drop_intermediate=False)
    maxY = max(Y[(X>0) & (X<=0.3)])
    minX = min(X[Y==maxY])
    indx = (X==minX) & (Y==maxY)
    
    ################### Classify VALIDATION Samples #######################
    ppValidation = model.predict_proba(testPixelDataset)[:,1]
    cancerScores = [0]*testPixelPartition.size
    for z in range(testPixelPartition.size):
        cancerScores[z] = np.mean(ppValidation[testPixelPartition[z][0]])
    predSampleLabels = (cancerScores >= T[indx][0])*1    
    return ppValidation, predSampleLabels, weights_norm, foldIndx    