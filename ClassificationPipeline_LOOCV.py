#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main Script for Leave-One-Patient-Out Cross-Validation (LOPOCV) of Cancerous vs. Benign Oral Lesions

@author: Elvis de Jesus Duran Sierra
"""

import time
import numpy as np
from scipy.io import loadmat, savemat
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from CustomFunctions_V2 import UpdatePartitionIndices, L1_SVM_LOOCV
from collections import Counter
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import multiprocessing as mp

""" LOAD FILES """ 
# LOAD DATA
dataset = 'SPEC_QATAR'
data = loadmat('/home/demo/Elvis/Python Codes/Classification/CAN_vs_BEN/Datasets/' + dataset)
train_decimData = data['decimData'][:,list(range(0,9))+[-1]]
train_originalData = data['originalData'][:,list(range(0,9))+[-1]]
train_decimPartition = data['decimPartition']
train_originalPartition = data['originalPartition']
# UPDATE PARTITION INDICES
train_decimPartition = UpdatePartitionIndices(train_decimPartition)
train_originalPartition = UpdatePartitionIndices(train_originalPartition)
trainLabels = np.squeeze(data['imageLabels'])
featNames = data['featNames']
featNames = [arr2[0] for arr1 in featNames for arr2 in arr1]
histology = data['histology']
histology  = np.array([arr2[0] for arr1 in histology for arr2 in arr1])

""" OPTIMIZE CLASSIFIERS """
np.random.seed(8)
nFolds = trainLabels.size
postProbs = np.zeros((nFolds,1)).tolist()
predSampleLabels = np.array([0]*nFolds)
#hyper1 = [0.01,0.1,1]
hyper1 = [1]
#hyper1 = [50,100,150]
#hyper2 = [2,3,4]
nComb = len(hyper1)
sen = [0]*nComb
spe = [0]*nComb
param_label = [0]*nComb
count = 0
allIndx = range(nFolds)
selectedFeatures = [0]*nFolds
featureWeights = np.zeros((nFolds,train_decimData.shape[-1]-1))
def collect_result(result):
    global results
    results.append(result)
start = time.time()

#L1_SVM_LOOCV(allIndx,0,trainLabels,train_decimPartition,train_decimData,train_originalPartition,train_originalData,1)

font = {
        'weight' : 'bold',
        'size'   : 11}
matplotlib.rc('font', **font)
barWidth, barWidth_h = 0.18, 0.23 
colors = sns.color_palette('deep',n_colors=4)
if __name__ == '__main__':
    for p1 in hyper1:
        param = p1
        #param = [p1,p2]
        #param_label[count] = 'nTrees=' + str(p1) + ';Depth=' + str(p2)
        #print('%%%%%%%%%%','RF('+param_label[count]+')','%%%%%%%%%%')
        param_label[count] = 'C='+str(p1)
        print('%%%%%%%%%%','SVM ('+param_label[count]+')','%%%%%%%%%%') 
        pool = mp.Pool(mp.cpu_count())
        results = []
        for validationIndx in range(nFolds):
            #pool.apply_async(Image_L1_SVM_LOOCV,args=(allIndx,validationIndx,trainLabels,trainDataset,
             #                                         CC,TOL),callback=collect_result)
            pool.apply_async(L1_SVM_LOOCV,args=(allIndx,validationIndx,trainLabels,train_decimPartition,train_decimData,
                             train_originalPartition,train_originalData,param),callback=collect_result)
        pool.close()
        pool.join()
        results.sort(key=lambda r: r[-1]) # Sort results based on fold number
        for i in range(nFolds):
            postProbs[i],predSampleLabels[i],selectedFeatures[i]= results[i][0],results[i][1],results[i][2]
        end = time.time() 
        print('Execution Time:',end-start,'sec') 
        
        """ FINAL CONFUSION MATRIX """
        confmat = confusion_matrix(trainLabels,predSampleLabels)
        sen[count] = round(confmat[1,1]/sum(confmat[1,:]),2)
        spe[count] = round(confmat[0,0]/sum(confmat[0,:]),2) 
        print(confmat)
        print(classification_report(trainLabels, predSampleLabels,target_names = ['Benign','Cancer/Precancer']))
        print(histology[np.where(predSampleLabels!=trainLabels)[0]])    
        
        # FREQUENCY OF SELECTED FEATURES
        fig, ax = plt.subplots(dpi=600)
        barTicks = []
        barLabels = []
        # Frequency of Individual Features (Bar Plot 1)
        featureList = [vals for l1 in selectedFeatures for vals in l1]
        mostCommon = Counter(featureList).most_common()
        freq = [t[1] for t in mostCommon]
        featNum = [t[0] for t in mostCommon]
        # Plot Bars
        barPos = [0 +  barWidth_h*i for i in range(len(freq)-1,-1,-1)] # The x position of bars
        bars = ax.barh(barPos,freq, height =  barWidth_h, color = colors[0], edgecolor = 'black')
        barTicks += barPos
        barLabels += featNum
        # Annotations on Top of Bars
        for indx, p in enumerate(barPos):
            ax.annotate('{}'.format(freq[indx]),xy=(freq[indx],p),xytext=(3,0),textcoords="offset points",va="center")
        # Format Bar Plot 1
        ax.axes.get_xaxis().set_ticks([])
        ax.set_title('Spectral Features (Absolute)',fontsize=15,fontweight='bold')
        ax.set_yticks(barTicks)
        ax.set_yticklabels(np.array(featNames)[barLabels].tolist(),fontsize=12,va='center')
        ax.set_xlabel('Number of Folds',fontsize=14,fontweight='bold')
        '''
        # PLOT FEATURE IMPORTANCES
        meanWeights = np.mean(featureWeights,0)
        stdWeights = np.std(featureWeights,0)
        sortedIndx = np.argsort(meanWeights)[::-1] # Descending order
        barWidth = 0.23
        colors = sns.color_palette('deep')
        fig1, ax1 = plt.subplots(dpi=600)
        freq = np.round(meanWeights[sortedIndx]*100).astype(int)
        std_freq = np.round(stdWeights[sortedIndx]*100).astype(int)
        barPos = [barWidth*i for i in range(freq.shape[0]-1,-1,-1)] # The y position of bars
        bars = ax1.barh(barPos,freq, height = barWidth, color = colors[0], edgecolor = 'black')
        # Annotations on Top of Bars
        for indx, p in enumerate(barPos):
            ax1.annotate('{}'.format(freq[indx])+'%'+'\u00B1'+format(std_freq[indx])+'%',xy=(freq[indx],p),xytext=(3,0),textcoords="offset points",va="center")
        ax1.set_title('Absolute Spectral Features',fontsize=15,fontweight='bold')
        ax1.set_yticks(barPos)
        ax1.set_yticklabels(np.array(featNames)[sortedIndx].tolist(),fontsize=12,va='center')
        ax1.set_xlabel('Mean Weight',fontsize=14,fontweight='bold')
        ax1.set_xlim(0,30)
        ax1.set_xticks([])
        '''
        count+=1

""" PLOT PERFORMANCE """
matplotlib.rc('font', **font)    
fig2, ax2 = plt.subplots(dpi=600)
ax2.plot(range(len(sen)),sen,'rd-',range(len(spe)),spe,'bs-')
#plt.xlabel('Hyperparameters',fontsize=15,fontweight='bold')
plt.xticks(range(len(sen)),labels=param_label,rotation=90)
plt.title('Hyperparameter Optimization',fontsize=15,fontweight='bold')
ax2.legend(['Sensitivity','Specificity'],loc='upper right',prop={'size': 10})
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax2.grid()
'''
# EXPORT RESULTS
folder = 'ALL'
y_pred = []
post_probs = []
for i in range(nFolds):
    y_pred += predSampleLabels[i].tolist()
    post_probs += postProbs[i].tolist() 
savemat('/home/demo/Elvis/Python Codes/Classification/CAN_vs_BEN/Results/'+ folder +'/y_pred.mat', {'y_pred':y_pred})
savemat('/home/demo/Elvis/Python Codes/Classification/CAN_vs_BEN/Results/'+ folder +'/post_probs.mat', {'post_probs':post_probs})
'''


