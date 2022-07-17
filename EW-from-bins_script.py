import numpy as np
import pylab as plt
from scipy import stats
from sklearn.neighbors import KDTree
import time
from sklearn.metrics import mean_squared_error
from os import listdir
import scipy
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from tensorflow import keras
from tensorflow.keras import layers
import xgboost as xgb
from LLR import LLR
import argparse

# Which line to use is set as an argument for the script
parser=argparse.ArgumentParser()
parser.add_argument('l', type=int)
args=parser.parse_args()

# setting up parameters

lines=["OII_DOUBLET_EW","HGAMMA_EW","HBETA_EW","OIII_4959_EW","OIII_5007_EW","NII_6548_EW"\
       ,"HALPHA_EW","NII_6584_EW","SII_6716_EW","SII_6731_EW"]
l=args.l     # line index
run=0         # selection index 
run_out=2     # output index
m=0           # model index. 0 is LLR, 1 is RandomForest, 2 is GradientBoosting from sklearn, 3 is XGboost, 4 is neural network
loga=True     # if true then uses log(EWs)
Ns=[6,11,16,21,26,31,41,51]     #different bin numbers
#Ns=[41]

for N in Ns:
    # importing fluxes
    fluxes_bin=np.load("/global/cscratch1/sd/ashodkh/results/fluxes_bin_selection"+str(1)+"_"+str(lines[l])+"_bins"+str(N)+".txt.npz")["arr_0"]
    #zs=np.load("/global/cscratch1/sd/ashodkh/results/zs_selection"+str(run)+"_"+str(lines[l])+".txt.npz")["arr_0"]
    target_lines=np.load("/global/cscratch1/sd/ashodkh/results/target_lines_selection"+str(run)+"_"+str(lines[l])+".txt.npz")["arr_0"]

    ## selecting only positive fluxes and saving them to use in other codes
    select_fluxes=fluxes_bin[:,0]>0
    for i in range(1,fluxes_bin.shape[1]):
        select_fluxes=select_fluxes*(fluxes_bin[:,i]>0)
    np.savez_compressed("/global/cscratch1/sd/ashodkh/results/select_positive_fluxes_selection"+str(1)+"_"+str(lines[l])+"_bins"+str(N)+".txt", select_fluxes)
    fluxes_bin=fluxes_bin[select_fluxes,:]
    target_lines=target_lines[select_fluxes]

    n=25*10**3    #number of data points that will be used
    # getting features and outcomes
    EWs=np.zeros([n,len(lines)])
    magnitudes=np.zeros([n,fluxes_bin.shape[1]])
    for i in range(n):
        magnitudes[i,:]=-2.5*np.log10(fluxes_bin[i,:])
        for j in range(len(lines)):
            EWs[i,j]=target_lines[i][j] # need to have this in a loop so that EWs is a normal numpy array

    ones=np.ones([n,1])
    x=np.zeros([n,magnitudes.shape[1]-1])
    for i in range(n):
        for j in range(magnitudes.shape[1]-1):
            x[i,j]=magnitudes[i,j]-magnitudes[i,j+1]
    # regularizing colors
    av_x=np.zeros(x.shape[1])
    std_x=np.zeros(x.shape[1])
    for i in range(x.shape[1]):
        av_x[i]=np.average(x[:,i])
        std_x[i]=np.std(x[:,i])
        x[:,i]=(x[:,i]-av_x[i])/std_x[i]
    # adding ones if model is LLR (for y-intercept)    
    if m==0:
        x=np.concatenate((ones,x),axis=1)

    # Old version looped over the lines here, but I always ended up doing it for 1 line.
    if loga:
        EW=np.log10(EWs[:,l])
    else:
        EW=EWs[:,l]
        
    # splits data into 90-10 for cross validation
    N_cv=10
    x_split=np.split(x,N_cv)
    EW_split=np.split(EW,N_cv)

    EW_fit_all=[]
    EW_obs_all=[]

    spearman_all=[]
    nmad_all=[]
    for i in range(N_cv):
        ## assigning the training and validation sets
        x_valid=x_split[i]
        EW_valid=EW_split[i]

        x_to_combine=[]
        EW_to_combine=[]
        for j in range(N_cv):
            if j!=i:
                x_to_combine.append(x_split[j])
                EW_to_combine.append(EW_split[j])
        x_train=np.concatenate(tuple(x_to_combine),axis=0)
        EW_train=np.concatenate(tuple(EW_to_combine),axis=0)

        # predicting EWs using different models
        if m==0:
            EW_fit,zeros=LLR.LLR(x_valid, x_train, EW_train, 100, 'inverse_distance')
        if m==1:
            model=RandomForestRegressor(n_estimators=200)
            model.fit(x_train, EW_train)
            EW_fit=model.predict(x_valid)
        if m==2:
            model=GradientBoostingRegressor(n_estimators=100)
            model.fit(x_train, EW_train)
            EW_fit=model.predict(x_valid)
        if m==3:
            model=xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05)
            model.fit(x_train, EW_train, early_stopping_rounds=5, eval_set=[(x_valid,EW_valid)], verbose=False)
            EW_fit=model.predict(x_valid)
            print(model.best_ntree_limit)
        if m==4:
            model=keras.Sequential([
                layers.Dense(units=10, activation='sigmoid', input_shape=[x.shape[1]]),
                layers.Dense(units=5, activation='sigmoid'),
                layers.Dense(units=3, activation='sigmoid'),
                layers.Dense(units=1),
            ])
            model.compile(optimizer='Adam', loss='mse')
            model.fit(x_train, EW_train,batch_size=5)
            EW_fit=model.predict(x_valid)


        # calculating spearman coefficient and nmad for fit.
        nmad=np.abs(EW_fit-EW_valid)

        EW_fit_all.append(EW_fit)
        EW_obs_all.append(EW_valid)

        spearman_all.append(stats.spearmanr(EW_fit,EW_valid)[0])
        nmad_all.append(1.48*np.median(nmad))

    print(lines[l])
    print(spearman_all)
    print("av spearman = "+str(np.average(spearman_all)))
    print(nmad_all)
    print("av nmad = "+str(np.average(nmad_all)))
    print("\n")

    if loga:
        np.savez_compressed("/global/cscratch1/sd/ashodkh/ML_results/logEW_fit_bins_selection"+str(run_out)+"_line"+str(lines[l])+"_bins"+str(N)\
                            +"_ML"+str(m)+".txt",EW_fit_all)
        np.savez_compressed("/global/cscratch1/sd/ashodkh/ML_results/logEW_obs_bins_selection"+str(run_out)+"_line"+str(lines[l])+"_bins"+str(N)\
                            +"_ML"+str(m)+".txt",EW_obs_all)
    else:
        np.savez_compressed("/global/cscratch1/sd/ashodkh/ML_results/EW_fit_bins_selection"+str(run_out)+"_line"+str(lines[l])+"_bins"+str(N)\
                        +"_ML"+str(m)+".txt",EW_fit_all)
        np.savez_compressed("/global/cscratch1/sd/ashodkh/ML_results/EW_obs_bins_selection"+str(run_out)+"_line"+str(lines[l])+"_bins"+str(N)\
                        +"_ML"+str(m)+".txt",EW_obs_all)