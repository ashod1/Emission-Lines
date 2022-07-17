from astropy.io import fits
from astropy.table import Table,join
import numpy as np
import pylab as plt
import random
from scipy import stats
from sklearn.neighbors import KDTree
import time
from sklearn.metrics import mean_squared_error
from desitarget.targetmask import desi_mask, bgs_mask, mws_mask
from tensorflow import keras
from tensorflow.keras import layers

## DATA ##
## I'm combining fastphot,fastspect, and ztile to make sure I use the same data everywhere ##

zall_path="/project/projectdirs/desi/spectro/redux/everest/zcatalog/ztile-main-bright-cumulative.fits"
data1=Table.read(zall_path,hdu=1)
needed1=["TARGETID","BGS_TARGET","SPECTYPE","DELTACHI2","Z","ZWARN"]

fastspec_path = "/project/projectdirs/desi/spectro/fastspecfit/everest/catalogs/fastspec-everest-main-bright.fits"
data2=Table.read(fastspec_path,hdu=1)
needed2=["TARGETID","OII_3726_EW","OII_3729_EW","HGAMMA_EW","HBETA_EW","OIII_4959_EW","OIII_5007_EW","NII_6548_EW","HALPHA_EW","NII_6584_EW","SII_6716_EW","SII_6731_EW",\
        "OII_3726_EW_IVAR","OII_3729_EW_IVAR","HGAMMA_EW_IVAR","HBETA_EW_IVAR","OIII_4959_EW_IVAR","OIII_5007_EW_IVAR","NII_6548_EW_IVAR","HALPHA_EW_IVAR","NII_6584_EW_IVAR","SII_6716_EW_IVAR","SII_6731_EW_IVAR"]

file_path = "/project/projectdirs/desi/spectro/fastspecfit/everest/catalogs/fastphot-everest-main-bright.fits"
data3=Table.read(file_path,hdu=1)
needed3=["TARGETID","ABSMAG_SDSS_U","ABSMAG_SDSS_G","ABSMAG_SDSS_R","ABSMAG_SDSS_I","ABSMAG_SDSS_Z"]

data4=join(data1[needed1],data2[needed2],keys="TARGETID")
data=join(data4,data3[needed3],keys="TARGETID")

## Adding the sum of OII doublets to use them as a single line
data.add_column(data["OII_3726_EW"]+data["OII_3729_EW"],name='OII_DOUBLET_EW')
data.add_column(1/(1/data["OII_3726_EW_IVAR"]+1/data["OII_3729_EW_IVAR"]),name='OII_DOUBLET_EW_IVAR')

## Selecting data and doing LLR to predict lines ##
lines=["OII_DOUBLET_EW","HGAMMA_EW","HBETA_EW","OIII_4959_EW","OIII_5007_EW","NII_6548_EW","HALPHA_EW"\
       ,"NII_6584_EW","SII_6716_EW","SII_6731_EW"]

magnitude_names=["ABSMAG_SDSS_U","ABSMAG_SDSS_G","ABSMAG_SDSS_R","ABSMAG_SDSS_I","ABSMAG_SDSS_Z"]

N=len(data["TARGETID"])
snr_cut=1 # signal to noise ratio cut
n=25*10**3 

# calculating snr for all lines
snr_all=np.zeros([N,len(lines)])
snr_all[:,0]=data[lines[0]]*np.sqrt(data[lines[0]+"_IVAR"])

for i in range(1,len(lines)):
    snr_all[:,i]=data[lines[i]]*np.sqrt(data[lines[i]+"_IVAR"])

select_pos=snr_all[:,6]>=1
select_neg=snr_all[:,6]<1

y=np.zeros(N)
y[select_pos]=1

# calculating minimum redshift to have de-redshifted wavelengths be in the interval 3400,7000 A

w1=3400
w_min=3600
z_min=w_min/w1-1

select=(data["SPECTYPE"]=="GALAXY")*(data["DELTACHI2"]>=25)*(data["Z"]>z_min)*(data["Z"]<0.3)*(data["ZWARN"]==0)
target_pos=np.where(select)[0][:n]

magnitudes_s=data[magnitude_names][target_pos]  
magnitudes=np.zeros([n,len(magnitude_names)])
for j in range(len(magnitude_names)):
    magnitudes[:,j]=magnitudes_s[magnitude_names[j]][:n]

# Getting features as colors and regularizing them    
#ones=np.ones([n,1])
x=np.zeros([n,len(magnitude_names)-1])
for i in range(n):
    for j in range(len(magnitude_names)-1):
        x[i,j]=magnitudes[i,j]-magnitudes[i,j+1]
av_x=np.zeros(x.shape[1])
std_x=np.zeros(x.shape[1])
for i in range(x.shape[1]):
    av_x[i]=np.average(x[:,i])
    std_x[i]=np.std(x[:,i])
    x[:,i]=(x[:,i]-av_x[i])/std_x[i]
        
#x=np.concatenate((ones,x),axis=1)

y=y_all[target_pos]

N_cv=10
x_split=np.split(x, N_cv)
y_split=np.split(y, N_cv)

for i in range(N_cv):
    x_valid=x_split[i]
    y_valid=y_split[i]
    x_to_combine=[]
    y_to_combine=[]
    for j in range(N_cv):
        if j!=i:
            x_to_combine.append(x_split[j])
            y_to_combine.append(y_split[j])
    x_train=np.concatenate(tuple(x_to_combine),axis=0)
    y_train=np.concatenate(tuple(y_to_combine),axis=0)
    
    model = keras.Sequential([
        layers.Dense(4, activation='sigmoid', input_dim=x.shape[1]),
        layers.Dense(4, activation='sigmoid'),    
        layers.Dense(1, activation='sigmoid'),
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])

    early_stopping = keras.callbacks.EarlyStopping(
        patience=10,
        min_delta=0.001,
        restore_best_weights=True,
    )
    
    model.fit(x_train, y_train, validation_data=(x_valid,y_valid), batch_size=10, epochs=1000, callbacks=[early_stopping], verbose=0)
    y_pred=model.predict(x_valid).reshape(len(x_valid))
