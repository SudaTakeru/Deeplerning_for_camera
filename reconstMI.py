# -*- coding: utf-8 -*-
"""
Created on Wed May 16 17:55:41 2018

@author: Takeru_2
"""


import scipy.io
from keras.layers import Input, Dense, Reshape, Multiply, Concatenate, Activation
from keras.models import Model 
from keras.layers import Lambda, Add, Conv2D,normalization
from keras.optimizers import Adam
from keras.regularizers import l1, l2, Regularizer
from keras import backend as K
import tensorflow as tf
import numpy as np
import os
import random
from PIL import Image
import math
import matplotlib.pyplot as plt
from unet_forMI import UNet

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#sys.path.append()
           
bandwidth=3
psize = 256
epochs = 300000
pe=30000
so = epochs/2
psize2 = psize
tp=4
last=30

def MItosRGB(data2,Ls,cmf,invM):
    size0=data2.shape
    data2=data2.reshape(size0[0]*size0[1],31)
    Lsr=np.dot(data2,Ls)
    invM=np.transpose(invM)   
    sRGB=np.dot(Lsr,cmf)
    sRGB=np.dot(sRGB,invM)
    sRGB=sRGB.reshape(size0[0],size0[1],3)
    return sRGB

def generator( data, psize, nb_image, nb_patch_per_image ):
	N = data.shape[0]
	W = data.shape[1]
	H = data.shape[2]
	C = data.shape[3]
    
	X = np.zeros( ( nb_image*nb_patch_per_image, psize[0], psize[1], C ), dtype = np.float32 )
	 
	while( True ):
		k = 0
		sampled_img = random.sample(range(N), nb_image)
		for img in sampled_img:
			pos = []
			for i in range(nb_patch_per_image):
				while( True ):
					whb = [\
					 random.randrange(0, W-psize[0]),\
					 random.randrange(0, H-psize[1]),\
					 random.randrange(0,8),\
					 ]
					if( not ( whb in pos ) ):
						pos.append( whb )
						break
			
			for whb in pos:
				x = data[ img, whb[0]:whb[0]+psize[0], whb[1]:whb[1]+psize[1], : ]
				b = whb[2]
				if( ( b & 0b100 ) > 0 ): #transpose
					x = np.swapaxes( x, 0, 1 )  
				if( ( b & 0b010 ) > 0 ): #mirror
					x = np.fliplr( x )
				if( ( b & 0b001 ) > 0 ): #flip
					x = np.flipud( x )

				X[k,:,:,:] = x
				k += 1
		yield (X,X)
        
def combine_images(generated_images):
    total = generated_images.shape[0]
    cols = int(math.sqrt(total))
    rows = math.ceil(float(total)/cols)
    width, height = generated_images.shape[1:3]
    combined_image = np.zeros((height*rows, width*cols),
                              dtype=generated_images.dtype)

    for index, image in enumerate(generated_images):
        i = int(index/cols)
        j = index % cols
        combined_image[width*i:width*(i+1), height*j:height*(j+1)] = image
    return combined_image        

def mse(predictions,targets):
    return np.sum(((predictions - targets) ** 2))/(predictions.shape[0]*predictions.shape[1]*predictions.shape[2]*predictions.shape[3])

def cpsnr(img1, img2):
    mse_tmp = mse(np.round(np.clip(img1,0,255)),np.round(np.clip(img2,0,255)))
    PIXEL_MAX = 255.0
    return 10 * math.log10(PIXEL_MAX**2 / mse_tmp)       


## DATA読み込み
flist=os.listdir('dataset/')
data0=list(range(len(flist)))
MIdata0=list(range(len(flist)))
sRGBdata=list(range(len(flist)))
os.chdir('dataset')

for i in range(len(flist)):
    data0[i] = scipy.io.loadmat(flist[i]) #Matlab_data
    if i==0:
        size=data0[0]['img'].shape
        MIdata0[i]=data0[i]['img']    
        maxv=np.amax(MIdata0[i])
        MI=np.array([MIdata0[i]/maxv])
    else:        
        MIdata0[i]=data0[i]['img'] 
        maxv=np.amax(MIdata0[i])
        MI=np.append(MI,np.array([MIdata0[i]/maxv]),axis=0)

os.chdir('../')
size=MIdata0[0].shape

## Load Initial Camera Sensitivity 
Sensitivity = scipy.io.loadmat('Camera_Sensitivity.mat') #Matlab_data     
S0 = Sensitivity['Sensitivity']
S = np.zeros((1,1,31,3))
S[0][0]=np.transpose(S0)                          
    
## sRGB画像作成
Ls0 = scipy.io.loadmat("L.mat")
cmf0 = scipy.io.loadmat("cmf.mat")
invM0 = scipy.io.loadmat("invM.mat")
Ls = Ls0['L']
cmf = cmf0['cmf']    
invM =invM0['invM'] 

sRGB_IMAGE_PATH = 'sRGB_images/' # 生成画像の保存先
sRGBg_IMAGE_PATH = 'sRGBg_images/' # 生成画像の保存先

gam=2.2
invgam=1/gam

            
for i in range(len(flist)):
    sRGBdata[i]=MItosRGB(MIdata0[i],Ls,cmf,invM)
    maxv=np.amax(sRGBdata[i])
    if i==0:
        sRGB=np.array([sRGBdata[i]/maxv])
    else:            
        sRGB=np.append(sRGB,np.array([sRGBdata[i]/maxv]),axis=0)
        
    if not os.path.exists(sRGB_IMAGE_PATH):
        os.mkdir(sRGB_IMAGE_PATH)
    rm=np.amax(sRGBdata[i])
    srgb=sRGBdata[i][:,:,0]/rm
    sRGBimg=np.uint8(srgb*255)
    Image.fromarray(sRGBimg)\
                   .save(sRGB_IMAGE_PATH+"%04d_%04d.png" % (i+1, 1), optimize=True)
    if not os.path.exists(sRGBg_IMAGE_PATH):
        os.mkdir(sRGBg_IMAGE_PATH)
    sRGB_gamma= np.uint8(np.power(srgb,invgam)*255)
    Image.fromarray(sRGB_gamma)\
                   .save(sRGBg_IMAGE_PATH+"%04d_%04d.png" % (i+1, 1), optimize=True)

# model
input_channel_count = 31
output_channel_count = 31
# 一番初めのConvolutionフィルタ枚数は64
first_layer_filter_count = 64
# U-Netの生成
g_opt = Adam(lr=0.001, beta_1=0.9,beta_2=0.999, epsilon=1e-8 )
network = UNet(input_channel_count, output_channel_count, first_layer_filter_count,S,psize)
model = network.get_model()
model.compile(loss='mean_squared_error', optimizer=g_opt)


encoder = network.get_submodel()
#decoder = Model(inputs=mosaicinput, outputs=[output])

encoder.compile(loss='mean_squared_error', optimizer=g_opt)


## Training
k=4
gen = generator( MI[:tp], [psize, psize], 1, k )
testgen = generator( MI[-tp:], [psize, psize], tp, k )


PREDICT_IMAGE_PATH = 'predict_MSimages/' # 生成画像の保存先
PREDICTG_IMAGE_PATH = 'predictganma_MSimages/' # 生成画像の保存先
ANS_IMAGE_PATH = 'answer_MSimages/' # 生成画像の保存先
ANSG_IMAGE_PATH = 'answerganma_MSimages/' # 生成画像の保存先

gam=2.2
invgam=1/gam
sRGB_gamma= (np.clip(sRGB,0,1))**invgam 
E=list(range(epochs))
trainE=list(range(epochs))
cpsnrs=list(range(int(epochs/pe)))
psnrs1=list(range(int(epochs/pe))) 
psnrs2=list(range(int(epochs/pe)))      
psnrs3=list(range(int(epochs/pe))) 

test=testgen.__next__()
f = open('psnrs.txt', 'w')
f.write('epoch  psnr  psnr:red  psnr:green psnr: bule'+"\n")
f2 = open('loss.txt', 'w')
f2.write('epoch  loss_training loss_testing'+"\n")

for epoch in range(epochs):
    epoch2 = epoch + 1    
    print("epoch: %d" % epoch2)        
    
    hist=model.fit_generator( gen, 4, epochs=1, verbose=1 )
    trainE[epoch]=hist.history['loss']
    
    
    E[epoch]=model.evaluate(test[0],test[1])
    f2.write(str(epoch2)+ ': '+str(trainE[epoch])+','+str(E[epoch])+ "\n")
    
    if epoch2 %pe==0:
        
        #model.save('msreconstruct.hdf5')
        #model.save_weights('msreconstruct_weight.hdf5')
        
        
        img = model.predict(test[0],batch_size=4)
        for i in range(31):
            image = combine_images(img[:,:,:,i])
            rm=np.amax(image[:,:])
            sr=image/rm
            image0=np.uint8(sr*255)
            imagegam= np.uint8(np.power(sr,invgam)*255)
            if not os.path.exists(PREDICT_IMAGE_PATH):
                os.mkdir(PREDICT_IMAGE_PATH)
            Image.fromarray(image0)\
                    .save(PREDICT_IMAGE_PATH+"%04d_%04d.png" % (epoch2/pe, i))
            if not os.path.exists(PREDICTG_IMAGE_PATH):
                os.mkdir(PREDICTG_IMAGE_PATH)
            Image.fromarray(imagegam)\
                   .save(PREDICTG_IMAGE_PATH+"%04d_%04d.png" % (epoch2/pe, i)) 
            if epoch2/pe==1:       
                image = combine_images(test[1][:,:,:,i])
                rm=np.amax(image[:,:])
                sr=image/rm
                image0=np.uint8(sr*255)
                imagegam= np.uint8(np.power(sr,invgam)*255)
                if not os.path.exists(ANS_IMAGE_PATH):
                    os.mkdir(ANS_IMAGE_PATH)
                Image.fromarray(image0)\
                        .save(ANS_IMAGE_PATH+"%04d_%04d.png" % (epoch2/pe, i))
                if not os.path.exists(ANSG_IMAGE_PATH):
                    os.mkdir(ANSG_IMAGE_PATH)
                Image.fromarray(imagegam)\
                        .save(ANSG_IMAGE_PATH+"%04d_%04d.png" % (epoch2/pe, i))                    
            
        cpsnrs[int(epoch2/pe)-1] = cpsnr(img,test[1])
        x=i*pe
        f.write(str(x)+ ': '+str(cpsnrs[int(epoch2/pe)-1])+ "\n")

f.close()
f2.close()

weights=model.get_weights()
Optw=weights[0][0,0,:,:]
np.save('Opt_Sensitivity.npy', Optw)

plt.plot(Optw)
plt.title("Sensitivity")
plt.xlabel("lambda")
plt.savefig("Optimazed_Sensitivity.png")
plt.show()              
                 
plt.plot(E)
plt.title("loss_testing")
plt.xlabel("batch")
plt.ylabel("loss")
plt.savefig("Evaluate_loss_testing.png")
plt.show()

plt.plot(E[-last:])
plt.title("loss_testing")
plt.xlabel("batch")
plt.ylabel("loss")
plt.savefig("Evaluate_loss_testing_fininary.png")
plt.show()

plt.plot(trainE)
plt.title("loss_training")
plt.xlabel("batch")
plt.ylabel("loss")
plt.savefig("Evaluate_loss_training.png")
plt.show()

plt.plot(trainE[-last:])
plt.title("loss_training")
plt.xlabel("batch")
plt.ylabel("loss")
plt.savefig("Evaluate_loss_training_fininary.png")
plt.show()



plt.plot(cpsnrs[-last:])
plt.title("psnr")
plt.xlabel("batch")
plt.ylabel("loss")
plt.savefig("psnr_fininary.png")
plt.show()


