# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 18:02:08 2018

@author: Takeru_2
"""

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import sys,os
from PIL import Image

gam=2.2

os.chdir('dataset')
matdata0 = scipy.io.loadmat("Cloth.mat")
os.chdir('../')
Ls0 = scipy.io.loadmat("L.mat")
cmf0 = scipy.io.loadmat("cmf.mat")
Ls = Ls0['L']
cmf = cmf0['cmf']
L=np.eye(31)

matdata=matdata0['img']
size=matdata.shape
matdata=matdata.reshape(size[0]*size[1],31)

Lsr=np.dot(matdata,Ls)

#Lr=matdata*L
invM=np.array([3.24096994190452,	-1.53738317757009,	-0.498610760293003,
-0.969243636280880,	1.87596750150772,	0.0415550574071756,
0.0556300796969937, -0.203976958888977,	1.05697151424288])

invM=invM.reshape(3,3)

sRGB=np.dot(Lsr,cmf)
sRGB=sRGB.reshape(size[0],size[1],3)

invgam=1/gam
sRGB_gamma= (np.clip(sRGB,0,1))**invgam 
            
plt.imshow(sRGB_gamma)
plt.show()
#
plt.imshow(sRGB)
plt.show()
#imgArray = np.asarray(sRGB)
pilImg = Image.fromarray(np.uint8(sRGB*255))
pilImg_gamma = Image.fromarray(np.uint8(sRGB_gamma*255))

#os.mkdir('sRGB')
os.chdir('sRGB')

# save Image
pilImg.save('img.jpg', 'JPEG', quality=size[0], optimize=True)
pilImg_gamma.save('img_gamma.jpg', 'JPEG', quality=size[0], optimize=True)
os.chdir('../')


