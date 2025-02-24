#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 22 May 2019

@author: M Roux
"""

############## le clear('all') de Matlab
for name in dir():
    if not name.startswith('_'):
        del globals()[name]
################################"

import math
import numpy as np
import matplotlib.pyplot as plt
import skimage as sk

from scipy import ndimage
from scipy import signal

from skimage import io

##############################################

def dericheGradX(ima,alpha):

    
    nl,nc=ima.shape
    ae=math.exp(-alpha)
    c=-(1-ae)*(1-ae)/ae

    b1=np.zeros(nc)
    b2=np.zeros(nc)

    gradx=np.zeros((nl,nc))


#gradx=np.zeros(nl,nc)
    for i in range(nl):
        
        l=ima[i,:].copy()
        
        for j in range(2,nc):
            b1[j] = l[j-1]+2*ae*b1[j-1]-ae**2*b1[j-2]     # LIGNE A MODIFIER
        b1[0]=b1[2]
        b1[1]=b1[2]
        
        for j in range(nc-3,-1,-1):
            b2[j] = l[j+1]+2*ae*b2[j+1]-ae**2*b2[j+2]    # LIGNE A MODIFIER
        b2[nc-2]=b2[nc-3]
        b2[nc-1]=b2[nc-3]

        gradx[i,:]=c*ae*(b1-b2);
  
    return gradx
        
def dericheGradY(ima,alpha):

    
    nl,nc=ima.shape
    ae=math.exp(-alpha)
    c=-(1-ae)*(1-ae)/ae

    b1=np.zeros(nl)
    b2=np.zeros(nl)

    grady=np.zeros((nl,nc))

    for i in range(nc):
        
        l=ima[:,i].copy()
        
        for j in range(2,nl):
            b1[j]=l[j-1]+2*ae*b1[j-1]-ae**2*b1[j-2]  # LIGNE A MODIFIER
        b1[0]=b1[2]
        b1[1]=b1[2]      
        
        for j in range(nl-3,-1,-1):
            b2[j]=l[j+1]+2*ae*b2[j+1]-ae**2*b2[j+2]   # LIGNE A MODIFIER
        b2[nl-1]=b2[nl-3]
        b2[nl-2]=b2[nl-3]
        
        grady[:,i]=c*ae*(b1-b2);
        
    
    return grady


def dericheSmoothX(ima,alpha):

    
    nl,nc=ima.shape
    ae=math.exp(-alpha)
    c=(1-ae)*(1-ae)/(1+2*alpha*ae-ae*ae)


    b1=np.zeros(nc)
    b2=np.zeros(nc)

    smoothx=np.zeros((nl,nc))


#gradx=np.zeros(nl,nc)
    for i in range(nl):
        l=ima[i,:].copy()
        for j in range(2,nc):
            b1[j]=c*(l[j]+ae*(alpha-1)*l[j-1])+2*ae*b1[j-1]-ae*ae*b1[j-2]
        b1[0]=b1[2]
        b1[1]=b1[2]
        for j in range(nc-3,-1,-1):
            b2[j]=c*(ae*(alpha+1)*l[j+1]-ae*ae*l[j+2])+2*ae*b2[j+1]-ae*ae*b2[j+2]
        b2[nc-1]=b2[nc-3]
        b2[nc-2]=b2[nc-3]
        smoothx[i,:]=b1+b2;
        
    
    return smoothx

def dericheSmoothY(ima,alpha):

    
    nl,nc=ima.shape
    ae=math.exp(-alpha)
    c=(1-ae)*(1-ae)/(1+2*alpha*ae-ae*ae)

    b1=np.zeros(nl)
    b2=np.zeros(nl)

    smoothy=np.zeros((nl,nc))

    for i in range(nc):
        l=ima[:,i].copy()
        for j in range(2,nl):
            b1[j]=c*(l[j]+ae*(alpha-1)*l[j-1])+2*ae*b1[j-1]-ae*ae*b1[j-2]
        b1[0]=b1[2]
        b1[1]=b1[2]      
        for j in range(nl-3,-1,-1):
            b2[j]=c*(ae*(alpha+1)*l[j+1]-ae*ae*l[j+2])+2*ae*b2[j+1]-ae*ae*b2[j+2]
        b2[nl-1]=b2[nl-3]
        b2[nl-2]=b2[nl-3]
        
        smoothy[:,i]=b1+b2;
        
    
    return smoothy
##############################################"

############## le close('all') de Matlab
plt.close('all')
################################"


ima=io.imread('cell.tif')
alpha=0.5


gradx=dericheGradX(dericheSmoothY(ima,alpha),alpha)
grady=dericheGradY(dericheSmoothX(ima,alpha),alpha)  

gradx2=dericheGradX(dericheSmoothY(gradx,alpha),alpha)
grady2=dericheGradY(dericheSmoothX(grady,alpha),alpha)  

  

plt.figure('Image originale')
plt.imshow(ima, cmap='gray')


lpima=gradx2+grady2

plt.figure('Laplacien')
plt.imshow(lpima, cmap='gray')


posneg=(lpima>=0)

plt.figure('Laplacien binarisé -/+')
plt.imshow(255*posneg, cmap='gray')

nl,nc=ima.shape
contours=np.uint8(np.zeros((nl,nc)))


for i in range(1,nl):
    for j in range(1,nc):
        if (((i>0) and (posneg[i-1,j] != posneg[i,j])) or
            ((j>0) and (posneg[i,j-1] != posneg[i,j]))):
            contours[i,j]=255
            
   
plt.figure('Contours')
plt.imshow(contours, cmap='gray')

io.imsave('contours_cells.tif',contours)
            
#io.imsave('contours.tif',np.uint8(255*valcontours))

plt.show()
  
