#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 22 May 2019

@author: M Roux
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import skimage as sk

from scipy import ndimage
from scipy import signal

from skimage import io

from skimage import filters


##############################################
## code de mrlab car echec lors de l'import ##

def interpolationbilineaire(ima,l,c):

    l,c
    
    l1=l-np.floor(l)
    l2=np.ceil(l)-l
    c1=c-np.floor(c)
    c2=np.ceil(c)-c   


    ll=np.uint32(np.floor(l))
    cc=np.uint32(np.floor(c))

    val=ima[ll,cc]*l2*c2+ima[ll+1,cc]*l1*c2+ima[ll,cc+1]*l2*c1+ima[ll+1,cc+1]*l1*c1

    return val 

def maximaDirectionGradient(gradx,grady):


    nl,nc=gradx.shape
 
    norme=np.sqrt(gradx*gradx+grady*grady)+0.1

    gradx=np.divide(gradx,norme)
    grady=np.divide(grady,norme)
 
    contours=np.zeros((nl,nc),dtype=int);

    for i in range(1,nl-1):
        for j in range(1,nc-1):
            G1=interpolationbilineaire(norme,i+grady[i,j],j+gradx[i,j]); 
            G2=interpolationbilineaire(norme,i-grady[i,j],j-gradx[i,j]); 
            if norme[i,j]>=G1 and norme[i,j]>=G2:
                contours[i,j]=1
            else:
                contours[i,j]=0

    return contours

##############################################"

############## le close('all') de Matlab
plt.close('all')
################################"


ima=io.imread('cell.tif')
sigma=1
seuilnorme=0.14


gfima=filters.gaussian(ima,sigma)

#plt.figure('Image originale')
#plt.imshow(ima, cmap='gray')

#plt.figure('Image filtrée (passe-bas)')
#plt.imshow(gfima, cmap='gray')

gradx=ndimage.sobel(gfima,0)
grady=ndimage.sobel(gfima,1)  
      
#plt.figure('Gradient horizontal')
#plt.imshow(gradx, cmap='gray')

#plt.figure('Gradient vertical')
#plt.imshow(grady, cmap='gray')

norme=np.sqrt(gradx*gradx+grady*grady)
io.imsave('contours_cells.tif',np.uint8(255*norme))
    
#plt.figure('Norme du gradient')
#plt.imshow(norme, cmap='gray')

#direction=np.arctan2(grady,gradx)
    
#plt.figure('Direction du Gradient')
#plt.imshow(direction, cmap='gray')


#contoursnorme =(norme>seuilnorme) 


#plt.figure('Norme seuillée')
#plt.imshow(255*contoursnorme)

contours=np.uint8(maximaDirectionGradient(gradx,grady))

#plt.figure('Maxima du gradient dans la direction du gradient')
#plt.imshow(255*contours)


valcontours=(norme>seuilnorme)*contours
      

plt.figure('gradient sobel 2D seuilé')
plt.imshow(255*valcontours)
plt.show()

