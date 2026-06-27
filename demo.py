#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  demo.py
#  
#  Copyright 2021 PMT <PMT@PN04>
 

import fire
from multiprocessing import freeze_support

import apc.base as base
import apc.dcmIO as dIO
import apc.seg_wV as sg
import apc.imgProc as imgP


# data process
import cv2
import numpy as np
import torch
import torch.nn.functional as F

import os
#import pydicom
import SimpleITK as sitk
import csv


def loadP3D(pdir):
    
    reader = sitk.ImageSeriesReader()
    reader.MetaDataDictionaryArrayUpdateOn()
    dicom_names = reader.GetGDCMSeriesFileNames(pdir)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    img_array = sitk.GetArrayFromImage(image)  #frame_num, width, height = img_array.shape
    
    x0,y0,z0 = image.GetOrigin()
    dx,dy,dz = image.GetSpacing()
    print(img_array.dtype) #  int16,uint16,float32,
    #vol = img_array.astype(np.uint16) # (65535//img_array.max())
    
    vol = base.stdVol16(img_array) # 对每一层std
    vol_std = base.stdImg(img_array)  # 对整个std
    
    return vol,vol_std,(x0,y0,z0),(dx,dy,dz)
    
def complexFilter(vol_e):
    
    # aug
    vol_eAug = imgP.augMRI(vol_e)
    filterApc = imgP.getFilter()
    
    # filter
    apc3D_e = imgP.apcFilter(vol_eAug,filterApc)
    lap_uint8_e = imgP.lapFilter(vol_eAug)
    
    
    # crop extra_edge
    ext = 10
    apcV = apc3D_e[:,ext:-ext,ext:-ext]
    lap_V = lap_uint8_e[:,ext:-ext,ext:-ext]
    
    return vol_eAug,apcV,lap_V
    
def vol_crop(vol,mri3D):
    center = dIO.getCenter(vol)
    bbox,bbox_e = dIO.getBBox(center)
    mri3D_e = dIO.cutVol(mri3D,bbox_e)
    return mri3D_e,bbox

def outApcVis(root,contours,vol):
    newPdir = base.makeNewPath(root,'result','apcVis')
    for idx,contour in enumerate(contours):
        newMdir = base.makeNewPath(newPdir,str(idx))
        for j in range(contour.shape[0]):
            img = vol[j]
            mrk = contour[j]
            
            img_3ch = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
            
            img_post = imgP.redMark(img_3ch,mrk)
            cv2.imwrite(os.path.join(newMdir,'mrk_%d.png'%(j)),img_post)    
        
def getCls(contours):
    cls = []
    for idx,contour in enumerate(contours):
        centers = []
        for j in range(contour.shape[0]):
                                                        #mri2D = mri3D[j]
                                                        #img = vol[j]
            mrk = contour[j]
            if mrk.max()==0: # 如果
                continue

            out = np.where(mrk>0)
            if len(out[0])<=3: # 为了排除用于测试(debug)的点
                #print(idx,j)
                continue

            mean_x, mean_y = int(out[0].mean()), int( out[1].mean() )
            center = [j,mean_x,mean_y]

            centers.append(center)
        cls.append(centers)

    return cls
    
def saveRes(root,cls,bbox,origin,spacing,h):
        
    x_t,y_l = bbox[2],bbox[0]
    
    x0,y0,z0 = origin
    dx,dy,dz =  spacing
    
    
    outPath = base.makeNewPath(root,'result','phyCoor')
    for idx,c in enumerate(cls):
        for seed in c:

            lyIdx,x_tmp,y_tmp = seed
            x_tmp += x_t
            y_tmp += y_l

            #
            center_phy = [z0+dz*lyIdx, x0+x_tmp*dx, y0+y_tmp*(h-1-dy)]
            #
            with open(os.path.join(outPath,'coord_%d.csv'%(idx)),'a',newline='') as f:
                writer = csv.writer(f)
                writer.writerow(center_phy)
                



def apcLoc(srcdir,inpType='dicom',rg=None,cuda=0,
          repoRoot=None,gThr = 70,k=14,debug=False):
    '''
    args:
        -debug: if True, vis the contour results
    
    
    '''
    
    repoRoot = os.path.dirname(os.path.realpath(__file__))
    
    #if inpType = 'dicom':
    mri3D,mri3D_u8,origin,spacing = loadP3D(srcdir)
    
    # crop
    mri3D_e, bbox= vol_crop(mri3D_u8,mri3D)# dIO.cutVol(mri3D,bbox_e)
    
    #mri3D_eAug = imgP.augMRI(mri3D_e)# uint16
    mri3D_eAug,apc3D , lap_uint8 = complexFilter(mri3D_e)
    ext = 10
    
    
    # ====================================================
    k_values = imgP.topK(apc3D,k) #N,1 
    apc3D_binary = imgP.bin3D(apc3D,k_values)   #二值化  uint8
    filling3D_binary = imgP.drawHole(lap_uint8)  # uint8
    dihs =  imgP.overlap(apc3D_binary,filling3D_binary) # dots in holes uint8 
    
    #===============================================================
    vol = mri3D_eAug[ :,ext:-ext,ext:-ext]
    vol = base.stdVol(vol)
    
    
    contours = []
    growed = []
    
    rg_thr = 48 # region growing 最小值，低于这个值就不grow
    
    depth,h,w = mri3D.shape
    
    
    for idx in range(depth): # visit all the slice
        dih = dihs[idx]
        
        if dih.max() == 0: # 如果没有candidates，那就下一层
            continue
        seeds = sg.makeSeeds(dih,idx) # For every slice, return some candidate seeds
        for seed in seeds: # loop over seeds
            
            # if seed had been visited or below the threshold; skip
            if sg.is_inMarked(seed,growed) or vol[seed[0],seed[1],seed[2]]<=rg_thr:
                continue
            # grow
            mrked3D, markedSet = sg.regionGrow3D(vol,apc3D,seed,thr=rg_thr,grad_thr = gThr)
            
            if mrked3D is None:# if no return, next
                continue
            
            # else: save
            contours.append(mrked3D)# 
            growed.append(markedSet)
        
        
    #===============================================================
    if debug:
        outApcVis(repoRoot,contours,vol)
        
    cls = getCls(contours)

    #x0,y0,z0 = orgin  #dx,dy,dz =  spacing  #loadP3D(pdir)
    
    #===============================================================
    saveRes(repoRoot,cls,bbox,origin,spacing,h)
    
    return 0
    

if __name__ == '__main__':
    #import sys
    #sys.exit(main(sys.argv))
    freeze_support()#https://github.com/pyinstaller/pyinstaller/wiki/Recipe-Multiprocessing
    fire.Fire(apcLoc)
