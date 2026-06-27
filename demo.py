#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  demo.py
#  
#  Copyright 2021. Ian
#  


#from multiprocessing import freeze_support




import fire
import os 

# data process
import torch
import torch.nn.functional as F
import cv2
import numpy as np


import apc.base as base
import apc.dcmIO as dIO
import apc.seg_ct as sgCT
import apc.seg_mr as sgMR
import apc.imgProc as imgP
import utils.masktocontours as masktocontours




#import pydicom
import SimpleITK as sitk
import csv



def loadP3D(pdir):
    
    
    reader = sitk.ImageSeriesReader()
    reader.MetaDataDictionaryArrayUpdateOn()
    dicom_names = reader.GetGDCMSeriesFileNames(pdir)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    img_array = sitk.GetArrayFromImage(image)
    
    mod = reader.GetMetaData(0,'0008|0060').lower()
    
    x0,y0,z0 = image.GetOrigin()
    dx,dy,dz = image.GetSpacing()
    
    if 'c' in mod:
        #TODO
        print('preparing the CT dcm data ...')
        img_array[img_array<-1000] = -1000
        min_bound = img_array.min()
        img_array += abs(min_bound)
        img_array[img_array<0] = 0 # 这里的min，max都是全局的不是对每个slice单独搞
        mx = img_array.max()
        vol_std = img_array.astype(np.float32)/mx*255
        #print(vol_std.max(),vol_std.min())
        return vol_std.astype(np.uint8),(x0,y0,z0),(dx,dy,dz),'ct'
        
    elif 'm' in mod:
        print('preparing the MRI dcm data ...') 
        #vol = base.stdVol16(img_array) # 这个有必要吗?? 
        vol_std = base.stdVol(img_array)
        
        return vol_std,(x0,y0,z0),(dx,dy,dz),'mr'

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
        open(os.path.join(outPath, 'coord_%d.csv' % (idx)), 'w', newline='')
        for seed in c:
            lyIdx,x_tmp,y_tmp = seed
            x_tmp += x_t
            y_tmp += y_l
            #
            center_phy = [x0+x_tmp*dx, y0+dy*(h-1-y_tmp), z0+dz*lyIdx]
            #
            with open(os.path.join(outPath,'coord_%d.csv'%(idx)),'a',newline='') as f:
                writer = csv.writer(f)
                writer.writerow(center_phy)

def getAppContours(cls,bbox):
    x_t, y_l = bbox[2], bbox[0]
    appcontour = []
    for idx, c in enumerate(cls):
        cls_pixel = []
        for seed in c:
            lyIdx, x_tmp, y_tmp = seed
            x_tmp += x_t
            y_tmp += y_l
            #
            center_pixel = [lyIdx, y_tmp, x_tmp]
            #
            cls_pixel.append(np.array(center_pixel))
        appcontour.append(cls_pixel)
    return appcontour
    
def vol_crop(vol,radius):
    center = dIO.getCenter(vol)
    bbox,bbox_e = dIO.getBBox(center,radius)
    mri3D_e = dIO.cutVol(vol,bbox_e)
    return mri3D_e,bbox
    
    
def seedsInCT(mri3D_e,ext):
    ext = 10
    filterApc = imgP.getFilter()
    apc3D_e = imgP.apcFilter(mri3D_e,filterApc)
    apc3De_bin = (apc3D_e>235).astype(np.uint8)
    
    #==========3D aga======================
    seedVol = apc3De_bin[ :,ext:-ext,ext:-ext]
    vol = mri3D_e[:, ext:-ext,ext:-ext]
    
    return seedVol,vol


def apcLoc(srcdir,inpType='dicom',rg=None,cuda=0,  #gradThr = 50,gray_min = 70,k=8,
          repoRoot=None,
          debug=False):
    '''
    args:
        -debug: if True, vis the contour results
    apcLoc --srcdir path2yourdata
    
    '''
    
    repoRoot = os.path.dirname(os.path.realpath(__file__))
    
    #if inpType = 'dicom':
    vol_u8,origin,spacing,mod = loadP3D(srcdir)
    ext = 10
    
    
    
    #==============find the seeds=================
    if mod=='ct':
        mri3D_e,bbox = vol_crop(vol_u8,60)
        gradThr = 50
        gray_min  = 70   #gray_min = 70 # 最小灰度值   thr_grad = 50
        seedVol,vol = seedsInCT(mri3D_e,ext) #apc3De_bin[ :,ext:-ext,ext:-ext]  vol = mri3D_e[:, ext:-ext,ext:-ext]
    
    #==========3D aga======================
    contours = []
    growed = []         
    
    depth,h,w = vol.shape
    for idx in range(depth): 
        seedPlane = seedVol[idx]

        if seedPlane.max() == 0: # 如果没有candidates，那就下一层
            continue
        seeds = sg.makeSeeds(seedPlane,idx) #
        for seed in seeds:

            if sg.is_inMarked(seed,growed) or vol[seed[0],seed[1],seed[2]]<=gray_min:
                continue
            mrked3D, markedSet = sg.regionGrow3D(vol,seed,thr=gray_min,grad_thr = gradThr)
            if mrked3D is None:
                continue
            contours.append(mrked3D)
            growed.append(markedSet)
    
    
    
    #===============================================================
    if debug:
        
        outApcVis(repoRoot,contours,vol)
        
    cls = getCls(contours)

    #x0,y0,z0 = orgin  #dx,dy,dz =  spacing  #loadP3D(pdir)
    
    #===============================================================
    saveRes(repoRoot,cls,bbox,origin,spacing,h)
    #=======from wen============================
    appcontour = getAppContours(cls,bbox)
    masktocontours.ApptoContours(appcontour,srcdir , srcdir)

    return 0
    

if __name__ == '__main__':
    #import sys
    #sys.exit(main(sys.argv))
    #freeze_support()#https://github.com/pyinstaller/pyinstaller/wiki/Recipe-Multiprocessing
    fire.Fire(apcLoc)
