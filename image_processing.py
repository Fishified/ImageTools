# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 11:10:32 2017

@author: Thinkpad
"""

import numpy as np
import cv2
import glob
import os
import pandas as pd

from pandas import DataFrame
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#CAPTURE IMAGES FOR CALIBRATION FROM RASPBERRY PIS
vidcap=0
vidcap=cv2.VideoCapture('C:\\Users\\dugj2403\\Desktop\\calibration_tests\\plate14light.h264')
#success, image=vidcap.read()
count=0
while(True):
    ret, frame = vidcap.read()
    cv2.imshow("Video",frame)
    count += 1
    
    if count == 30:
        cv2.imwrite("C:\\Users\\dugj2403\\Desktop\\calibration_tests\\plate14light.jpg", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#newidxVert=[24,25,26,27,28,29,30,31,16,17,18,19,20,21,22,23,8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7]
#SharedimgPointsLongLat=reindexImgPoints(SharedimgPointsLongLat, newidxVert)
background="C:\\Users\\dugj2403\\Desktop\\calibration_tests\\frame30.jpg"
background = cv2.imread(background)
backgroundgray = cv2.cvtColor(background,cv2.COLOR_BGR2GRAY)
#height, length, channels = img.shape

foreground="C:\\Users\\dugj2403\\Desktop\\calibration_tests\\plate14light.jpg"
foreground = cv2.imread(foreground)
foregroundgray = cv2.cvtColor(foreground,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(foregroundgray,100,255,cv2.THRESH_BINARY_INV)
kernel=np.ones((5,5),np.uint8)
erosion=cv2.erode(thresh,kernel,iterations=1)

#img=backgroundgray-foregroundgray

#gray = cv2.cvtColor(thresh,cv2.COLOR_BGR2GRAY)
(cnts, _) = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:400]

i=0
for c in cnts:
    print cv2.contourArea(c)
    if 100 <= cv2.contourArea(c) <= 1500:
#        print i
        cv2.drawContours(foreground, [c], -5, (0, 255, 0), 1)
        i=i+1
        
cv2.imshow("Calibration points", foreground)
#cv2.imshow("Calibration points", erosion)
cv2.waitKey(0)



#cam1_images, cam2_images= sampleMultiPageTIF('C:\\Users\\Thinkpad\\Desktop\\Stereo_test\\*.tif', '14376','14375',20)
#
#objPoints_cam1, imgPoints_cam1, imgPaths_cam1, rawImgs_cam1 = loopgetChessboardCorners(cam1_images,8, 4,100,True,True,False)
#objPoints_cam2, imgPoints_cam2, imgPaths_cam2, rawImgs_cam2 = loopgetChessboardCorners(cam2_images,8, 4,1000,True,True,True)
#
#commonCam1, commonCam2=findCommonImages(imgPaths_cam1, imgPaths_cam2, '14376', '14375', 'C:\\Users\\Thinkpad\\Desktop\\Stereo_test')
#
#objPoints_cam1, imgPoints_cam1, imgPaths_cam1, rawImgs_cam1 = loopgetChessboardCorners(commonCam1,8, 4,100,True,True,False)
#objPoints_cam2, imgPoints_cam2, imgPaths_cam2, rawImgs_cam2 = loopgetChessboardCorners(commonCam2,8, 4,1000,True,True,False)
#
#img = cv2.imread(imgPaths_cam1[0])
#gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#size=gray.shape[::-1]
#cv2.imshow('img',gray)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#
#ret_cam1, mtx_cam1, dist_cam1, rvecs_cam1, tvecs_cam1, newcameramtx_cam1 = solveCamera(objPoints_cam1,imgPoints_cam1,imgPaths_cam1[1])
#ret_cam2, mtx_cam2, dist_cam2, rvecs_cam2, tvecs_cam2, newcameramtx_cam2 = solveCamera(objPoints_cam2,imgPoints_cam2,imgPaths_cam2[1])
#
#stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
#
#donno, newcameramtx_cam1, dist_cam1, newcameramtx_cam2, dist_cam2, R,T,E,F = cv2.stereoCalibrate(objPoints_cam1, imgPoints_cam1, imgPoints_cam2, gray.shape[::-1],  newcameramtx_cam1, dist_cam1, newcameramtx_cam2, dist_cam2, criteria = stereocalib_criteria, flags = (cv2.CALIB_FIX_INTRINSIC + cv2.CALIB_RATIONAL_MODEL))
#
#img1 = cv2.imread(imgPaths_cam1[0])
#img2 = cv2.imread(imgPaths_cam2[0])
#
##img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
##img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
#
#
#R1, R2, P1, P2, Q,dunno1,dunno2=cv2.stereoRectify(newcameramtx_cam1,dist_cam1,newcameramtx_cam2,dist_cam2,gray.shape[::-1],R,T,alpha=0)
#
#
#mat,rot,trans,x,y,z,dunno3=cv2.decomposeProjectionMatrix(P1)
#mapx1, mapx1 = cv2.initUndistortRectifyMap(newcameramtx_cam1, dist_cam1, R, newcameramtx_cam1, size, cv2.CV_32FC1)
#mapx2, mapx2 = cv2.initUndistortRectifyMap(newcameramtx_cam2, dist_cam2, R, newcameramtx_cam2, size, cv2.CV_32FC1)
#
#img_rect1=cv2.remap(img1,mapx1,mapx1,cv2.INTER_LINEAR)
#img_rect2=cv2.remap(img2,mapx2,mapx2,cv2.INTER_LINEAR)
#
#total_size =(max(img_rect1.shape[0],img_rect2.shape[0]),
#             img_rect1.shape[1]+img_rect2.shape[1], 3)
#img=np.zeros(total_size, dtype=np.uint8)
#img[:img_rect1.shape[0],:img_rect1.shape[1]]=img_rect1
#img[:img_rect2.shape[0],img_rect1.shape[1]:]=img_rect2
#
#
#for i in range(20, img.shape[0], 25):
#    cv2.line(img, (0, i), (img.shape[1],i),(255,0,0))
#cv2.imshow('imgRectified',img_rect1)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#
#cc=cv2.undistortPoints(imgPoints_cam1[1], newcameramtx_cam1, dist_cam1)
#dd=cv2.undistortPoints(imgPoints_cam2[1], newcameramtx_cam2, dist_cam2)
#
#vert=np.squeeze(cc)
#vert=pd.DataFrame(vert)
#vert=vert.transpose()
#vert=vert.as_matrix()
#
#lat=np.squeeze(dd)
#lat=pd.DataFrame(lat)
#lat=lat.transpose()
#lat=lat.as_matrix()
#
#points=cv2.triangulatePoints(P1, P2, cc, dd) 
#points_converted=cv2.convertPointsFromHomogeneous(points)
#a=pd.DataFrame(points)
#aa=a.transpose()
#aa.columns=['x','y','z','dunno']
#
##aa.x=aa.x/aa.dunno
##aa.y=aa.x/aa.dunno
##aa.z=aa.x/aa.dunno
#
#threedee = plt.figure().gca(projection='3d')
##plt.axes().set_aspect('equal')
##plt.axis([0, 1, 0, 1, 0, 1])
#threedee.scatter(aa.x, aa.y, aa.z)
#threedee.set_xlabel('x')
#threedee.set_ylabel('y')
#threedee.set_zlabel('dunno')
#plt.show()


"""
Returns list of image paths from a specified sample of tif files obtained from video in PCC software.
-use with syncronized dual camera stereo system

Usage:
    path - path to multipage tif files
    cameraID_1 - characteristic identifier for camera1 in each file name
    cameraID_2 - characteristic identifier for camera2 in each file name
    step - number of images to skip over from inclusion in output lists
"""
def sampleMultiPageTIF(path,cameraID_1, cameraID_2,step):
    
    images=glob.glob(path)  

    cam1= filter(lambda x: cameraID_1 in x, images)
    cam2= filter(lambda x: cameraID_2 in x, images)
    
    sampled_cam1=cam1[0::step]
    sampled_cam2=cam2[0::step]
    
    return sampled_cam1, sampled_cam2


"""
Finds the set of common positively identified (corners) images between two sterocameras
-sampled_cam1 list of sampled images for camera 1 returned from sampleMultiPageTIF
-sampled_cam2 list of sampled images for camera 2 returned from sampleMultiPageTIF
-path, path to directory containing tif images
"""
def findCommonImages(sampled_cam1,sampled_cam2,cameraID_1, cameraID_2,path):
    
    rhs_cam1=[]
    rhs_cam2=[]
    
    for i in sampled_cam1:
        head, tail = os.path.split(i)
        lhs, rhs = tail.split("_", 1)
        rhs_cam1.append(rhs)
        
    for i in sampled_cam2:
        head, tail = os.path.split(i)
        lhs, rhs = tail.split("_", 1)
        rhs_cam2.append(rhs)
        
    rhs_cam1=pd.DataFrame(rhs_cam1)
    rhs_cam2=pd.DataFrame(rhs_cam2)
    common_rhs=pd.merge(rhs_cam1, rhs_cam2)
    common_rhs.columns=['rhs']
    
    common_rhs=common_rhs['rhs'].tolist()
    path=path+'\\'+'*.tif'
    images=glob.glob(path)
    
    common_images=[]
    for i in common_rhs:
        imgs=(filter(lambda x: i in x, images))
        for i in imgs:
            common_images.append(i)
            
    cam1_common= filter(lambda x: cameraID_1 in x, common_images)
    cam2_common= filter(lambda x: cameraID_2 in x, common_images)

            
    return cam1_common, cam2_common

    

"""
Because of the mirrored image of the calibration plate for the vertical view,
the sequence of detected chessboard corners does not follow the same order as 
the lateral and longitudinal view. This causes a problem for the stereoCalibrate
function that requires only one set of objPoints and the corresponding image 
locations of these points for each camera in the stereo setup. This code allows
the detected corners in the LongLat view to be reassigned to match the index 
of the corresponding point in the vertical view. 
"""
def reindexImgPoints(points,new_idx):
    imgPoints=[]
    for i in range(len(points)):
        a=points[0]
        b=np.squeeze(a)
        df=pd.DataFrame(b)
        df=df.reindex(new_idx)
        df=df.reset_index(drop=True)
        df=df.as_matrix()
        df=np.expand_dims(df,1)
        imgPoints.append(df)
    return imgPoints
    
"""
Code below draws orientation axis on first detected point
"""
#axis = np.float32([[1,0,0], [0,1,0], [0,0,-1]]).reshape(-1,3)
# # Find the rotation and translation vectors.
#rvecs, tvecs, inliers = cv2.solvePnPRansac(objPointsVert[2], imgPointsVert[2], mtx_Ver, dist_Ver)
## project 3D points to image plane
#imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx_Ver, dist_Ver)
#img=cv2.imread(imgPathsVert[2])
#corner = tuple(imgPointsVert[2][0].ravel())
#cv2.line(img, corner, tuple(imgpts[0].ravel()), (0,255,0), 1)
#cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 1)
#cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 1)
#cv2.imshow('img',img)
#cv2.waitKey(0) 

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

def solveCamera(objPoints,imgPoints,image):
    
    img = cv2.imread(image)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    print (gray.shape[::-1])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objPoints, imgPoints, gray.shape[::-1], flags = cv2.CALIB_RATIONAL_MODEL)
    h,  w = gray.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
#    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
#    x,y,w,h = roi
#    dst = dst[y:y+h, x:x+w]
#    cv2.imwrite('calibresult_Lat.png',dst)
    return ret, mtx, dist, rvecs, tvecs, newcameramtx

def splitImages(images,xSplitCoord, ySplitCoord):
    for image in images: 
        verticalSplit(image, xSplitCoord,  ySplitCoord)
        print "Image split successfully"
    LatLongImageNames=os.path.dirname(images[0])+'\\'+'LatLongCal'+'\\'+'*.jpg*'
    VerticalImageNames=os.path.dirname(images[0])+'\\'+'VertCal'+'\\'+'*.jpg*'
    LatLongImages = glob.glob(LatLongImageNames)
    VerticalImages = glob.glob(VerticalImageNames)    
    return LatLongImages, VerticalImages

def loopgetChessboardCorners(images,xIntersections, yIntersections,delay,saveImage,subPix,show):
    
    for i in range(len(images)):
        if i == 0:
            success=0
            objpoints = [] # 3d point in real world space
            imgpoints = [] # 2d points in image plane.
            imgPaths= []
            rawImgs=[]
            objp = np.zeros((xIntersections*yIntersections,3), np.float32)
            objp[:,:2] = np.mgrid[0:xIntersections,0:yIntersections].T.reshape(-1,2)
    
        corners, ret = getChessboardCorners(images[i],xIntersections,yIntersections,delay,saveImage,subPix,show)
        
        if ret:
            success += 1
            print "Pattern found on image %s" % os.path.basename(images[i])
            b=os.path.basename(images[i]).split('_')
            rawImgs.append(b[1])
            imgpoints.append(corners)
            imgPaths.append(images[i])
            objpoints.append(objp)
        else:
            print "Failure: no pattern found on image %s" % os.path.basename(images[i]) 
        if i == len(images)-1:
            print "The %d images in the list returned %d positives" % (len(images), len(imgPaths))
            
    return objpoints, imgpoints, imgPaths, rawImgs
            
"""
Split image vertically
-useful to split mirrored and direct views captured in a single frames
-saves split images into LatLong and CalVert folders
"""
def verticalSplit(imgPath, xSplitCoord, ySplitCoord):
#    savePath,folder=os.path.split(os.path.dirname(imgPath))
    savePath=os.path.dirname(imgPath)
    imgName=os.path.basename(imgPath)
    
    img = cv2.imread(imgPath)
    height, length, channels = img.shape
    imgVert= img[0:ySplitCoord, xSplitCoord:length]  
    imgLatLong= img[ySplitCoord:height, xSplitCoord:length]

    try:
        os.mkdir(savePath+'\\'+'LatLongCal')
    except WindowsError:
        pass
    try:
        os.mkdir(savePath+'\\'+'VertCal')
    except WindowsError:
        pass
    
    saveVert=savePath+'\\'+'VertCal'+'\\'+'Vert_'+imgName
    saveLatLong=savePath+'\\'+'LatLongCal'+'\\'+'LatLong_'+imgName
    cv2.imwrite(saveVert, imgVert)
    cv2.imwrite(saveLatLong, imgLatLong)
    
#    return imgLatLong.shape[::-1], imgVert.shape[::-1]
    
def getChessboardCorners(imgPath,yIntersections, xIntersections,delay,saveImage,subPix,show):
    savePath=os.path.dirname(imgPath)
    imgName=os.path.basename(imgPath)
    print imgName
    print imgPath
    img = cv2.imread(imgPath)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (yIntersections,xIntersections),None)
    
    if ret == True:
        if subPix:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            cv2.cornerSubPix(gray,corners,(yIntersections,xIntersections),(-1,-1),criteria)
        cv2.drawChessboardCorners(img, (yIntersections,xIntersections), corners, ret)
        
        if show:
            cv2.imshow('img',img)
            cv2.waitKey(delay)
            cv2.destroyAllWindows()
        
        if saveImage:
            try:
                os.mkdir(savePath+'//'+'WithCorners')
            except WindowsError:
                pass
            saveImg=savePath+'\\'+'WithCorners'+'\\'+'Corners_'+imgName
            cv2.imwrite(saveImg, img)
            
        
    return corners, ret
   

