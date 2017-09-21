import numpy as np
import cv2
import glob
import pickle
import matplotlib.pyplot as plt

objp = np.zeros((6*9,3), np.float32)
# Fill all elements with values except for last column-z axis
# Followed by reshape to prevent error
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

objpoints =[] # Store 3d points
imgpoints =[] # Store 2d points

# Create a list of calibration images
images = glob.glob('D:/SDCND/Project_4/CarND-Advanced-Lane-Lines/camera_cal/calibration*.jpg')

for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
    
    if ret == True:
        objpoints.append(objp) # creating coordinate system
        imgpoints.append(corners)
        
    cv2.drawChessboardCorners(img, (9,6), corners, ret)
    write_name = 'corners' + str(idx) +'.jpg'
    cv2.imwrite(write_name,img)
    
img = cv2.imread('D:/SDCND/Project_4/CarND-Advanced-Lane-Lines/camera_cal/calibration1.jpg')
img_size = img.shape[1], img.shape[0]       
    
ret, mtx , dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None,None)

# Save new images with points
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump(dist_pickle, open('D:/SDCND/Project_4/CarND-Advanced-Lane-Lines/camera_cal/calibration_pickle.p','wb'))