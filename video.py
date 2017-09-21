import numpy as np
import cv2
import glob
import pickle
from tracker import tracker
from moviepy.editor import VideoFileClip
from IPython.display import HTML


dist_pickle = pickle.load(open('D:/SDCND/Project_4/CarND-Advanced-Lane-Lines/camera_cal/calibration_pickle.p','rb'))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

images = glob.glob('D:/SDCND/Project_4/CarND-Advanced-Lane-Lines/test_images/test*.jpg')

# Function that takes an image, gradient orientation,
# and threshold min / max values.
def abs_sobel_thresh(img, orient='x', thresh=(0,255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # Return the result
    return binary_output

# Function to return the magnitude of the gradient
# for a given sobel kernel size and threshold values
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output

# Function to threshold an image for a given range and Sobel kernel
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output

# Function that thresholds the S-channel of HLS
def color_threshold(image, sthresh=(0,255), vthresh=(0,255)):
    hls=cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel > sthresh[0]) & (s_channel <= sthresh[1])] = 1
    
    hsv=cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    v_channel = hsv[:,:,2]
    v_binary = np.zeros_like(v_channel)
    v_binary[(v_channel > vthresh[0]) & (v_channel <= vthresh[1])] = 1
    
    output = np.zeros_like(s_channel)
    output[(s_binary==1) & (v_binary==1)] = 1
    
    return output

def window_mask(width,height,img_ref,center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-(level)*height),max(0,int(center-width)):min(int(center+width),img_ref.shape[1])]=1
    return output

def processImage(img):
      
    img=cv2.undistort(img,mtx,dist,None,mtx)
    #preprocessing image
    preprocessImage = np.zeros_like(img[:,:,0])
    gradx = abs_sobel_thresh(img, orient='x', thresh=(12,255))
    grady = abs_sobel_thresh(img, orient='y', thresh=(25,255))
    c_binary =color_threshold(img, sthresh=(100,255), vthresh =(50,255))
    preprocessImage[((gradx == 1) & (grady == 1) | (c_binary == 1))] = 255
    
    # perspective transformation area
    img_size=(img.shape[1],img.shape[0])
    #src = np.float32( [[585. /1280.*img_size[0], 455./720.*img_size[1]],
    #                    [705. /1280.*img_size[0], 455./720.*img_size[1]],
    #                    [1270./1280.*img_size[0], 720./720.*img_size[1]],
    #                    [190. /1280.*img_size[0], 720./720.*img_size[1]]])
    bot_width=0.76
    mid_width=0.08
    height_pct=0.62
    bottom_trim=0.935
    src=np.float32([[img.shape[1]*(0.5-mid_width/2),img.shape[0]*height_pct],
                     [img.shape[1]*(0.5+mid_width/2),img.shape[0]*height_pct],
                     [img.shape[1]*(0.5+bot_width/2),img.shape[0]*bottom_trim],
                     [img.shape[1]*(0.5-bot_width/2),img.shape[0]*bottom_trim]])
    offset=img_size[0]*0.25
    dst=np.float32([[offset,0],[img_size[0]-offset,0],[img_size[0]-offset,img_size[1]],[offset,img_size[1]]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    
    warped = cv2.warpPerspective(preprocessImage, M, img_size, flags=cv2.INTER_LINEAR)
    
    # tracker
    windowWidth = 25
    windowHeight = 80
    curveCenters = tracker(myWindowWidth = windowWidth, myWindowHeight=windowHeight, myMargin=25,myYM=10/720,myXM=4/384, mySmoothFactor=15)
    windowCentroids = curveCenters.findWindowCentroids(warped)
    lPoints = np.zeros_like(warped)
    rPoints = np.zeros_like(warped)
    rightx=[]
    leftx=[]
    for level in range(0,len(windowCentroids)):
        leftx.append(windowCentroids[level][0])
        rightx.append(windowCentroids[level][1])
        lMask = window_mask(windowWidth,windowHeight,warped,windowCentroids[level][0],level)
        rMask = window_mask(windowWidth,windowHeight,warped,windowCentroids[level][1],level)

        lPoints[(lPoints == 255)|((lMask ==1))]=255
        rPoints[(rPoints == 255)|((rMask ==1))]=255
        
    #fitting lane lines
    yvals=range(0,warped.shape[0])
    res_yvals=np.arange(warped.shape[0]-(windowHeight/2),0,-windowHeight)
    
    left_fit=np.polyfit(res_yvals,leftx,2)
    left_fitx=left_fit[0]*yvals*yvals+left_fit[1]*yvals+left_fit[2]
    left_fit=np.array(left_fitx,np.int32)
    
    right_fit=np.polyfit(res_yvals,rightx,2)
    right_fitx=right_fit[0]*yvals*yvals+right_fit[1]*yvals+right_fit[2]
    right_fit=np.array(right_fitx,np.int32)
    
    left_lane=np.array(list(zip(np.concatenate((left_fitx-windowWidth/2,left_fitx[::-1]+windowWidth/2),axis=0),np.concatenate((yvals,yvals[::-1]),axis=0))),np.int32)
    right_lane=np.array(list(zip(np.concatenate((right_fitx-windowWidth/2,right_fitx[::-1]+windowWidth/2),axis=0),np.concatenate((yvals,yvals[::-1]),axis=0))),np.int32)
    inner_lane=np.array(list(zip(np.concatenate((left_fitx-windowWidth/2,right_fitx[::-1]-windowWidth/2),axis=0),np.concatenate((yvals,yvals[::-1]),axis=0))),np.int32)
    
    road=np.zeros_like(img)
    cv2.fillPoly(road,[left_lane],color=[0,0,255])
    cv2.fillPoly(road,[right_lane],color=[0,0,255])
    cv2.fillPoly(road,[inner_lane],color=[0,255,0])
    
    # Output
    road_warped=cv2.warpPerspective(road,Minv,img_size,flags=cv2.INTER_LINEAR)
    result = cv2.addWeighted(img,1.0,road_warped,1.0,0.0)
    
    # Radius of curvature
    ym_per_pix=curveCenters.ymPerPix
    xm_per_pix=curveCenters.xmPerPix
    curve_fit_cr=np.polyfit(np.array(res_yvals,np.float32)*ym_per_pix,np.array(leftx,np.float32)*xm_per_pix,2)
    # left radius of curvature
    curvead=((1+(2*curve_fit_cr[0]*yvals[-1]*ym_per_pix+curve_fit_cr[1])**2)**1.5)/np.absolute(2*curve_fit_cr[0])
    
    # Offset of the car on road
    camera_center=(left_fitx[-1]+right_fitx[-1])/2
    center_diff=(camera_center-warped.shape[1]/2)*xm_per_pix
    side_pos='left'
    if center_diff<=0:
        side_pos='right'
    cv2.putText(result,'Radius of Curvature='+str(round(curvead,3))+'(m)',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    cv2.putText(result,'Vehicle is '+str(abs(round(center_diff,3)))+'m '+side_pos+' of center',(50,100),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    
    return result

outputVideo='D:/SDCND/Project_4/CarND-Advanced-Lane-Lines/output_video.mp4'
inputVideo='D:/SDCND/Project_4/CarND-Advanced-Lane-Lines/project_video.mp4'
clip1=VideoFileClip(inputVideo)
video_clip=clip1.fl_image(processImage)
video_clip.write_videofile(outputVideo,audio=False)
