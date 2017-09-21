import numpy as np

class tracker():
    
    def __init__(self, myWindowWidth, myWindowHeight, myMargin, myYM=1, myXM=1,mySmoothFactor=15):
        self.recentCenters = []
        self.windowWidth = myWindowWidth
        self.windowHeight = myWindowHeight
        self.margin = myMargin
        self.ymPerPix = myYM
        self.xmPerPix = myXM
        self.smoothFactor = mySmoothFactor
        
    def findWindowCentroids(self, warped):
        windowWidth = self.windowWidth
        windowHeight = self.windowHeight
        margin = self.margin
        windowCentroids = []
        window = np.ones(windowWidth) # Template to generate convolution
        
        lSum=np.sum(warped[int(3*warped.shape[0]/4):,:int(warped.shape[1]/2)],axis=0) # vertical pixels squashed to 1D array
        lCenter=np.argmax(np.convolve(window,lSum))-windowWidth/2
        rSum=np.sum(warped[int(3*warped.shape[0]/4):,int(warped.shape[1]/2):],axis=0)
        rCenter=np.argmax(np.convolve(window,rSum))-windowWidth/2 + int(warped.shape[1]/2)
        
        windowCentroids.append((lCenter,rCenter))
        
        # Look for maximum pixel locations in each layer
        for level in range(1,(int)(warped.shape[0]/windowHeight)):
            
            # Convolve window into a vertical slice of the iamge
            imageLayer = np.sum(warped[int(warped.shape[0]-(level+1)*windowHeight):int(warped.shape[0]-level*windowHeight),:], axis=0)
            convSignal=np.convolve(window,imageLayer)
            offset=windowWidth/2
            
            # best left centroid using previous left center as reference
            lMinIndex = int(max(lCenter+offset-margin,0))
            lMaxIndex = int(min(lCenter+offset+margin,warped.shape[1]))
            lCenter = np.argmax(convSignal[lMinIndex:lMaxIndex])+lMinIndex-offset
            
            # best right centroid using previous right center as reference
            rMinIndex = int(max(rCenter+offset-margin,0))
            rMaxIndex = int(min(rCenter+offset+margin,warped.shape[1]))
            rCenter = np.argmax(convSignal[rMinIndex:rMaxIndex])+rMinIndex-offset
            
            windowCentroids.append((lCenter,rCenter))
            
        self.recentCenters.append(windowCentroids)
        return np.average(self.recentCenters[-self.smoothFactor:],axis=0)