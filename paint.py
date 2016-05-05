# reference
# http://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=findcontours
# http://www.pyimagesearch.com/2014/04/21/building-pokedex-python-finding-game-boy-screen-step-4-6/
# http://www.pyimagesearch.com/2014/05/05/building-pokedex-python-opencv-perspective-warping-step-5-6/
# https://en.wikipedia.org/wiki/Bilateral_filter
# /opencv/samples/python2/contours.py
# https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm
import cv2
import numpy as np
def find(img,debug=False):
    while (max(img.shape[:2])>1000):
        img = cv2.pyrDown(img)

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    if debug:
        cv2.imshow("gray",gray)
    h,w = gray.shape[:2]
    
    # blur image to remove noise using Bilateral filtering
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    if debug:
        cv2.imshow("blur",gray)
    
    # get the edges using canny
    edge =  cv2.Canny(gray, 30, 200)
    if debug:
        cv2. imshow("edge",edge)
    
    # find contours, edge data will be changed 
    (_, cnts, _) = cv2.findContours(edge.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # find the top 10 contours
    cnts = sorted(cnts, key = cv2. contourArea, reverse = True)[:10]
    
    # loop over contours
    for c in cnts:
        # perimeter
        peri = cv2.arcLength(c, True)
        # Ramer-Douglas-Peucker algorithm
        # https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        # if the approximated contour has four points, assume it is a rect
        if len(approx) == 4:
            screenCnt = approx
            break
                
    # draw contours
    if debug:
        cv2.drawContours(img, [screenCnt] , -1, (255,0,0), 2)
        cv2.imshow("cnts",img)

 
    # determine the top-left top-right bottom-right bottom-left, clockwise order
    pts = screenCnt.reshape(4,2)
    rect = np.zeros((4,2), dtype = "float32")
    # the top-left has min x+y, the bottom-right has max x+y
    xpy = np.sum(pts,axis = 1)
    # top-left
    rect[0] = pts[np.argmin(xpy)]
    # bottom-right
    rect[2] = pts[np.argmax(xpy)]
    # the top-right has min y-x, the bottom-right has max y-x
    ysx = np.diff(pts, axis = 1)
    # top-right
    rect[1] = pts[np.argmin(ysx)]
    # bottom-left
    rect[3] = pts[np.argmax(ysx)]

    if debug:
        print pts
        print ysx
        print rect
        cv2.waitKey(0)
    pass
    
if __name__ == "__main__":
    import sys
    if len(sys.argv)>0:
        img = cv2.imread(sys.argv[1])
        find(img,1)
