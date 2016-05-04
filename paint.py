import cv2
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
    
    if debug:
        cv2.waitKey(0)
    pass
    
if __name__ == "__main__":
    import sys
    if len(sys.argv)>0:
        img = cv2.imread(sys.argv[1])
        find(img,1)
