import cv2
import numpy as np
  
# Let's load a simple image with 3 black squares
image = cv2.imread('./apriltags_test2.jpeg')
# cv2.imshow('Color', image)
# cv2.waitKey(0)
print(image.shape)


# Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imshow('Gray Scale', gray)
# cv2.waitKey(0)
# Find Canny edges
#edged = cv2.Canny(gray, 30, 200)


#get contours
edged, thresh = cv2.threshold(gray, 65, 255, cv2.THRESH_BINARY)
# cv2.imshow('Binary Image', thresh)
# cv2.waitKey(0)
contours, hierarchy = cv2.findContours(image=thresh, 
    mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(image=image, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
cv2.imshow('Contours Simple', image)
cv2.waitKey(0)
  
print("Number of Contours found = " + str(len(contours)))

hulls = []
areas = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    if(area < 20):
        continue
    ep = 0.04 * cv2.arcLength(cnt, True)
    #hull = cv2.convexHull(cv2.approxPolyDP(cnt, ep, True))
    approx = cv2.approxPolyDP(cnt, ep, True)
    if(len(approx) == 4):
        hulls.append(approx)
    print("hull has length ", len(approx))


# Draw all contours
# -1 signifies drawing all hulls
# cv2.drawContours(image=image, contours=hulls, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
  
# cv2.imshow('Contours Simple Hull', image)
# cv2.waitKey(0)
cv2.destroyAllWindows()