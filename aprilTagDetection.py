import cv2
import numpy as np
  
# Let's load a simple image with 3 black squares
image = cv2.imread('./apriltag_test3.png')
print(image.shape)
#cv2.imshow("siuuuu", image)
#cv2.waitKey(0)

# Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  
# Find Canny edges
#edged = cv2.Canny(gray, 30, 200)


#get contours
edged, thresh = cv2.threshold(gray, 65, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh, 
    cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  
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
cv2.drawContours(image, hulls, -1, (0, 255, 0), 6)
  
cv2.imshow('Contours', image)
cv2.waitKey(0)
cv2.destroyAllWindows()