import cv2
import numpy as np
  
# Let's load a simple image with 3 black squares
image = cv2.imread('./apriltag_test1.png')
  
# Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  
# Find Canny edges
edged = cv2.Canny(gray, 30, 200)

#get contours
contours, hierarchy = cv2.findContours(edged, 
    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  
print("Number of Contours found = " + str(len(contours)))

hulls = []
for cnt in contours:
    hull = cv2.convexHull(cnt)
    hulls.append(hull)
# Draw all contours
# -1 signifies drawing all hulls
cv2.drawContours(image, hulls, -1, (0, 255, 0), 3)
  
cv2.imshow('Contours', image)
cv2.waitKey(0)
cv2.destroyAllWindows()