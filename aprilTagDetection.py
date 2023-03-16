import cv2
import argparse
import apriltag
import numpy as np
import json

"""
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image containing AprilTag")
args = vars(ap.parse_args())
"""

print("[INFO] loading image...")
image = cv2.imread("apriltag_test3.png")
image = cv2.pyrDown(cv2.pyrDown(image))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

print("[INFO] detecting AprilTags...")
options = apriltag.DetectorOptions(families="tag16h5") #USE 16H5 WHEN TESTING ON FIELD 
detector = apriltag.Detector(options)
results = detector.detect(gray)
print("[INFO] {} total AprilTags detected".format(len(results)))

pointsProj = []
#pointsModel = []
#idPositions = {1: (0, 0, 0), 2: (0, 0, 0), 3: (0, 0, 0), 4: (0, 0, 0), 5: (0, 0, 0), 6: (0, 0, 0), 7: (0, 0, 0), 8: (0, 0, 0)}
posJson = open("./tagPositions.json")
idPositions = json.load(posJson)

# loop over the AprilTag detection results
for r in results:
	# extract the bounding box (x, y)-coordinates for the AprilTag
	# and convert each of the (x, y)-coordinate pairs to integers
    (ptA, ptB, ptC, ptD) = r.corners
    ptB = (int(ptB[0]), int(ptB[1]))
    ptC = (int(ptC[0]), int(ptC[1]))
    ptD = (int(ptD[0]), int(ptD[1]))
    ptA = (int(ptA[0]), int(ptA[1]))
    # draw the bounding box of the AprilTag detection
    cv2.line(image, ptA, ptB, (0, 255, 0), 2)
    cv2.line(image, ptB, ptC, (0, 255, 0), 2)
    cv2.line(image, ptC, ptD, (0, 255, 0), 2)
    cv2.line(image, ptD, ptA, (0, 255, 0), 2)
    # draw the center (x, y)-coordinates of the AprilTag
    (cX, cY) = (int(r.center[0]), int(r.center[1]))
    cv2.circle(image, (cX, cY), 5, (0, 0, 255), -1)
    # draw the tag family on the image
    pointsProj.append((str(r.tag_id), (cX, cY)))
    tagFamily = r.tag_family.decode("utf-8")
    cv2.putText(image, tagFamily, (ptA[0], ptA[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    print("[INFO] tag family: {}".format(tagFamily))

focal_length = image.shape[1]
center = (image.shape[1]/2, image.shape[0]/2)

imgPoints = []
modelPoints = []

for point in pointsProj:
    ptId = point[0]
    ptPos = point[1]
    imgPoints.append(ptPos)
    modelPoints.append(idPositions[ptId]) 

imgPoints = np.array(imgPoints, dtype=np.float32)
modelPoints = np.array(modelPoints, dtype=np.float32)

print(imgPoints, modelPoints)

camMat = np.array([
    [focal_length, 0, center[0]],
    [0, focal_length, center[1]],
    [0, 0, 1]
], dtype=np.float32)

distCoefs = np.zeros((4, 1))

success, rotation, translation, _ = cv2.solvePnPGeneric(modelPoints, imgPoints, camMat, distCoefs, flags=cv2.SOLVEPNP_EPNP)
translation = translation[0].reshape((3,))
print(translation);

#cv2.imshow("Image", image)

field = cv2.imread("./gamefield_kumquat.png")
def click_event(event, x, y, flags, params):    
    if event==cv2.EVENT_LBUTTONDOWN:
        print(x, ' ', y)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = field[y, x, 0]
        g = field[y, x, 1]
        r = field[y, x, 2]
        cv2.putText(field, str(b) + ',' +
                    str(g) + ',' + str(r),
                    (x,y), font, 1,
                    (255, 255, 0), 2)

"""
IMAGE COORDINATES
top left: 293, 180
top right: 1475, 180
bottom left (origin): 293, 750
bottom right: 1475, 750
REAL FIELD COORDINATES
top left: 0, 319
top right: 649, 319
bottom left (origin): 0, 0
bottom right: 649, 0
"""

scale = (1475-293)/649 #using above comment

orig = np.array([297, 750])
pos = orig + np.array([translation[0], translation[2]]) * scale
pos = pos.astype(np.int32)
print(pos)
cv2.circle(field, orig, 10, (0, 255, 0), thickness=-1)
cv2.circle(field, pos, 10, (0, 0, 255), thickness=-1)
cv2.imshow("field", field)
cv2.setMouseCallback("field", click_event)
cv2.waitKey(0)