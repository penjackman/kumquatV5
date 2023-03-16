import cv2
import argparse
import apriltag
import numpy as np
import json
import aprilTagDetection
import solvePos

image = cv2.imread("./assets/apriltag_test3.png")
image = cv2.pyrDown(cv2.pyrDown(image))

rotation, translation = solvePos.solvePos(image)

# showing results
field = cv2.imread("./assets/gamefield_kumquat.png")

"""
IMAGE COORDINATES (REDO IF YOU USE A NEW IMAGE)
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

cv2.circle(field, pos, 10, (0, 0, 0), thickness=-1)
cv2.imshow("field", field)
cv2.waitKey(0)