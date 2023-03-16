import cv2
import numpy as np
import aprilTagDetection
import json

def solvePos(image, focal=None, center=None, posJsonPath=None):
    if(focal == None):
        focal = image.shape[1]
    if(center == None):
        center = (image.shape[1]/2, image.shape[0]/2)
    if(posJsonPath == None):
        posJsonPath = "./config/tagPositions.json"

    # detect tags
    pointsProj = aprilTagDetection.detectTags(image)
    # get coordinates of april tags in real life
    idPositions = json.load(open(posJsonPath))
    
    # getting projected coordinates and model coordinates so solvePnP can read it
    imgPoints = []
    modelPoints = []

    for point in pointsProj:
        ptId = point[0]
        ptPos = point[1]
        imgPoints.append(ptPos)
        modelPoints.append(idPositions[ptId]) 

    imgPoints = np.array(imgPoints, dtype=np.float32)
    modelPoints = np.array(modelPoints, dtype=np.float32)

    camMat = np.array([
        [focal, 0, center[0]],
        [0, focal, center[1]],
        [0, 0, 1]
    ], dtype=np.float32)

    distCoefs = np.zeros((4, 1))

    success, rotation, translation, _ = cv2.solvePnPGeneric(modelPoints, imgPoints, camMat, distCoefs, flags=cv2.SOLVEPNP_EPNP)
    translation = translation[0].reshape((3,))
    rotation = rotation[0].reshape((3,))

    return rotation, translation
