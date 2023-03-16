import cv2
import numpy as np
import apriltag

def detectTags(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    options = apriltag.DetectorOptions(families="tag16h5") #USE 16H5 WHEN TESTING ON FIELD 
    detector = apriltag.Detector(options)
    
    results = detector.detect(gray)
    
    pointsProj = []
    for r in results:
        # extract the bounding box (x, y)-coordinates for the AprilTag
        # and convert each of the (x, y)-coordinate pairs to integers
        (ptA, ptB, ptC, ptD) = r.corners
        ptB = (int(ptB[0]), int(ptB[1]))
        ptC = (int(ptC[0]), int(ptC[1]))
        ptD = (int(ptD[0]), int(ptD[1]))
        ptA = (int(ptA[0]), int(ptA[1]))
        
        (cX, cY) = (int(r.center[0]), int(r.center[1]))
        pointsProj.append((str(r.tag_id), (cX, cY)))

    return pointsProj