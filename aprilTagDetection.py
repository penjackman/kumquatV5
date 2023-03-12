# IMPORTANT:
# To run this code, use:
# "python3.10 aprilTagDetection.py --image example.png"


import apriltag # run "pip install CMake", and then "pip install apriltag"
import argparse
import cv2

# construct argument parser & parse arguments (no way)
# this isn't required, but just makes life easier
ap = argparse.ArgumentParser();
ap.add_argument("-i", "--image", required=True, help="path to input image containing AprilTag")
args = vars(ap.parse_args());

# Loading image
image = cv2.imread(args["image"])

# Grayscale
grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

options = apriltag.DetectorOptions(families="tag36h11") # changable depending on what type of tag
detector = apriltag.Detector(options)
results = detector.detect(grey);
print("Detected {} total AprilTags".format(len(results)))

for i in results:
    # make points for corners
    (ptA, ptB, ptC, ptD) = i.corners;
    ptA = (int(ptA[0]), int(ptA[1]))
    ptB = (int(ptB[0]), int(ptB[1]))
    ptC = (int(ptC[0]), int(ptC[1]))
    ptD = (int(ptD[0]), int(ptD[1]))
    
    # draw lines using corners
    cv2.line(image, ptA, ptB, (0, 255, 0), 2)
    cv2.line(image, ptB, ptC, (0, 255, 0), 2)
    cv2.line(image, ptC, ptD, (0, 255, 0), 2);
    cv2.line(image, ptD, ptA, (0, 255, 0), 2)

    # make centre of apriltag with a red dot
    (cX, cY) = (int(i.center[0]), int(i.center[1]))
    cv2.circle(image, (cX, cY), 5, (0, 0, 255), -1);

    tagFamily = i.tag_family.decode("utf-8")
    cv2.putText(image, tagFamily, (ptA[0], ptA[1] - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    print("Tag family: {}".format(tagFamily))



cv2.imshow('Contours', image)
cv2.waitKey(0);