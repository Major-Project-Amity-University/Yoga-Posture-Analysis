# 1 -> OPEN POSE ALGORITHM FOR POSE DETECTION
import cv2 as cv
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', help='Path to image or video. Skip to capture frames from the camera')
parser.add_argument('--thr', default=0.09, type=float, help='Threshold value for pose parts heat map')
parser.add_argument('--width', default=368, type=int, help='Resize input to a specific width.')
parser.add_argument('--height', default=368, type=int, help='Resize input to a specific height.')

args = parser.parse_args()

# Define body part keypoints and pose pairs
BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

inWidth = args.width
inHeight = args.height

# Load the pre-trained model
net = cv.dnn.readNetFromTensorflow("graph_opt.pb")

# Load the input image
image = cv.imread("MOUNTAIN_POSE.jpg")

if image is None:
    print("Error: Could not load the input image.")
    exit(1)

frameWidth = image.shape[1]
frameHeight = image.shape[0]

# Prepare the input image for the model
blob = cv.dnn.blobFromImage(image, scalefactor=1.0, size=(inWidth, inHeight), mean=(255, 255, 255), swapRB=True, crop=False)

# Set the input for the model
net.setInput(blob)

# Forward pass through the model
out = net.forward()
out = out[:, :18, :, :]

assert(len(BODY_PARTS) == out.shape[1])

points = []
for i in range(len(BODY_PARTS)):
    # Slice heatmap of corresponding body's part.
    heatMap = out[0, i, :, :]

    # Find the local maximum (peak) in the heatmap
    _, conf, _, point = cv.minMaxLoc(heatMap)

    # Calculate the (x, y) coordinates of the keypoint
    x = (frameWidth * point[0]) / out.shape[3]
    y = (frameHeight * point[1]) / out.shape[2]

    # Add the keypoint if its confidence is higher than the threshold
    points.append((int(x), int(y)) if conf > 0.09 else None)

def calculate_angle(keypoint1, keypoint2, keypoint3, points):
    if points[BODY_PARTS[keypoint1]] and points[BODY_PARTS[keypoint2]] and points[BODY_PARTS[keypoint3]]:
        kp1 = np.array(points[BODY_PARTS[keypoint1]])
        kp2 = np.array(points[BODY_PARTS[keypoint2]])
        kp3 = np.array(points[BODY_PARTS[keypoint3]])

        vector1 = kp1 - kp2
        vector2 = kp3 - kp2

        cosine_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        angle_rad = np.arccos(cosine_angle)
        angle_deg = np.degrees(angle_rad)

        return angle_deg % 360

text_y = 40
# Calculate and display angles for all keypoints
for pair in POSE_PAIRS:
    partFrom = pair[0]
    partTo = pair[1]
    assert(partFrom in BODY_PARTS)
    assert(partTo in BODY_PARTS)

    # For v pose
    angle_keypoints = [partTo, "Neck", partFrom]
    angle = calculate_angle(*angle_keypoints, points)

    if angle is not None:
        angle_text = f"Angle ({partFrom} to {partTo}): {angle:.2f} degrees"
        print(angle_text)
        cv.putText(image, angle_text, (10, text_y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        text_y += 20  
    else:
        angle_text = f"Angle ({partFrom} to {partTo}): Not detected"
        cv.putText(image, angle_text, (10, text_y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        text_y += 20 

# Draw keypoints on the image
for i, point in enumerate(points):
    if point:
        x, y = point
        cv.circle(image, (x, y), 4, (0, 0, 255), thickness=-1, lineType=cv.FILLED)
        cv.putText(image, str(i), (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# Draw lines between keypoints
for pair in POSE_PAIRS:
    partFrom = pair[0]
    partTo = pair[1]
    assert(partFrom in BODY_PARTS)
    assert(partTo in BODY_PARTS)

    idFrom = BODY_PARTS[partFrom]
    idTo = BODY_PARTS[partTo]

    if points[idFrom] and points[idTo]:
        cv.line(image, points[idFrom], points[idTo], (0, 255, 0), 3)
        cv.ellipse(image, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
        cv.ellipse(image, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

cv.imshow('OpenPose using OpenCV', image)
cv.waitKey(0)
cv.destroyAllWindows()