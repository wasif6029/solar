from datetime import datetime
import triangulation as tri
import calibration
import cvzone
from cvzone.ColorModule import ColorFinder
import cv2


# Stereo vision setup parameters
frame_rate = 30  # Camera frame rate (maximum at 120 fps)87..o
B = 20  # Distance between the cameras [cm]
f = 28  # Camera lense's focal length [mm]
alpha = 75  # Camera field of view in the horisontal plane [degrees]
#cap_right = cv2.VideoCapture('left.MP4', cv2.CAP_DSHOW)
#cap_left =  cv2.VideoCapture('right.MP4', cv2.CAP_DSHOW)

cap_right = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap_left = cv2.VideoCapture(0, cv2.CAP_DSHOW)


myColorFinder = ColorFinder(False)
hsvVals_left = {'hmin': 7, 'smin': 72, 'vmin': 120, 'hmax': 19, 'smax': 160, 'vmax': 208}
hsvVals_right = {'hmin': 7, 'smin': 72, 'vmin': 120, 'hmax': 19, 'smax': 160, 'vmax': 208}

#
# hsvVals_left = {'hmin': 0, 'smin': 209, 'vmin': 222, 'hmax': 29, 'smax': 255, 'vmax': 255}
# hsvVals_right = {'hmin': 0, 'smin': 209, 'vmin': 222, 'hmax': 29, 'smax': 255, 'vmax': 255}

x = []
y = []
z = []

while True:
    success_left, img_left = cap_left.read()
    success_right, img_right = cap_right.read()
    img_right, img_left = calibration.undistortRectify(img_right, img_left)
    imgColor_left, mask_left = myColorFinder.update(img_left, hsvVals_left)
    imgColor_right, mask_right = myColorFinder.update(img_right, hsvVals_right)
    imgContour_left, contours_left = cvzone.findContours(
        img_left, mask_left, minArea=50, sort=True, filter=0)  # USE MINAREA AS FILTER
    imgContour_right, contours_right = cvzone.findContours(img_right, mask_right, minArea=50, sort=True, filter=0)
    if contours_left and contours_right:
        data_left = contours_left[0]['center'][0], contours_left[0]['center'][1]
        data_right = contours_right[0]['center'][0], contours_left[0]['center'][1]
        # print(data_left, data_right)
        X = (contours_left[0]['center'][0] + contours_right[0]['center'][0]) // 2
        Y = (contours_left[0]['center'][1] + contours_right[0]['center'][1]) // 2
        depth = tri.find_depth(data_right, data_left, img_right, img_left, B, f, alpha)
        # print(depth)
        Z = depth

        x.append(X)
        y.append(Y)
        z.append(Z)

    cv2.imshow("imageLeft", imgContour_left)
    cv2.imshow("imageRight", imgContour_right)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


name = "..//yolov5//Stereo_Points//stereo-points.txt"
f = open(name, 'w')
for i in range(len(x)):
    f.writelines(str(x[i]) + ' ' + str(y[i]) + ' ' + str(z[i]) + '\n')
