import cvzone
from cvzone.ColorModule import ColorFinder
import cv2

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

success, img = cap.read()

myColorFinder = ColorFinder(False)
hsvVals = {'hmin': 0, 'smin': 209, 'vmin': 222, 'hmax': 29, 'smax': 255, 'vmax': 255}


while True:
    success, img = cap.read()
    imgColor, mask =myColorFinder.update(img, hsvVals)
    imgContour, contours =cvzone.findContours(img, mask,minArea=300,sort=True, filter=0) #USE MINAREA AS FILTER
    if contours:
        data = contours[0]['center'][0], contours[0]['center'][1], int(contours[0]['area'])
        print(data)
    imgStack = cvzone.stackImages([img, imgColor, mask, imgContour], 2, 0.5)
    cv2.imshow("image", imgStack)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
