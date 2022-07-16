import cvzone
from cvzone.ColorModule import ColorFinder
import cv2

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

myColorFinder = ColorFinder(True)
hsvVals = {'hmin': 0, 'smin': 209, 'vmin': 222, 'hmax': 29, 'smax': 255, 'vmax': 255}


while True:
    success, img = cap.read()
    imgColor, mask =myColorFinder.update(img, hsvVals)
    imgStack = cvzone.stackImages([img, imgColor], 2, 0.5)
    cv2.imshow("image", imgStack)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
