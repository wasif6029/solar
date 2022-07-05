from tkinter import *
from tkvideo import tkvideo
fileName = "E:/Python/yolov5/yolov5/videos/b_slowmo_path1.avi"
root = Tk()


def a():
    my_label = Label(root)
    my_label.pack()
    player = tkvideo(fileName, my_label, loop=0, size=(1280, 720))
    player.play()


a()
root.mainloop()
