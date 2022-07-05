from tkinter import *

import tkinter as tk
import subprocess


from tkinter import filedialog

# Function for opening the
# file explorer window

# fileName = "E:\\Python\\yolov5\\yolov5\\videos\\b_slowmo_path1.avi"
fileName = "E:/Python/yolov5/yolov5/videos/b_slowmo_path1.avi"


def browseFiles():
    global fileName
    fileName = filedialog.askopenfilename(initialdir="/",
                                          title="Select a File",
                                          filetypes=(("Text files",
                                                      "*.mp4*"),
                                                     ("all files",
                                                      "*.*")))

    # Change label contents
    label_file_explorer.configure(text="File Opened: " + fileName)


def doorbell(event):

    global fileName

    fileName = fileName.replace("/", "\\\\")

    print(" You rang the Doorbell !")
    command = "python detect_shadreza.py --source 0 --weights yolov5x.pt --classes 32"
    command = "python detect_shadreza.py --source " + fileName + " --weights yolov5x.pt --classes 32"

    ret = subprocess.run(command, capture_output=True, shell=True)

    # command = "python graphing2d.py"
    # ret = subprocess.run(command, capture_output=True, shell=True)

    command = "python graphing3d.py"
    ret = subprocess.run(command, capture_output=True, shell=True)


window = tk.Tk()
window.title(" Start App")
window.geometry("600x600")
mybutton = tk.Button(window, text="yolov5")
mybutton.grid(column=1, row=0)
mybutton.bind("<Button-1>", doorbell)

label_file_explorer = Label(window,
                            text="File Explorer using Tkinter",
                            width=100, height=4,
                            fg="blue")


button_explore = Button(window,
                        text="Browse Files",
                        command=browseFiles)

# button_exit = Button(window,
#                      text="Exit",
#                      command=exit)

# Grid method is chosen for placing
# the widgets at respective positions
# in a table like structure by
# specifying rows and columns
label_file_explorer.grid(column=1, row=1)

button_explore.grid(column=1, row=2)

# button_exit.grid(column=1, row=3)


window.mainloop()
