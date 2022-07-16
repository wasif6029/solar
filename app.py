from matplotlib.animation import PillowWriter
import re
from numpy.polynomial import Polynomial
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import sys
from doctest import master
import tkinter as tk
import tkinter.messagebox
import customtkinter
from tkinter import filedialog
import subprocess
from tkinter import *
from tkvideo import tkvideo

customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("green")  # Themes: "blue" (standard), "green", "dark-blue"

fileName = "E:/Python/yolov5/yolov5/videos/b_slowmo_path1.avi"
fileName2 = "E:/Python/yolov5/yolov5/videos/b_slowmo_path1.avi"
fileName3 = "E:/Python/yolov5/yolov5/runs/detect/exp302/right_centers.txt"


class App(customtkinter.CTk):

    WIDTH = 780
    HEIGHT = 520

    def browseFiles(self):
        global fileName
        fileName = filedialog.askopenfilename(initialdir="",
                                              title="Select a File",
                                              filetypes=(("Video files",
                                                          "*.mp4*"),
                                                         ("all files",
                                                         "*.*")))

    def browseFiles2(self):
        global fileName2
        fileName2 = filedialog.askopenfilename(initialdir="",
                                               title="Select a File",
                                               filetypes=(("Video files",
                                                          "*.mp4*"),
                                                          ("all files",
                                                          "*.*")))

    def browseFiles3(self):
        global fileName3
        fileName3 = filedialog.askopenfilename(initialdir="",
                                               title="Select a File",
                                               filetypes=(("Text files",
                                                           "*.txt*"),
                                                          ("all files",
                                                           "*.*")))

        # # Change label contents
        # label_file_explorer.configure(text="File Opened: " + fileName)

    def __init__(self):
        super().__init__()

        self.title("Track App")
        self.geometry(f"{App.WIDTH}x{App.HEIGHT}")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)  # call .on_closing() when app gets closed

        # ============ create two frames ============

        # configure grid layout (2x1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.frame_left = customtkinter.CTkFrame(master=self,
                                                 width=180,
                                                 corner_radius=0)
        self.frame_left.grid(row=0, column=0, sticky="nswe")

        self.frame_right = customtkinter.CTkFrame(master=self)
        self.frame_right.grid(row=0, column=1, sticky="nswe", padx=20, pady=20)

        # ============ frame_left ============

        # configure grid layout (1x11)
        self.frame_left.grid_rowconfigure(0, minsize=10)   # empty row with minsize as spacing
        self.frame_left.grid_rowconfigure(5, weight=1)  # empty row as spacing
        self.frame_left.grid_rowconfigure(8, minsize=20)    # empty row with minsize as spacing
        self.frame_left.grid_rowconfigure(11, minsize=10)  # empty row with minsize as spacing

        self.label_1 = customtkinter.CTkLabel(master=self.frame_left,
                                              text="Select Options To Run",
                                              text_font=("Roboto Medium", -16))  # font name and size in px
        self.label_1.grid(row=1, column=0, pady=10, padx=10)

        self.button_1 = customtkinter.CTkButton(master=self.frame_left,
                                                text="Run New YOLO",
                                                command=self.run_new_yolo_btn_clicked)
        self.button_1.grid(row=2, column=0, pady=10, padx=20)

        self.button_2 = customtkinter.CTkButton(master=self.frame_left,
                                                text="See Saved Works",
                                                command=self.see_saved_works_btn_clicked)
        self.button_2.grid(row=3, column=0, pady=10, padx=20)

        self.button_33 = customtkinter.CTkButton(master=self.frame_left,
                                                 text="Run Live",
                                                 command=self.run_new_yolo_live)
        self.button_33.grid(row=4, column=0, pady=10, padx=20)

        self.button_3322 = customtkinter.CTkButton(master=self.frame_left,
                                                   text="View 3d",
                                                   command=self.view_3d)
        self.button_3322.grid(row=4, column=0, pady=10, padx=20)

        self.button_3 = customtkinter.CTkButton(master=self.frame_left,
                                                text="Exit",
                                                command=self.on_closing)
        self.button_3.grid(row=5, column=0, pady=10, padx=20)

        self.label_mode = customtkinter.CTkLabel(master=self.frame_left, text="Appearance Mode:")
        self.label_mode.grid(row=9, column=0, pady=0, padx=20, sticky="w")

        self.optionmenu_1 = customtkinter.CTkOptionMenu(master=self.frame_left,
                                                        values=["Light", "Dark", "System"],
                                                        command=self.change_appearance_mode)
        self.optionmenu_1.grid(row=10, column=0, pady=10, padx=20, sticky="w")

        self.optionmenu_1.set("White")

    def button_event(self):
        print("Button pressed")

    def change_appearance_mode(self, new_appearance_mode):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def view_3d(self):
        self.destroyStuff()
        global fileName3
        fileName3 = fileName3.replace("/", "\\\\")
        self.button_select_video222 = customtkinter.CTkButton(master=self.frame_right,
                                                              text="Select Points",
                                                              command=self.browseFiles3)
        self.button_select_video222.grid(row=3, column=0, pady=10, padx=20)

        self.label_selected_video222 = customtkinter.CTkLabel(master=self.frame_right,
                                                              text=fileName3,
                                                              text_font=("Roboto Medium", -12))  # font name and size in px
        self.label_selected_video222.grid(row=4, column=0, pady=10, padx=10)

        self.button_422 = customtkinter.CTkButton(master=self.frame_right,
                                                  text="Plot in 3d",
                                                  command=plotIn3d)
        self.button_422.grid(row=5, column=0, pady=10, padx=20)

    def on_closing(self, event=0):
        self.destroy()

    def playVideo(self):
        global fileName2
        self.label_new_video1111 = customtkinter.CTkLabel(master=self.frame_right,
                                                          text="",
                                                          text_font=("Roboto Medium", -14))  # font name and size in px
        self.label_new_video1111.grid(row=1, column=0, pady=10, padx=10)
        player = tkvideo(fileName2, self.label_new_video1111, loop=0, size=(400, 400))
        player.play()

    def a(self):
        global fileName2
        fileName2 = fileName2.replace("/", "\\\\")
        self.button_select_video22 = customtkinter.CTkButton(master=self.frame_right,
                                                             text="Select Work",
                                                             command=self.browseFiles2)
        self.button_select_video22.grid(row=3, column=0, pady=10, padx=20)

        self.label_selected_video2 = customtkinter.CTkLabel(master=self.frame_right,
                                                            text=fileName2,
                                                            text_font=("Roboto Medium", -12))  # font name and size in px
        self.label_selected_video2.grid(row=4, column=0, pady=10, padx=10)

        self.button_42 = customtkinter.CTkButton(master=self.frame_right,
                                                 text="Play",
                                                 command=self.playVideo)
        self.button_42.grid(row=5, column=0, pady=10, padx=20)

    def run_new_yolo_btn_clicked(self):
        self.destroyStuff()
        self.label_new_yolo = customtkinter.CTkLabel(master=self.frame_right,
                                                     text="Run YOLOv5 with new project",
                                                     text_font=("Roboto Medium", -14))  # font name and size in px
        self.label_new_yolo.grid(row=1, column=0, pady=10, padx=10)

        self.button_select_video = customtkinter.CTkButton(master=self.frame_right,
                                                           text="Select Video",
                                                           command=self.browseFiles)
        self.button_select_video.grid(row=3, column=0, pady=10, padx=20)

        self.label_selected_video = customtkinter.CTkLabel(master=self.frame_right,
                                                           text=fileName,
                                                           text_font=("Roboto Medium", -12))  # font name and size in px
        self.label_selected_video.grid(row=4, column=0, pady=10, padx=10)

        self.button_4 = customtkinter.CTkButton(master=self.frame_right,
                                                text="Start",
                                                command=self.run_yolo)
        self.button_4.grid(row=5, column=0, pady=10, padx=20)

    def run_new_yolo_live(self):
        self.destroyStuff()
        command_for_live = "python detect_triangulation.py --source 0 --weights yolov5x.pt --classes 32"
        ret = subprocess.run(command_for_live, capture_output=True, shell=True)

    def run_yolo(self):
        self.destroyStuff()
        global fileName
        fileName = fileName.replace("/", "\\\\")
        command_for_filename = "python detect_triangulation.py --source " + fileName + " --weights yolov5x.pt --classes 32"
        self.label_yolo_running = customtkinter.CTkLabel(master=self.frame_right,
                                                         text="YOLO is running",
                                                         text_font=("Roboto Medium", -14))  # font name and size in px
        self.label_yolo_running.grid(row=6, column=0, pady=10, padx=10)

        ret = subprocess.run(command_for_filename, capture_output=True, shell=True)
        if(ret):
            self.label_yolo_running.destroy()

    def see_saved_works_btn_clicked(self):
        self.destroyStuff()
        self.a()

    def destroyStuff(self):
        try:
            self.label_new_yolo.destroy()
        except:
            pass
        try:
            self.button_select_video.destroy()
        except:
            pass
        try:
            self.label_selected_video.destroy()
        except:
            pass
        try:
            self.button_4.destroy()
        except:
            pass
        try:
            self.label_new_video1111.destroy()
        except:
            pass
        try:
            self.button_select_video22.destroy()
        except:
            pass
        try:
            self.label_selected_video2.destroy()
        except:
            pass
        try:
            self.button_42.destroy()
        except:
            pass
        try:
            self.label_new_video1111.destroy()
        except:
            pass
        try:
            self.button_select_video22.destroy()
        except:
            pass
        try:
            self.button_select_video222.destroy()
        except:
            pass
        try:
            self.label_selected_video2.destroy()
        except:
            pass
        try:
            self.label_selected_video222.destroy()
        except:
            pass
        try:
            self.button_42.destroy()
        except:
            pass
        try:
            self.button_422.destroy()
        except:
            pass

# import stuff2
# import stuff1


def plotIn3d():

    global fileName3
    filePath = fileName3

    x = []
    y = []
    z = []

    file1 = open(filePath, 'r')
    count = 0
    while True:
        count += 1

        # Get next line from file
        line = file1.readline()
        # print(line)
        if not line:
            break
        cord = re.findall(r"[-+]?\d*\.\d+|\d+", line)

        x.append(float(cord[0]))
        y.append(float(cord[1]))
        z.append(float(cord[2]))

    numbers = len(x)
    np.random.seed(123)
    n = numbers
    # t = np.random.choice(np.linspace(-1000000, 10000000, 10000002), n)

    t = []
    for i in range(numbers):
        t.append(i)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # ax.scatter(x, y, z, label="raw data")
    # for i in range(1):
    #     ax.scatter(x[0], y[0], z[0])
    # ax.scatter(x[n-1], y[n-1], z[n-1])

    def func(t, x2, x1, x0, y2, y1, y0, z2, z1, z0):
        Px = Polynomial([x2, x1, x0])
        Py = Polynomial([y2, y1, y0])
        Pz = Polynomial([z2, z1, z0])

        return np.concatenate([Px(t), Py(t), Pz(t)])

    start_vals = [x[0], x[1], x[2],
                  y[0], y[1], y[2],
                  z[0], z[1], z[2]]

    xyz = np.concatenate([x, y, z])
    popt, _ = curve_fit(func, t, xyz, p0=start_vals)

    t_fit = np.linspace(min(t), max(t) + (abs(max(t) - min(t)) // 10))
    xyz_fit = func(t_fit, *popt).reshape(3, -1)
    ax.plot(xyz_fit[0, :], xyz_fit[1, :], xyz_fit[2, :], color="green", label="fitted data")

    metadata = dict(title="Movie")
    writer = PillowWriter(fps=15, metadata=metadata)

    ax.legend()
    ax.view_init(-90, 90)
    plt.show()


if __name__ == "__main__":
    app = App()
    app.mainloop()
