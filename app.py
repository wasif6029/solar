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
                                                 text="Run Live YOLO",
                                                 command=self.run_new_yolo_live)
        self.button_33.grid(row=4, column=0, pady=10, padx=20)

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

    def on_closing(self, event=0):
        self.destroy()

    def a(self):
        self.label_new_video = customtkinter.CTkLabel(master=self.frame_right,
                                                      text="",
                                                      text_font=("Roboto Medium", -14))  # font name and size in px
        self.label_new_video.grid(row=1, column=0, pady=10, padx=10)
        player = tkvideo(fileName, self.label_new_video, loop=0, size=(600, 600))
        player.play()

    def run_new_yolo_btn_clicked(self):
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
        command_for_live = "python detect_triangulation.py --source 0 --weights yolov5x.pt --classes 32"
        ret = subprocess.run(command_for_live, capture_output=True, shell=True)

    def run_yolo(self):
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
        try:
            self.label_new_yolo.destroy()
        except:
            pass
        self.button_select_video.destroy()
        self.label_selected_video.destroy()
        self.button_4.destroy()
        self.a()


if __name__ == "__main__":
    app = App()
    app.mainloop()
