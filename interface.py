import os
import sys

# interface
import tkinter as tk
from tkinter import Tk
from tkinter.filedialog import askopenfilename, askdirectory

# inference
import cv2
import time
import torch
import numpy as np
from torchvision.transforms import ToTensor

# models
from models.frvsr import RDN
from models.fnet import FNet

# utils
from utils.utils import space_to_depth, backward_warp, create_mp4, save_img
from utils.flags import flags as opt
from utils.upsample import BicubicUpsample
from utils.utils import print_log


class TextRedirector(object):
    def __init__(self, widget, tag="stdout"):
        self.widget = widget
        self.tag = tag

    def write(self, str):
        self.widget.configure(state="normal")
        self.widget.insert("end", str, (self.tag,))
        self.widget.configure(state="disabled")


class VSRApp():
    def __init__(self):
        super(VSRApp, self).__init__()
        self.video_path = ""
        self.folder_path = ""
        self.checkpoint_path = ""

        self.root = Tk()
        self.output_messages = None

        self.file_button = None
        self.folder_button = None
        self.checkpoint_button = None
        self.start_button = None

    def file_dialog(self):
        filename = askopenfilename(filetypes=[(".mp4","*.mp4"),(".avi", "*.avi")])
        self.video_path = filename
        self.validate()

    def folder_dialog(self):
        foldername = askdirectory()
        self.folder_path = foldername
        self.validate()

    def chackpoint_dialog(self):
        checkpoint_file = askopenfilename(filetypes=[(".ckpt","*.ckpt")])
        self.checkpoint_path = checkpoint_file
        self.validate()

    def time_to_pack(self):
        self.output_messages.pack()
        self.file_button.pack()
        self.checkpoint_button.pack()
        self.folder_button.pack()
        self.start_button.pack()

    def print_stdout(self, message, level):
        print_log(message=message, level=level)
        self.output_messages.update()
        self.output_messages.see("end")

    def validate(self):
        if self.video_path:
            self.checkpoint_button.config(state='normal')
        else:
            self.checkpoint_button.config(state='disable')
            self.folder_button.config(state='disable')
            self.start_button.config(state='disable')

        if self.video_path != "" and self.checkpoint_path != "":
            self.folder_button.config(state='normal')
        else:
            self.folder_button.config(state='disable')
            self.start_button.config(state='disable')

        if self.video_path != "" and self.checkpoint_path != "" and self.folder_path != "":
            self.start_button.config(state='normal')
        else:
            self.start_button.config(state='disable')

    def inference_video(self, checkpoint_path, input_path, output_path):

        # get images paths
        self.print_stdout("Getting the images paths!", "LOG")
        images_names = os.listdir(input_path)
        images_names.sort()

        # get device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.print_stdout(f"Device available: {device}", "LOG")

        # prepare models + optimizators
        net = RDN()
        net_flow = FNet(3)

        # up
        bicubic = BicubicUpsample(scale_factor=4)

        transform = ToTensor()

        with torch.no_grad():
            self.print_stdout("Loading the checkpoint...", "LOG")
            net.load_state_dict(torch.load(checkpoint_path)["modelG_state_dict"])
            net_flow.load_state_dict(torch.load(checkpoint_path)["modelF_state_dict"])
            self.print_stdout("Done!", "LOG")


            net.eval().to(device)
            net_flow.eval().to(device)

            # get first output
            lr_first = cv2.cvtColor(cv2.imread(f"{opt.inference_path}/{images_names[0]}"), cv2.COLOR_BGR2RGB)
            lr_first = transform(lr_first)
            lr_first = torch.unsqueeze(lr_first, axis=0).to(device)

            n, c, lr_h, lr_w = lr_first.shape

            hr_pre = torch.zeros((n, c, lr_h * 4, lr_w * 4), dtype=torch.float32).to(device)
            lr_prev = torch.zeros((n, c, lr_h, lr_w), dtype=torch.float32).to(device)

            number_of_images = len(images_names)

            self.print_stdout("Starting the inference...", "LOG")

            for t in range(0, number_of_images):
                self.print_stdout(f"Frames done: {t + 1} / {number_of_images}", "INFERENCE")
                lr_curr = cv2.cvtColor(cv2.imread(f"{opt.inference_path}/{images_names[t]}"), cv2.COLOR_BGR2RGB)
                lr_curr = transform(lr_curr)
                lr_curr = torch.unsqueeze(lr_curr, axis=0).to(device)

                # get flow
                lr_flow = net_flow(lr_curr, lr_prev)

                # up flow
                hr_flow = bicubic(lr_flow)

                # backward
                hr_warped = backward_warp(hr_pre, hr_flow)

                # space-to-depth
                input_net = space_to_depth(hr_warped)

                # net
                net_output = net(lr_curr, input_net)
                lr_prev, hr_pre = lr_curr, net_output

                # prepare output
                if torch.cuda.is_available():
                    out = net_output[0].cpu().detach().numpy()
                else:
                    out = net_output[0].numpy()
                out = out.transpose((1, 2, 0))
                out = np.uint8(np.clip(np.round(out * 255.0), 0, 255))

                if t < 10:
                    dir_number = f"000{t}"
                elif 10 <= t <= 99:
                    dir_number = f"00{t}"
                elif 100 <= t <= 999:
                    dir_number = f"0{t}"
                else:
                    dir_number = f"{t}"
                # save output
                save_img(out, f"inference/output_video/{dir_number}.png")


        self.print_stdout("Inference done!", "LOG")
        self.print_stdout("Creating mp4 file...", "LOG")
        curr_dir_path = os.path.dirname(os.path.abspath(__file__))

        create_mp4(f"{curr_dir_path}/inference/output_video", output_path)

        self.print_stdout("Done!", "LOG")


    def get_frames(self):
        curr_dir_path = os.path.dirname(os.path.abspath(__file__))
        os.system(f"ffmpeg -i {self.video_path} -vf scale=640x360 {curr_dir_path}\inference\input\%04d.png")

    def start_inference(self):
        self.output_messages.delete(0.0, tk.END)
        self.print_stdout("Starting the inference!", "LOG")
        self.print_stdout("Getting the low res frames, please wait...", "LOG")
        self.get_frames()

        curr_dir_path = os.path.dirname(os.path.abspath(__file__))
        self.print_stdout("Getting the high res frames, please wait...!", "LOG")
        self.inference_video(checkpoint_path=self.checkpoint_path, input_path=f"{curr_dir_path}/inference/input",
                             output_path=self.folder_path)

    def start_app(self):
        self.root.geometry("700x480")
        self.root.title("VSR Application")
        self.root.resizable(False, False)
        self.root.configure(background='#404040')

        self.output_messages = tk.Text(master=self.root, height=15, width=80, bg='#33363a', fg='white')
        self.output_messages.tag_configure("stderr", foreground="#b22222")
        self.output_messages.grid(columnspan=3, row=0, padx=30, pady=25)
        self.output_messages.insert(0.0,
        """
        
        Welcome to VSR App!
        
        To inference a video, please follow the next steps:
        
        1. Choose the low resolution video file. (.mp4 / .avi formats)
        2. Choose the checkpoint file to load.
        3. Choose a directory to save the output video.
        4. Press "Start" button to start the inference.
        
        ______________________________________________________________
        """)

        sys.stdout = TextRedirector(self.output_messages, "stdout")

        self.file_button = tk.Button(self.root, height=3, width=20, background='#5B5B5B', foreground='#F0F0F0',
                                     activebackground='#686868',
                                     text="Select file", relief=tk.GROOVE,
                                     command=lambda: self.file_dialog())
        self.file_button.grid(column=0, row=1, pady=10)

        self.checkpoint_button = tk.Button(self.root, height=3, width=20, background='#5B5B5B', foreground='#F0F0F0',
                                           text="Select checkpoint", relief=tk.GROOVE,
                                           command=lambda: self.chackpoint_dialog())
        self.checkpoint_button.grid(column=1, row=1)

        self.folder_button = tk.Button(self.root, height=3, width=20, background='#5B5B5B', foreground='#F0F0F0',
                                       text="Select output folder", relief=tk.GROOVE,
                                       command=lambda: self.folder_dialog())
        self.folder_button.grid(column=2, row=1)

        self.start_button = tk.Button(self.root, height=2, width=25, background='#5B5B5B', foreground='#F0F0F0',
                                      text="Start", relief=tk.GROOVE,
                                      command=lambda: self.start_inference())
        self.start_button.grid(column=1, row=2, pady=40, stick=tk.N)

        self.validate()

        self.root.mainloop()


if __name__ == '__main__':
    app = VSRApp()
    app.start_app()
