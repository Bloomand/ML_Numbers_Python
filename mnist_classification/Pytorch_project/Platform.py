from tkinter import filedialog as fd
from tkinter import *
import torch
from torch.utils.data import DataLoader
from mnist_dataset import MNISTDataset
import tkinter
import numpy as np
import cv2
from tkinter import Tk, Button
from train import parse_args
from net import ConvNet
from classify import LinearSVM
from PIL import ImageGrab
from torchvision import transforms
from tkinter import messagebox
from PIL import Image, ImageDraw


class Paint(Frame):

    def __init__(self, parent, path_to_model, path_to_model2):
        Frame.__init__(self, parent)

        self.parent = parent
        self.color = "white"
        self.brush_size = 10
        self.setUI()
        self.model = ConvNet()
        self.model2 = LinearSVM(loading_file=path_to_model2)
        self.model.load_state_dict(torch.load(path_to_model))
        self.img = None

    def set_color(self, new_color):
        self.color = new_color

    def set_brush_size(self, new_size):
        self.brush_size = new_size

    def draw(self, event):

        self.canv.create_oval(event.x - self.brush_size,
                              event.y - self.brush_size,
                              event.x + self.brush_size,
                              event.y + self.brush_size,
                              fill='White', width=0)

    def load(self):
        file_name = fd.askopenfilename(filetypes=[("PNG.files", "*.png")])
        img = tkinter.PhotoImage(file=file_name)

        self.img = cv2.imread(file_name)

        root.geometry(str(img.width())+'x'+str(img.height()+100))
        self.canv.create_image(0, 0, image=img, anchor='nw')
        self.canv.config(width=img.width(), height=img.height())
        self.mainloop()
        pass

    def predict(self):
        x = self.parent.winfo_rootx() + self.canv.winfo_x()
        y = self.parent.winfo_rooty() + self.canv.winfo_y()
        x1 = x + self.canv.winfo_width()
        y1 = y + self.canv.winfo_height()
        if self.img is not None:
            img = self.img
            self.img = None
        else:
            img = np.array(ImageGrab.grab().crop((x, y, x1, y1)))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = Image.fromarray(img)
        img.save('1.png')
        trans = transforms.Compose(
            [
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        ds = MNISTDataset(train=False, transform=trans, img=img)
        test_loader = DataLoader(dataset=ds, batch_size=1, shuffle=False)
        y_pred = 0
        for (images, labels) in test_loader:
            outputs = self.model(images)
            _, predicted = torch.max(outputs.data, 1)
            y_pred = predicted.detach().numpy()
        messagebox.showinfo("Предсказание", "Это {}".format(y_pred[0]))
        return y_pred

    def predict2(self):
        x = self.parent.winfo_rootx() + self.canv.winfo_x()
        y = self.parent.winfo_rooty() + self.canv.winfo_y()
        x1 = x + self.canv.winfo_width()
        y1 = y + self.canv.winfo_height()
        img = cv2.resize(cv2.cvtColor(np.array(ImageGrab.grab([x, y, x1, y1])), cv2.COLOR_BGR2GRAY), (28, 28))
        img = img.ravel().reshape(1, -1)
        y_pred = self.model2.inference(img)
        messagebox.showinfo("Предсказание", "Это {}".format(y_pred))
        return y_pred


    def setUI(self):

        self.parent.title("Platform")
        self.pack(fill=BOTH, expand=1)

        self.columnconfigure(6, weight=1)
        self.rowconfigure(2, weight=1)

        self.canv = Canvas(self, bg='black')
        self.canv.grid(row=2, column=0, columnspan=8,
                       padx=5, pady=5, sticky=E+W+S+N)
        self.canv.bind("<B1-Motion>", self.draw)

        seven_btn = Button(self, text="Ten", width=10,
                           command=lambda: self.set_brush_size(10))
        seven_btn.grid(row=0, column=1)

        ten_btn = Button(self, text="Twenty", width=10,
                         command=lambda: self.set_brush_size(20))
        ten_btn.grid(row=0, column=2)

        clear_btn = Button(self, text="Clear all", width=10,
                           command=lambda: self.canv.delete("all"))
        clear_btn.grid(row=1, column=4)

        pred_btn = Button(self, text="Predict1", width=10,
                          command=lambda: self.predict())
        pred_btn.grid(row=1, column=1)

        pred2_btn = Button(self, text="Predict2", width=10,
                          command=lambda: self.predict2())
        pred2_btn.grid(row=1, column=2)

        load_btn = Button(self, text="Download", width=10,
                          command=lambda: self.load())
        load_btn.grid(row=1, column=3)


def main():
    global root
    args = parse_args()
    root = Tk()
    root.geometry("400x400")
    app = Paint(root, args.m, args.m2)
    root.mainloop()


if __name__ == '__main__':
    main()
