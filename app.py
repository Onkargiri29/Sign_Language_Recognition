import cv2
import tkinter as tk
from PIL import Image, ImageTk
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

class HandGestureApp:
    def __init__(self, root, root_title):
        self.root = root
        self.root.title(root_title)
        
        self.cap = cv2.VideoCapture(0)
        self.detector = HandDetector(maxHands=1)
        self.classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
        
        self.offset = 20
        self.img_size = 300
        self.labels = ["Hello","ThankYou","BestLuck","A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
        
        self.video_label = tk.Label(root)
        self.video_label.pack()

        self.start_button = tk.Button(root, text="Start Recognition", command=self.start_camera)
        self.start_button.pack()

        self.stop_button = tk.Button(root, text="Stop Recognition", command=self.stop_camera)
        self.stop_button.pack()
        self.stop_button["state"] = "disabled"

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        self.update()

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.start_button["state"] = "disabled"
        self.stop_button["state"] = "active"

    def stop_camera(self):
        self.cap.release()
        self.start_button["state"] = "active"
        self.stop_button["state"] = "disabled"

    def on_close(self):
        self.cap.release()
        self.root.destroy()

    def update(self):
        ret, frame = self.cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hands, frame = self.detector.findHands(frame)
        
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            img_white = np.ones((self.img_size, self.img_size, 3), np.uint8) * 255
            img_crop = frame[y - self.offset:y + h + self.offset, x - self.offset:x + w + self.offset]

            aspect_ratio = h / w

            if aspect_ratio > 1:
                k = self.img_size / h
                w_cal = math.ceil(k * w)
                img_resize = cv2.resize(img_crop, (w_cal, self.img_size))
                img_white[:, :w_cal] = img_resize
                prediction, index = self.classifier.getPrediction(img_white, draw=False)
            else:
                k = self.img_size / w
                h_cal = math.ceil(k * h)
                img_resize = cv2.resize(img_crop, (self.img_size, h_cal))
                img_white[:h_cal, :] = img_resize
                prediction, index = self.classifier.getPrediction(img_white, draw=False)

            cv2.rectangle(frame, (x - self.offset, y - self.offset - 50),
                          (x - self.offset + 90, y - self.offset - 50 + 50), (255, 0, 255), cv2.FILLED)
            cv2.putText(frame, self.labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
            cv2.rectangle(frame, (x - self.offset, y - self.offset),
                          (x + w + self.offset, y + h + self.offset), (255, 0, 255), 4)

        image = Image.fromarray(frame)
        image = ImageTk.PhotoImage(image=image)
        self.video_label.image = image
        self.video_label.config(image=image)

        self.root.after(10, self.update)


if __name__ == "__main__":
    root = tk.Tk()
    app = HandGestureApp(root, "Hand Gesture Recognition")
    root.mainloop()
