import cv2
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier


class SignLanguageDetector:
    """Handles hand detection and sign language classification"""

    def __init__(self):
        self.detector = HandDetector(maxHands=2) #to detect 2 hands instead of 1
        self.classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
        self.offset = 20
        self.img_size = 300
        self.labels = [
            "One",
            "Two",
            "Three",
            "Four",
            "Five",
            "Six",
            "Seven",
            "Eight",
            "Father",
            "Mother",
            "Stepfather",
            "Stepmother",
            "Cents",
            "Know",
            "Love",
            "Wrong",
            "Beef",
            "Who",
            "Yes",
            "You",
            "Where"

]


    def detect_hand_sign(self, img):
        """Detect hand and classify the sign"""
        hands, img = self.detector.findHands(img)

        if not hands:
            return None, img, None, None, None


        hand = hands[0]
        x, y, w, h = hand['bbox']

        img_white = np.ones((self.img_size, self.img_size, 3), np.uint8) * 255
        img_crop = img[y - self.offset:y + h + self.offset, x - self.offset:x + w + self.offset]
        aspect_ratio = h / w

        if aspect_ratio > 1:
            k = self.img_size / h
            w_cal = math.ceil(k * w)
            img_resize = cv2.resize(img_crop, (w_cal, self.img_size))
            w_gap = math.ceil((self.img_size - w_cal) / 2)
            img_white[:, w_gap:w_cal + w_gap] = img_resize
        else:
            k = self.img_size / w
            h_cal = math.ceil(k * h)
            img_resize = cv2.resize(img_crop, (self.img_size, h_cal))
            h_gap = math.ceil((self.img_size - h_cal) / 2)
            img_white[h_gap:h_cal + h_gap, :] = img_resize

        prediction, index = self.classifier.getPrediction(img_white, draw=False)
        detected_sign = self.labels[index]

        return detected_sign, img, img_crop, img_white, (x, y, w, h)

    def draw_hand_bbox(self, img, bbox):
        """Draw bounding box and label on image"""
        x, y, w, h = bbox
        detected_sign, _, _, _, _ = self.detect_hand_sign(img)

        cv2.rectangle(img, (x - self.offset, y - self.offset - 50),
                     (x - self.offset + 90, y - self.offset - 50 + 50),
                     (255, 0, 255), cv2.FILLED)
        cv2.putText(img, detected_sign, (x, y - 26),
                   cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(img, (x - self.offset, y - self.offset),
                     (x + w + self.offset, y + h + self.offset),
                     (255, 0, 255), 4)

        return img