#!/usr/bin/env python

import numpy as np
import cv2
import sys
from glob import glob
import itertools as it

face1 = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
video_capture = cv2.VideoCapture(0)
overlay = cv2.imread("overlay.png", -1)

def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30), flags = cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

def draw_overlay(img, rect):
    x1, y1, x2, y2 = rect
    y=y2-y1 + 40
    x=x2-x1 + 40
    small = cv2.resize(overlay, (x, y))

    x_offset = x1 - 10
    y_offset = y1 - 10

    for c in range(0,3):
        img[y_offset:y_offset + small.shape[0], x_offset:x_offset+ small.shape[1], c] = small[:,:,c] * (small[:,:,3]/255.0) + img[y_offset:y_offset+small.shape[0], x_offset:x_offset+small.shape[1], c] * (1.0 - small[:,:,3]/255.0)

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

ret = True
while ret:
    # Capture frame-by-frame
    ret, frame = video_capture.read(-1)

    if ret == True:
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      gray = cv2.equalizeHist(gray)

      found = detect(gray, face1)

      if len(found) > 0:
          #for rect in found:
          #    draw_overlay(frame, rect)

          draw_rects(frame, found, (0, 255, 0))

      # Display the resulting frame
      cv2.imshow('laugh', frame)
      cv2.waitKey(1)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
