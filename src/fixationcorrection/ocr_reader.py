from __future__ import annotations

import math

import cv2
import numpy as np
import pytesseract
from pytesseract import Output


def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def find_closest_top_box(px, py, list_of_centers):
    closest_top_box = None
    closest_distance = float('inf')
    for center in list_of_centers:
        cx, cy = center
        if cy < py and np.hypot(px-cx, py-cy) < closest_distance:
            closest_distance = np.hypot(px-cx, py-cy)
            closest_top_box = center
    return closest_top_box


def find_closest_left_box(px, py, list_of_centers):
    closest_left_box = None
    closest_distance = float('inf')
    for center in list_of_centers:
        cx, cy = center
        if cx < px and np.hypot(px-cx, py-cy) < closest_distance:
            closest_distance = np.hypot(px-cx, py-cy)
            closest_left_box = center
    return closest_left_box


def find_closest_bottom_box(px, py, list_of_centers):
    closest_bottom_box = None
    closest_distance = float('inf')
    for center in list_of_centers:
        cx, cy = center
        if cy > py and np.hypot(px-cx, py-cy) < closest_distance:
            closest_distance = np.hypot(px-cx, py-cy)
            closest_bottom_box = center
    return closest_bottom_box


def find_closest_right_box(px, py, list_of_centers):
    closest_right_box = None
    closest_distance = float('inf')
    for center in list_of_centers:
        cx, cy = center
        if cx > px and np.hypot(px-cx, py-cy) < closest_distance:
            closest_distance = np.hypot(px-cx, py-cy)
            closest_right_box = center
    return closest_right_box


class OCR_Reader:
    def __init__(self, path_to_image):
        self.path_to_image = path_to_image
        self.list_of_centers = []

    def read_image(self):
        img = cv2.imread(self.path_to_image)
        d = pytesseract.image_to_data(img, output_type=Output.DICT)
        n_boxes = len(d['text'])
        for i in range(n_boxes):
            if int(d['conf'][i]) > 60 and d['width'][i] / d['height'][i] > 0.3 and d['height'][
                i
            ] > 2:  # make sure there are no weird long boxes that are not around words
                (x, y, w, h) = (
                    d['left'][i], d['top']
                    [i], d['width'][i], d['height'][i],
                )
                center = (int(x + w / 2), int(y + h / 2))
                self.list_of_centers.append(center)

                # Uncomment this to see the boxes around the words
                # img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # img = cv2.circle(img, center, 5, (0, 0, 255), -1)
        # cv2.imshow('image', img)
        # cv2.waitKey(0)
        return d


# path = 'reading-dickens-1.png'
# reader = OCR_Reader(path)
# dic = reader.read_image()

# print(reader.list_of_centers)
# left = reader.find_closest_left_box(300,300)
# right = reader.find_closest_right_box(300,300)
# up = reader.find_closest_top_box(300,300)
# down = reader.find_closest_bottom_box(300,300)


# my_img = cv2.imread('reading-dickens-1.png')

# cv2.circle(my_img, (10,10),5,(0, 0, 255),-1)
# cv2.circle(my_img, (300,300),10,(0, 0, 0),-1)
# cv2.circle(my_img, left,5,(147, 50, 168),-1)
# cv2.circle(my_img, right,5,(0, 0, 255),-1)
# cv2.circle(my_img, up,5,(12, 201, 189),-1)
# cv2.circle(my_img, down,5,(12, 18, 201),-1)
# Coordinates for box
# box_top_left = (50, 10)
# box_bottom_right = (250, 60)

# Draw the box
# cv2.rectangle(my_img, box_top_left, box_bottom_right, (200, 200, 200), -1)  # Grey background
# cv2.rectangle(my_img, box_top_left, box_bottom_right, (0, 0, 0), 2)  # Black border

# Add custom text
# cv2.putText(my_img, "My Word", (box_top_left[0] + 10, box_top_left[1] + 30),
#            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

# cv2.imshow('image', my_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
