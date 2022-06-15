import cv2
from matplotlib import pyplot as plt

# import matplotlib.patches as patches
# import numpy as np
import albumentations as A


def visualize(image):
    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.imshow(image)
    plt.show()


def plot_examples(images, bboxes, keypoints):
    fig = plt.figure(figsize=(15, 15))
    columns = 4
    rows = 5

    for i in range(1, len(images)):
        img = visualize_bbox(images[i - 1], bboxes[i - 1], keypoints[i - 1])
        # else:
        #     img = images[i - 1]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
    plt.show()


def visualize_bbox(img, bbox, keypoints, color=(255, 0, 0), thickness=5):
    # """Visualizes a single bounding box on the image"""
    # x_min, y_min, x_max, y_max = map(int, bbox)
    print(keypoints)
    keypoints = [tuple(map(int, xy)) for xy in keypoints]
    for x, y in keypoints:
        cv2.circle(img, (x, y), 1, (0, 255, 0), thickness)
    x_min, y_min, w, h = map(int, bbox)
    cv2.rectangle(img, (x_min, y_min), (x_min + w, y_min + h), color, thickness)
    return img
