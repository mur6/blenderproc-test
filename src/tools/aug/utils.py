import cv2
from matplotlib import pyplot as plt

# # import matplotlib.patches as patches
# # import numpy as np
# import albumentations as A


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
        img = visualize_bbox_wh(images[i - 1], bboxes[i - 1], keypoints[i - 1])
        # else:
        #     img = images[i - 1]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
    plt.show()


# def _visualize_bbox_min_max_mode(img, bbox, keypoints, color=(255, 0, 0), thickness=5):
#     # """Visualizes a single bounding box on the image"""
#     # x_min, y_min, x_max, y_max = map(int, bbox)
#     keypoints = [tuple(map(int, xy)) for xy in keypoints]
#     for x, y in keypoints:
#         cv2.circle(img, (x, y), 1, (0, 255, 0), thickness)
#     x_min, y_min, x_max, y_max = map(int, bbox)
#     cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, thickness)
#     return img


def visualize_bbox_wh(img, bbox, keypoints, color=(255, 0, 0), thickness=5):
    keypoints = [tuple(map(int, xy)) for xy in keypoints]
    for x, y in keypoints:
        cv2.circle(img, (x, y), 1, (0, 255, 0), thickness)
    x_min, y_min, w, h = map(int, bbox)
    print((x_min, y_min), (x_min + w, y_min + h))
    cv2.rectangle(img, (x_min, y_min), (x_min + w, y_min + h), color, thickness)
    return img
