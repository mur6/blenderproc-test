import json
import os
import pathlib
import shutil
import sys

import cv2
from PIL import Image
import albumentations as A


def load(json_file_path):
    t = json_file_path.read_text()
    d = json.loads(t)
    return d


# OUTPUT_DATA_DIR =
def _iter_coco_anno(im_list, anno_list, image_dir):
    assert len(im_list) == len(anno_list)
    for im, anno in zip(im_list, anno_list):
        assert im["id"] == anno["image_id"]
        file_name = image_dir / im["file_name"]
        assert file_name.exists()
        yield file_name, im, anno


transform = A.Compose(
    # [
    #     A.RandomCrop(width=330, height=330),
    #     A.RandomBrightnessContrast(p=0.2),
    # ],
    [
        # A.RandomCrop(width=512, height=512),
        A.Rotate(limit=40, p=0.9, border_mode=cv2.BORDER_CONSTANT),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.OneOf(
            [
                A.Cutout(
                    num_holes=30, max_h_size=30, max_w_size=30, fill_value=64, p=1
                ),
                # A.GridDistortion(p=0.1),
                A.PiecewiseAffine(p=0.5),
            ],
            p=0.75,
        ),
        A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.9),
        A.OneOf(
            [
                A.Blur(blur_limit=3, p=0.5),
                A.ColorJitter(p=0.5),
            ],
            p=0.3,
        ),
        # A.OneOf(
        #     [
        #         A.IAAAdditiveGaussianNoise(),
        #         A.GaussNoise(),
        #         A.RandomBrightnessContrast(),
        #     ],
        #     p=0.3,
        # ),
    ],
    keypoint_params=A.KeypointParams(
        format="xy",
        # label_fields=["class_labels"],
        remove_invisible=True,
        # angle_in_degrees=True,
    ),
    bbox_params=A.BboxParams(format="coco", label_fields=["class_labels"]),
)


def reg(file_path, keypoints, bbox):
    print(bbox)

    def _iter():
        for i in range(0, len(keypoints), 3):
            yield tuple(keypoints[i : i + 2])

    keypoints = list(_iter())
    print(keypoints)
    f = str(file_path)
    image = cv2.imread(f)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    images_list = [image]
    saved_keypoints_list = [keypoints]
    saved_bboxes_list = [bbox]
    for _ in range(15):
        transformed = transform(
            image=image, keypoints=keypoints, bboxes=[bbox], class_labels=["mat"]
        )
        transformed_image = transformed["image"]
        transformed_keypoints = transformed["keypoints"]
        transformed_bboxes = transformed["bboxes"]
        if len(transformed_bboxes) == 0:
            continue
        # print(transformed_keypoints)

        images_list.append(transformed_image)
        saved_keypoints_list.append(transformed_keypoints)
        saved_bboxes_list.append(transformed_bboxes[0])
    plot_examples(
        images_list,
        saved_bboxes_list,
        saved_keypoints_list,
    )


def main(base_dir):
    image_dir = base_dir / "mathand" / "train"
    coco_json = base_dir / "mathand_train.json"
    d = load(coco_json)
    im_list = d["images"]
    anno_list = d["annotations"]
    for file_path, im, anno in _iter_coco_anno(im_list, anno_list, image_dir):
        # print(anno)
        reg(file_path, anno["keypoints"], anno["bbox"])
        break


###########################################
# import random
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
    x_min, y_min, x_max, y_max = map(int, bbox)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, thickness)
    return img


if __name__ == "__main__":
    base_dir = sys.argv[1]
    main(pathlib.Path(base_dir))
