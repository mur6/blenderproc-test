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
    [
        A.RandomCrop(width=330, height=330),
        A.RandomBrightnessContrast(p=0.2),
    ],
    keypoint_params=A.KeypointParams(
        format="xy",
        label_fields=["class_labels"],
        remove_invisible=True,
        angle_in_degrees=True,
    ),
)


def reg(file_path):
    f = str(file_path)
    image = cv2.imread(f)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transformed = transform(image=image, keypoints=keypoints)
    transformed_image = transformed["image"]
    transformed_keypoints = transformed["keypoints"]


def main(base_dir):
    image_dir = base_dir / "mathand" / "train"
    coco_json = base_dir / "mathand_train.json"
    d = load(coco_json)
    im_list = d["images"]
    anno_list = d["annotations"]
    for file_path, im, anno in _iter_coco_anno(im_list, anno_list, image_dir):
        reg(file_path)
        break


if __name__ == "__main__":
    base_dir = sys.argv[1]
    main(pathlib.Path(base_dir))
