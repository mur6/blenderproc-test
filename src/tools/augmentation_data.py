import json
import pathlib
import sys

import cv2

# import albumentations as A

import src.tools.aug.utils
from src.tools.aug.transforms import custom_transform


def load(json_file_path):
    t = json_file_path.read_text()
    d = json.loads(t)
    return d


def _iter_coco_anno(im_list, anno_list, image_dir):
    assert len(im_list) == len(anno_list)
    for im, anno in zip(im_list, anno_list):
        assert im["id"] == anno["image_id"]
        file_name = image_dir / im["file_name"]
        assert file_name.exists()
        yield file_name, im, anno


def convert_from_coco_format(keypoints):
    def _iter():
        for i in range(0, len(keypoints), 3):
            yield tuple(keypoints[i : i + 2])

    return list(_iter())


# def exec_aug(original_image, *, keypoints, bbox, aug_count):
#     images_list = [image]
#     saved_keypoints_list = [keypoints]
#     saved_bboxes_list = [bbox]

#     cv2.imwrite("data/dst/lena_opencv_red.jpg", im)
#     src.tools.aug.utils.plot_examples(
#         images_list,
#         saved_bboxes_list,
#         saved_keypoints_list,
#     )


def _iter_augment(original_image, *, keypoints, bbox, aug_count):
    for _ in range(aug_count):
        transformed = custom_transform(
            image=original_image,
            keypoints=keypoints,
            bboxes=[bbox],
            class_labels=["mat"],
        )
        transformed_image = transformed["image"]
        transformed_keypoints = transformed["keypoints"]
        transformed_bboxes = transformed["bboxes"]
        if len(transformed_bboxes) == 0:
            return
        # images_list.append(transformed_image)
        # saved_keypoints_list.append(transformed_keypoints)
        # saved_bboxes_list.append(transformed_bboxes[0])
        yield transformed_image, transformed_keypoints, transformed_bboxes[0]


def main(base_dir):
    image_dir = base_dir / "images"  # / "mathand" / "train"
    coco_json = base_dir / "mathand_train.json"
    d = load(coco_json)
    im_list = d["images"]
    anno_list = d["annotations"]
    idx = 1
    for file_path, im, anno in _iter_coco_anno(im_list, anno_list, image_dir):
        original_image = load_as_cv2(file_path)
        keypoints = convert_from_coco_format(anno["keypoints"])
        bbox = anno["bbox"]
        for t_image, t_keypoints, t_box in _iter_augment(
            original_image, keypoints=keypoints, bbox=bbox, aug_count=3
        ):
            idx += 1


def load_as_cv2(file_path):
    f = str(file_path)
    image = cv2.imread(f)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


from matplotlib import pyplot as plt


def main_for_check_data(base_dir):
    image_dir = base_dir / "images"  # / "mathand" / "train"
    coco_json = base_dir / "mathand_train.json"
    d = load(coco_json)
    im_list = d["images"]
    anno_list = d["annotations"]

    fig = plt.figure(figsize=(15, 15))
    columns = 8
    rows = 8
    count = 1

    def _iter(keypoints):
        for i in range(0, len(keypoints), 3):
            yield tuple(keypoints[i : i + 2])

    for file_path, im, anno in _iter_coco_anno(im_list, anno_list, image_dir):
        assert im["id"] == anno["image_id"]
        keypoints = list(_iter(anno["keypoints"]))
        print(f"keypoints: {keypoints}")
        image = load_as_cv2(file_path)
        src.tools.aug.utils.visualize_bbox_wh(image, anno["bbox"], keypoints)
        fig.add_subplot(rows, columns, count)
        count += 1
        plt.imshow(image)
    plt.show()


if __name__ == "__main__":
    base_dir = sys.argv[1]
    main(pathlib.Path(base_dir))
