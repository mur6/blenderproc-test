import json
import os
import pathlib
import sys

import cv2
import albumentations as A


import src.tools.aug.utils


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
    src.tools.aug.utils.plot_examples(
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


def check_images(file_path, keypoints, bbox):
    f = str(file_path)
    image = cv2.imread(f)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


from matplotlib import pyplot as plt


def main2(base_dir):
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
        image = check_images(file_path, keypoints, anno["bbox"])
        img = src.tools.aug.utils.visualize_bbox(image, anno["bbox"], keypoints)
        fig.add_subplot(rows, columns, count)
        count += 1
        plt.imshow(image)
    plt.show()


if __name__ == "__main__":
    base_dir = sys.argv[1]
    main2(pathlib.Path(base_dir))
