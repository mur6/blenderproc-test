import json

import pathlib
import argparse
import itertools

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


def convert_to_coco_format(keypoints):
    lis = []
    keypoints = [map(int, t) for t in keypoints]
    for x, y in keypoints:
        lis += [x, y, 2]
    return lis


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


# from collections import namedtuple
from dataclasses import dataclass

# AugmentedItem = namedtuple("AugmentedItem", "cv2_image im anno")


@dataclass
class AugmentedItem:
    cv2_image: str
    im: dict[str, str]
    anno: dict[str, str]
    keypoints: list[tuple[str, int]]
    bbox: list[int]

    # def total_cost(self) -> float:
    #     return self.unit_price * self.quantity_on_hand
    def __post_init__(self):
        self.anno["keypoints"] = convert_to_coco_format(self.keypoints)
        self.anno["bbox"] = list(map(int, self.bbox))


def _iter_augmented_item(im_list, anno_list, *, image_dir, aug_count):
    # idx = 1
    for file_path, im, anno in _iter_coco_anno(im_list, anno_list, image_dir):
        original_image = load_as_cv2(file_path)
        keypoints = convert_from_coco_format(anno["keypoints"])
        bbox = anno["bbox"]
        yield AugmentedItem(original_image, dict(im), dict(anno), keypoints, bbox)

        for t_image, t_keypoints, t_box in _iter_augment(
            original_image, keypoints=keypoints, bbox=bbox, aug_count=aug_count
        ):
            # idx += 1
            # yield (idx, file_path)
            yield AugmentedItem(t_image, dict(im), dict(anno), t_keypoints, t_box)


from operator import attrgetter


def main_for_aug(base_dir, target_dir, *, aug_count):
    image_dir = base_dir / "images"  # / "mathand" / "train"
    coco_json = base_dir / "mathand_train.json"
    d = load(coco_json)
    new_d = dict(d)
    im_list = d["images"]
    anno_list = d["annotations"]
    it = _iter_augmented_item(
        im_list, anno_list, image_dir=image_dir, aug_count=aug_count
    )

    def _iter_new_data():
        for idx, aug_item in enumerate(it, 1):
            print(idx, aug_item.im, aug_item.anno)
            new_file_path = target_dir / "images" / f"{idx:06}.jpg"
            print(new_file_path.name)
            cv2.imwrite(str(new_file_path), aug_item.cv2_image)

            aug_item.im["id"] = idx
            aug_item.im["file_name"] = new_file_path.name
            aug_item.anno["image_id"] = idx
            aug_item.anno["id"] = idx
            yield aug_item

    items = list(_iter_new_data())

    new_d["images"] = list(map(attrgetter("im"), items))  # new_im_list
    new_d["annotations"] = list(map(attrgetter("anno"), items))  # new_anno_list
    (target_dir / "mathand_train.json").write_text(json.dumps(new_d, indent=4))


def load_as_cv2(file_path):
    f = str(file_path)
    image = cv2.imread(f)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


from matplotlib import pyplot as plt


def main_for_vis_data(base_dir):
    image_dir = base_dir / "images"  # / "mathand" / "train"
    coco_json = base_dir / "mathand_train.json"
    d = load(coco_json)
    im_list = d["images"]
    anno_list = d["annotations"]

    fig = plt.figure(figsize=(15, 15))
    columns = 8
    rows = 4
    count = 1

    def _iter(keypoints):
        for i in range(0, len(keypoints), 3):
            yield tuple(keypoints[i : i + 2])

    coco_anno_iter = _iter_coco_anno(im_list, anno_list, image_dir)

    for file_path, im, anno in itertools.islice(coco_anno_iter, columns * rows):
        assert im["id"] == anno["image_id"]
        keypoints = list(_iter(anno["keypoints"]))
        print(f"keypoints: {keypoints}")
        image = load_as_cv2(file_path)
        src.tools.aug.utils.visualize_bbox_wh(image, anno["bbox"], keypoints)
        fig.add_subplot(rows, columns, count)
        count += 1
        plt.imshow(image)
    plt.show()


parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(dest="subcommand")
vis_data_parser = subparsers.add_parser("vis", help="Vis coco dataset")
vis_data_parser.add_argument("base_dir", type=pathlib.Path)
aug_image_parser = subparsers.add_parser("aug", help="Execute image data augmentation")
aug_image_parser.add_argument("source_dir", type=pathlib.Path)
aug_image_parser.add_argument("target_dir", type=pathlib.Path)
aug_image_parser.add_argument("--aug_count", type=int, default=8)


if __name__ == "__main__":
    args = parser.parse_args()
    print(args.subcommand)
    if args.subcommand == "vis":
        main_for_vis_data(args.base_dir)
    if args.subcommand == "aug":
        (args.target_dir / "images").mkdir(exist_ok=True)
        main_for_aug(args.source_dir, args.target_dir, aug_count=args.aug_count)
