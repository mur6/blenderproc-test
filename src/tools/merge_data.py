import json
import os
import pathlib
import shutil
import sys


OUTPUT_DATA_DIR = pathlib.Path("data/outputs/coco3")


def load(p):
    t = p.read_text()
    d = json.loads(t)
    return d


def load_coco_data(dir):
    json_path = dir / "coco_annotations.json"
    d = load(json_path)
    im_list = d["images"]
    anno_list = d["annotations"]
    return im_list, anno_list, d


def load_keypoints_and_bbox_data(json_file_path):
    d = load(json_file_path)
    keypoints_list = d["keypoints_list"]
    bbox_list = d["bbox_list"]
    return keypoints_list, bbox_list


def _iter_all():
    for dir in sorted(OUTPUT_DATA_DIR.iterdir()):
        keypoints, bboxes = load_keypoints_and_bbox_data(
            dir / "keypoints_and_bbox.json"
        )
        im_list, anno_list, _ = load_coco_data(dir)
        yield from _iter_coco_anno(dir.name, im_list, anno_list, keypoints, bboxes)


def _iter_coco_anno(number_dir, im_list, anno_list, keypoints, bboxes):
    # print(len(im_list), len(anno_list), len(keypoints), len(bboxes))
    assert len(im_list) == len(anno_list) == len(keypoints) == len(bboxes)
    for im, anno, keypoint, bbox in zip(im_list, anno_list, keypoints, bboxes):
        assert im["id"] == anno["image_id"]
        file_name = im["file_name"]
        yield number_dir, file_name, im, anno, keypoint, bbox


def make_source_image_path(number_dir, file_name):
    p = OUTPUT_DATA_DIR / number_dir / file_name
    assert p.exists()
    return p


def image_file_copy(dist_dir):
    (dist_dir / "images").mkdir(exist_ok=True)
    for idx, (number_dir, file_name, im, anno, _, _) in enumerate(_iter_all(), 1):
        src_im_path = make_source_image_path(number_dir, file_name)
        dist_im_path = dist_dir / "images" / f"{idx:06}.jpg"
        shutil.copy(src_im_path, dist_im_path)
        # print(src_im_path, dist_im_path)


def _iter_image_infos():
    for idx, (_, _, im, anno, _, _) in enumerate(_iter_all(), 1):
        im["id"] = idx
        im["file_name"] = f"{idx:06}.jpg"
        yield im


def _iter_annotation_infos():
    for idx, (_, _, im, anno, keypoint, bbox) in enumerate(_iter_all(), 1):
        del anno["segmentation"]
        print(anno)
        anno["image_id"] = idx
        anno["id"] = idx
        anno["keypoints"] = keypoint
        anno["bbox"] = bbox
        anno["num_keypoints"] = 3
        print(anno)
        yield anno


def main(dist_dir):
    dist_dir = pathlib.Path(dist_dir)
    image_paths = []
    for dir in sorted(OUTPUT_DATA_DIR.iterdir()):
        image_paths += list((dir / "images").iterdir())

    image_file_copy(dist_dir)

    _, _, d = load_coco_data(OUTPUT_DATA_DIR / "1")
    d["images"] = list(_iter_image_infos())
    d["annotations"] = list(_iter_annotation_infos())
    js_str = json.dumps(d, indent=4)
    (dist_dir / "mathand_train.json").write_text(js_str)


if __name__ == "__main__":
    target_dir = sys.argv[1]
    main(target_dir)
