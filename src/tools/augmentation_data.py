import PIL
import json
import os
import pathlib
import shutil
import sys


def load(json_file_path):
    t = json_file_path.read_text()
    d = json.loads(t)
    return d


# OUTPUT_DATA_DIR =
def _iter_coco_anno(im_list, anno_list):
    assert len(im_list) == len(anno_list)
    for im, anno in zip(im_list, anno_list):
        assert im["id"] == anno["image_id"]
        # file_name = im["file_name"]
        yield im, anno


def main(base_dir):
    image_data_dir = base_dir / "mathand" / "images"
    coco_json = base_dir / "mathand_train.json"
    d = load(coco_json)
    im_list = d["images"]
    anno_list = d["annotations"]
    for im, anno in _iter_coco_anno(im_list, anno_list):
        print(im)


if __name__ == "__main__":
    base_dir = sys.argv[1]
    main(pathlib.Path(base_dir))
