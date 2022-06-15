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


def main():
    coco_json = pathlib.Path("augmentation/mathand_train.json")
    d = load(coco_json)
    im_list = d["images"]
    anno_list = d["annotations"]


if __name__ == "__main__":
    main()
