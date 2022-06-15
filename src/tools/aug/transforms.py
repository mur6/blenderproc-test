import albumentations as A

custom_transform = A.Compose(
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
                # A.Affine(p=0.5),
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
        A.OneOf(
            [
                A.IAAAdditiveGaussianNoise(),
                A.GaussNoise(),
                A.RandomBrightnessContrast(p=0.2),
            ],
            p=0.4,
        ),
    ],
    keypoint_params=A.KeypointParams(
        format="xy",
        # label_fields=["class_labels"],
        remove_invisible=True,
        # angle_in_degrees=True,
    ),
    bbox_params=A.BboxParams(format="coco", label_fields=["class_labels"]),
)
