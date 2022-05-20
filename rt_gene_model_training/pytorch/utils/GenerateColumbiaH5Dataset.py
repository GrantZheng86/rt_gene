from __future__ import print_function, division, absolute_import

import argparse
import glob
import os

import h5py
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from torchvision import transforms
from tqdm import tqdm

script_path = os.path.dirname(os.path.realpath(__file__))

# Augmentations following `prepare_dataset.m`: randomly crop and resize the image 10 times,
# along side two blurring stages, grayscaling and histogram normalisation
_required_size = (224, 224)
_transforms_list = [transforms.RandomResizedCrop(size=_required_size, scale=(0.85, 1.0)),  # equivalent to random 5px from each edge
                    transforms.RandomResizedCrop(size=_required_size, scale=(0.85, 1.0)),
                    transforms.RandomResizedCrop(size=_required_size, scale=(0.85, 1.0)),
                    transforms.RandomResizedCrop(size=_required_size, scale=(0.85, 1.0)),
                    transforms.RandomResizedCrop(size=_required_size, scale=(0.85, 1.0)),
                    transforms.RandomResizedCrop(size=_required_size, scale=(0.85, 1.0)),
                    transforms.RandomResizedCrop(size=_required_size, scale=(0.85, 1.0)),
                    transforms.RandomResizedCrop(size=_required_size, scale=(0.85, 1.0)),
                    transforms.RandomResizedCrop(size=_required_size, scale=(0.85, 1.0)),
                    transforms.RandomResizedCrop(size=_required_size, scale=(0.85, 1.0)),
                    transforms.Grayscale(num_output_channels=3),
                    lambda x: x.filter(ImageFilter.GaussianBlur(radius=1)),
                    lambda x: x.filter(ImageFilter.GaussianBlur(radius=3)),
                    lambda x: ImageOps.equalize(x)]  # histogram equalisation


def load_and_augment(file_path, augment=False):
    image = Image.open(file_path).resize(_required_size)
    augmented_images = [np.array(trans(image)) for trans in _transforms_list if augment is True]
    augmented_images.append(np.array(image))

    return np.array(augmented_images, dtype=np.uint8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Estimate gaze from images')
    parser.add_argument('--columbia_root', type=str, required=True, nargs='?', help='Path to the base directory of RT_GENE')
    parser.add_argument('--columbia_name', type=str, default='columbia_dataset')
    parser.add_argument('--augment_dataset', type=bool, required=False, default=False, help="Whether to augment the dataset with predefined transforms")
    parser.add_argument('--compress', action='store_true', dest="compress")
    parser.add_argument('--no-compress', action='store_false', dest="compress")
    parser.set_defaults(compress=False)
    args = parser.parse_args()

    _compression = "lzf" if args.compress is True else None

    subject_path = [os.path.join(args.columbia_root, "{:04d}/".format(_i)) for _i in range(1, 57)]

    hdf_file = h5py.File(os.path.abspath(os.path.join(args.columbia_root, "{}.hdf5".format(args.columbia_name))), mode='w')
    for subject_id, subject_data in enumerate(subject_path):
        subject_id = str("s{:03d}".format(subject_id))
        subject_grp = hdf_file.create_group(subject_id)

        left_img_path = os.path.join(subject_data, 'Processed_eyes/left_new')
        right_img_path = os.path.join(subject_data, 'Processed_eyes/right_new')
        if os.path.exists(left_img_path) and os.path.exists(right_img_path):
            image_index = 0
            for single_eye_img_name in tqdm(glob.glob('{}/*.png'.format(left_img_path))):
                left_image_name = single_eye_img_name
                right_image_name = single_eye_img_name.replace("left", "right")
                assert os.path.exists(right_image_name), "Error, right eye patch cannot be found"
                image_data = left_image_name.split('/')[-1]
                image_data = image_data.split('_')
                horizontal_head = float(image_data[3][:-1])
                vertical_head = 0.0
                horizontal_eye = float(image_data[4][:-1])
                vertical_eye = float(image_data[5][:-1])
                labels = [(horizontal_head, vertical_head), (horizontal_eye, vertical_eye)]

                image_grp = subject_grp.create_group("{:0=6d}".format(image_index))
                image_index += 1
                left_data = load_and_augment(left_image_name, augment=args.augment_dataset)
                right_data = load_and_augment(right_image_name, augment=args.augment_dataset)
                image_grp.create_dataset("left", data=left_data, compression=_compression)
                image_grp.create_dataset("right", data=right_data, compression=_compression)
                image_grp.create_dataset("label", data=labels)
        # with open(os.path.join(subject_data, "label_combined.txt"), "r") as f:
        #     _lines = f.readlines()
        #
        #     for line in tqdm(_lines, desc="Subject {}".format(subject_id)):
        #
        #         split = line.split(",")
        #         image_name = "{:0=6d}".format(int(split[0]))
        #         image_grp = subject_grp.create_group(image_name)
        #         left_img_path = os.path.join(subject_data, "inpainted/Processed_eyes/left_new/", "left_{:0=6d}_rgb.png".format(int(split[0])))
        #         right_img_path = os.path.join(subject_data, "inpainted/Processed_eyes/right_new/", "right_{:0=6d}_rgb.png".format(int(split[0])))
        #         if os.path.exists(left_img_path) and os.path.exists(right_img_path):
        #             head_phi = float(split[1].strip()[1:])
        #             head_theta = float(split[2].strip()[:-1])
        #             gaze_phi = float(split[3].strip()[1:])
        #             gaze_theta = float(split[4].strip()[:-1])
        #             labels = [(head_theta, head_phi), (gaze_theta, gaze_phi)]
        #
        #             left_data = load_and_augment(left_img_path, augment=args.augment_dataset)
        #             right_data = load_and_augment(right_img_path, augment=args.augment_dataset)
        #             image_grp.create_dataset("left", data=left_data, compression=_compression)
        #             image_grp.create_dataset("right", data=right_data, compression=_compression)
        #             image_grp.create_dataset("label", data=labels)

    hdf_file.flush()
    hdf_file.close()
