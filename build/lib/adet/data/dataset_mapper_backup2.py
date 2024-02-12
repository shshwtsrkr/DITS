import copy
import logging
import os.path as osp

import numpy as np
import torch
from fvcore.common.file_io import PathManager
from PIL import Image
from pycocotools import mask as maskUtils

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.detection_utils import SizeMismatchError
from detectron2.structures import BoxMode

from .augmentation import RandomCropWithInstance
from .detection_utils import (annotations_to_instances, build_augmentation,
                              transform_instance_annotations)

import cv2
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
import os
from inference_realesrgan import run_realesrgan

"""
This file contains the default mapping that's applied to "dataset dicts".
"""

__all__ = ["DatasetMapperWithBasis"]

logger = logging.getLogger(__name__)


def segmToRLE(segm, img_size):
    h, w = img_size
    if type(segm) == list:
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(segm, h, w)
        rle = maskUtils.merge(rles)
    elif type(segm["counts"]) == list:
        # uncompressed RLE
        rle = maskUtils.frPyObjects(segm, h, w)
    else:
        # rle
        rle = segm
    return rle


def segmToMask(segm, img_size):
    rle = segmToRLE(segm, img_size)
    m = maskUtils.decode(rle)
    return m

def filter_empty_instances(instances):
    """
    Filter out empty instances in an `Instances` object.

    Args:
        instances (Instances):
        by_box (bool): whether to filter out instances with empty boxes
        by_mask (bool): whether to filter out instances with empty masks
        box_threshold (float): minimum width and height to be considered non-empty
        return_mask (bool): whether to return boolean mask of filtered instances

    Returns:
        Instances: the filtered instances.
        tensor[bool], optional: boolean mask of filtered instances
    """
    pass
    r = []
    r.append(instances.gt_boxes.nonempty())

    if not r:
        return instances
    m = r[0]
    for x in r[1:]:
        m = m & x

    return instances[m]


class DatasetMapperWithBasis(DatasetMapper):
    """
    This caller enables the default Detectron2 mapper to read an additional basis semantic label
    """

    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)

        # Rebuild augmentations
        logger.info(
            "Rebuilding the augmentations. The previous augmentations will be overridden."
        )
        self.augmentation = build_augmentation(cfg, is_train)

        if cfg.INPUT.CROP.ENABLED and is_train and cfg.MODEL.TRANSFORMER.BOUNDARY_HEAD:
            self.augmentation.insert(
                0,
                RandomCropWithInstance(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                    cfg.INPUT.CROP.CROP_INSTANCE,
                ),
            )
            logging.getLogger(__name__).info(
                "Cropping used in training: " + str(self.augmentation[0])
            )
        if cfg.INPUT.ROTATE and is_train:
            if cfg.MODEL.TRANSFORMER.BOUNDARY_HEAD:
                self.augmentation.insert(0, T.RandomRotation(angle=[-45, 45]))
            else:
                self.augmentation.insert(0, T.RandomRotation(angle=[-90, 90]))



    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        print(dataset_dict)
        enhancement_factor = 1.0  # Initialize to 1.0 (i.e., no enhancement)
        print(f"Current working directory: {os.getcwd()}")

        # Load the set of enhanced image file paths
        enhanced_images = set()
        with open("adet/data/enhanced_images.txt", "r") as f:
            for line in f:
                enhanced_images.add(line.strip())

        try:
            image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
            h, w = image.shape[:2]

            # Check if the image should be enhanced based on its file path
            if dataset_dict["file_name"] in enhanced_images:
                enhancement_factor = 2.0  # Update the annotations and other parameters by a factor of 1.5
                print(f"++++++++Enhancing annotations for image {dataset_dict['file_name']} by factor of {enhancement_factor}++++++++")

                # Update bounding boxes and segmentation masks
                for anno in dataset_dict["annotations"]:
                    anno["bbox"] = [coord * enhancement_factor for coord in anno["bbox"]]
                    if "segmentation" in anno:
                        if type(anno["segmentation"]) == list:
                            anno["segmentation"] = [[coord * enhancement_factor for coord in polygon] for polygon in anno["segmentation"]]
                        elif type(anno["segmentation"]) == dict:
                            anno["segmentation"] = maskUtils.resize(anno["segmentation"], (int(h * enhancement_factor), int(w * enhancement_factor)))

                dataset_dict["width"], dataset_dict["height"] = int(w * enhancement_factor), int(h * enhancement_factor)
                
            else:
                print(f"++++++++Keeping image {dataset_dict['file_name']} and annotations as they are++++++++")
                # Do not enhance, keep annotation as is
                dataset_dict["width"], dataset_dict["height"] = w, h

        except Exception as e:
            print(dataset_dict["file_name"])
            print(e)
            raise e
            
        try:
            utils.check_image_size(dataset_dict, image)
        except SizeMismatchError as e:
            # Set initial expected dimensions based on original image dimensions
            expected_wh = (w, h)

            # Update expected dimensions based on enhancement factor
            expected_wh = (int(expected_wh[0] * enhancement_factor), int(expected_wh[1] * enhancement_factor))

            # Update image dimensions based on enhancement factor
            image_wh = (int(w * enhancement_factor), int(h * enhancement_factor))

            if image_wh == expected_wh:
                print("transposing image {}".format(dataset_dict["file_name"]))
                image = image.transpose(1, 0, 2)
            else:
                print("The error occurred HERE")
                print(f"Expected dimensions: {expected_wh}, but got: {image_wh}")
                raise e

        ######################################################################
        boxes = np.asarray(
            [
                BoxMode.convert(
                    instance["bbox"], instance["bbox_mode"], BoxMode.XYXY_ABS
                )
                for instance in dataset_dict["annotations"]
            ]
        )
        ######################################################################

        # aug_input = T.StandardAugInput(image)
        aug_input = T.StandardAugInput(image, boxes=boxes)

        transforms = aug_input.apply_augmentations(self.augmentation)
        image = aug_input.image

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            dataset_dict.pop("pano_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                transform_instance_annotations(
                    obj,
                    transforms,
                    image_shape,
                    keypoint_hflip_indices=self.keypoint_hflip_indices,
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = annotations_to_instances(
                annos, image_shape, mask_format=self.instance_mask_format
            )

            # dataset_dict["instances"] = instances
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        return dataset_dict
