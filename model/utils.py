# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/util/misc.py
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
from typing import List, Optional

import torch
import torch.distributed as dist
import torchvision
from torch import Tensor
import random
from detectron2.data.transforms import Augmentation, CropTransform
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

def custom_mapper(dataset_dict):
    dataset_dict = dataset_dict.copy()
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    dataset_dict["height"], dataset_dict["width"] = image.shape[:2]

    aug = T.AugmentationList([
        RandomCropByClip(
            crop_choices_dict={
                "wide": [(512, 256), (640, 320)],
                "tall": [(256, 512), (320, 640)],
                "square": [(512, 512), (640, 640)]
            },
            clip_threshold=1.5
        ),
        T.ResizeShortestEdge([512, 768], 1024)
    ])
    aug_input = T.AugInput(image)
    transforms = aug(aug_input)
    image = aug_input.image
    
    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
    return dataset_dict

class RandomCropByClip(Augmentation):
    """
    自訂的 choice_by_clip augmentation
    根據圖片的 aspect ratio，從對應的候選列表中選 crop size
    """
    def __init__(self, crop_choices_dict, clip_threshold=1.5):
        """
        Args:
            crop_choices_dict: dict
                {
                  "wide": [(512, 256), (640, 320)],
                  "tall": [(256, 512), (320, 640)],
                  "square": [(512, 512), (640, 640)]
                }
            clip_threshold: float
                判斷 wide vs tall 的比例閾值
        """
        super().__init__()
        self.crop_choices_dict = crop_choices_dict
        self.clip_threshold = clip_threshold

    def get_transform(self, image):
        h, w = image.shape[:2]
        aspect = w / h

        if aspect > self.clip_threshold:
            # 寬圖
            choice = random.choice(self.crop_choices_dict["wide"])
        elif aspect < 1.0 / self.clip_threshold:
            # 長圖
            choice = random.choice(self.crop_choices_dict["tall"])
        else:
            # 方圖
            choice = random.choice(self.crop_choices_dict["square"])

        ch, cw = choice
        x0 = random.randint(0, max(0, w - cw))
        y0 = random.randint(0, max(0, h - ch))

        return CropTransform(x0, y0, cw, ch)
    
def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes

class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        if torchvision._is_tracing():
            # nested_tensor_from_tensor_list() does not export well to ONNX
            # call _onnx_nested_tensor_from_tensor_list() instead
            return _onnx_nested_tensor_from_tensor_list(tensor_list)

        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
    else:
        raise ValueError("not supported")
    return NestedTensor(tensor, mask)


# _onnx_nested_tensor_from_tensor_list() is an implementation of
# nested_tensor_from_tensor_list() that is supported by ONNX tracing.
@torch.jit.unused
def _onnx_nested_tensor_from_tensor_list(tensor_list: List[Tensor]) -> NestedTensor:
    max_size = []
    for i in range(tensor_list[0].dim()):
        max_size_i = torch.max(
            torch.stack([img.shape[i] for img in tensor_list]).to(torch.float32)
        ).to(torch.int64)
        max_size.append(max_size_i)
    max_size = tuple(max_size)

    # work around for
    # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    # m[: img.shape[1], :img.shape[2]] = False
    # which is not yet supported in onnx
    padded_imgs = []
    padded_masks = []
    for img in tensor_list:
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        padded_img = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
        padded_imgs.append(padded_img)

        m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
        padded_mask = torch.nn.functional.pad(m, (0, padding[2], 0, padding[1]), "constant", 1)
        padded_masks.append(padded_mask.to(torch.bool))

    tensor = torch.stack(padded_imgs)
    mask = torch.stack(padded_masks)

    return NestedTensor(tensor, mask=mask)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True