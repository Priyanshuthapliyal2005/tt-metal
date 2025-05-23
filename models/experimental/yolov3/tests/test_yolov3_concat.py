# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

from loguru import logger

import ttnn

from models.experimental.yolov3.reference.models.common import (
    DetectMultiBackend,
)
from models.experimental.yolov3.tt.yolov3_concat import TtConcat
from models.utility_functions import (
    comp_allclose_and_pcc,
    comp_pcc,
    torch2tt_tensor,
    tt2torch_tensor,
)


def test_concat_module(device, model_location_generator):
    torch.manual_seed(1234)

    # Load yolo
    model_path = model_location_generator("models", model_subdir="Yolo")
    data_path = model_location_generator("data", model_subdir="Yolo")

    data_image_path = str(data_path / "images")
    data_coco = str(data_path / "coco128.yaml")
    model_config_path = str(data_path / "yolov3.yaml")
    weights_loc = str(model_path / "yolov3.pt")

    reference_model = DetectMultiBackend(weights_loc, device=torch.device("cpu"), dnn=False, data=data_coco, fp16=False)
    state_dict = reference_model.state_dict()

    INDEX = 18
    base_address = f"model.model.{INDEX}"

    torch_model = reference_model.model.model[INDEX]

    tt_model = TtConcat(
        base_address=base_address,
        state_dict=state_dict,
        device=device,
    )

    # Create random Input image with channels > 3
    im_1 = torch.rand(1, 64, 512, 640)
    im_2 = torch.rand(1, 64, 512, 640)
    # Inference
    pred = torch_model([im_1, im_2])

    tt_im_1 = torch2tt_tensor(im_1, device, tt_layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_im_2 = torch2tt_tensor(im_2, device, tt_layout=ttnn.ROW_MAJOR_LAYOUT)
    x = [tt_im_1, tt_im_2]
    tt_pred = tt_model(x)

    # Compare outputs
    tt_output_torch = tt2torch_tensor(tt_pred)

    does_pass, pcc_message = comp_pcc(pred, tt_output_torch)
    logger.info(pcc_message)
    _, comp_out = comp_allclose_and_pcc(pred, tt_output_torch)
    logger.info(comp_out)

    if does_pass:
        logger.info("Yolo TtConcat Passed!")
    else:
        logger.warning("Yolo TtConcat Failed!")

    assert does_pass
