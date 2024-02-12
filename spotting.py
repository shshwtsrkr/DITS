# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import argparse

from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from adet.config import get_cfg

# constants
WINDOW_NAME = "COCO detections"

confidence_array = []

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg

def run_on_images(input_path, output_path):
    # Set global args object
    global args

    class Args:
        config_file = "configs/ViTAEv2_S/pretrain/150k_tt_mlt_13_15.yaml"
        opts = ["MODEL.WEIGHTS", "pretrained/vitaev2-s_pretrain_synth-tt-mlt-13-15-textocr.pth"]
        input = [input_path]
        output = output_path
        confidence_threshold = 0.3

    args = Args()

    from predictor import VisualizationDemo  # Import after args is set up

    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)



    if os.path.isdir(args.input[0]):
        args.input = [os.path.join(args.input[0], fname) for fname in os.listdir(args.input[0])]
    elif len(args.input) == 1:
        args.input = glob.glob(os.path.expanduser(args.input[0]))
        assert args.input, "The input path(s) was not found"
    for path in tqdm.tqdm(args.input, disable=not args.output):
        img = read_image(path, format="BGR")
        start_time = time.time()



        predictions, visualized_output, confidence_scores = demo.run_on_image(img)  # Changed line
        logger.info(
            "{}: detected {} instances in {:.2f}s".format(
                path, len(predictions["instances"]), time.time() - start_time
            )
        )
        #logger.info(f"Confidence scores: {confidence_scores}")  # Added line

        if args.output:
            if os.path.isdir(args.output):
                assert os.path.isdir(args.output), args.output
                out_filename = os.path.join(args.output, os.path.basename(path))
            else:
                assert len(args.input) == 1, "Please specify a directory with args.output"
                out_filename = args.output
            visualized_output.save(out_filename)
        else:
            cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
            if cv2.waitKey(0) == 27:
                break  # esc to quit
    return confidence_scores

