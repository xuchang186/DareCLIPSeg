import os
import cv2
import torch
import argparse
import logging
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets.dataloader import DatasetSegmentation, ValGenerator
from trainers import *
from utils.main_utils import load_cfg_from_cfg_file, read_text, normalize
import matplotlib.pyplot as plt

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config-file",
        required=True,
        type=str,
        help="Path to config file",
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help="Random seed for reproducibility."
    )

    parser.add_argument(
        '--prompt_design',
        type=str,
        default="original",
        help="Text prompt design."
    )

    parser.add_argument(
        "--data_percentage",
        type=int,
        default=100,
        help="Percentage of data to use."
    )

    parser.add_argument(
        "--source_dataset",
        type=str,
        help="source dataset name for loading trained model."
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="output directory"
    )

    parser.add_argument(
        "--run_name",
        type=str,
        default="",
        help="Directory name for this run, e.g. seed24 or seed24(2)."
    )

    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )

    args = parser.parse_args()

    cfg = load_cfg_from_cfg_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.update({k: v for k, v in vars(args).items()})

    return cfg

def logger_config(log_path):
    logger = logging.getLogger()
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger


def resolve_run_name(cfg):
    run_name = getattr(cfg, "run_name", "")

    if run_name is not None and str(run_name).strip() != "":
        return str(run_name).strip()

    return f"seed{cfg.seed}"


def main():
    cfg = get_arguments()

    if cfg.seed >= 0:
        print(f"Setting fixed seed: {cfg.seed}")
        set_random_seed(cfg.seed)

    cfg.DATASET.NAME = cfg.DATASET.NAME + f"_{cfg.data_percentage}" if cfg.data_percentage != 100 else cfg.DATASET.NAME
    run_name = resolve_run_name(cfg)

    results_root = os.path.join(
        cfg.output_dir,
        cfg.DATASET.NAME,
        "seg_results",
        run_name
    )
    unc_root = os.path.join(
        cfg.output_dir,
        cfg.DATASET.NAME,
        "unc_results",
        run_name
    )
    os.makedirs(results_root, exist_ok=True)
    os.makedirs(unc_root, exist_ok=True)

    logger = logger_config(os.path.join(results_root, "log.txt"))

    logger.info("************")
    logger.info("** Config **")
    logger.info("************")
    logger.info(cfg)
    logger.info(f"Run name: {run_name}")
    logger.info(f"Segmentation result directory: {results_root}")
    logger.info(f"Uncertainty result directory: {unc_root}")

    backbone_name = cfg.MODEL.BACKBONE.replace("/", "-")
    results_name = (
        f"DareCLIPSeg_"
        f"{cfg.MODEL.CLIP_MODEL}_"
        f"{backbone_name}"
    )

    checkpoint_type = "latest" if cfg.TEST.USE_LATEST else "best_dice"

    checkpoint_dataset = cfg.source_dataset if cfg.data_percentage == 100 else cfg.DATASET.NAME
    checkpoint_path = os.path.join(
        cfg.output_dir,
        checkpoint_dataset,
        "trained_models",
        run_name,
        f"{results_name}_{checkpoint_type}.pth"
    )

    logger.info(f"Loading checkpoint: {checkpoint_path}")

    if cfg.MODEL.CLIP_MODEL == "unimedclip":
        model = build_dareclipseg_unimedclip(cfg)
    else:
        raise NotImplementedError(f"Unknown CLIP model: {cfg.MODEL.CLIP_MODEL}")

    checkpoint = torch.load(checkpoint_path, map_location=cfg.MODEL.DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval().to(cfg.MODEL.DEVICE)

    test_tf = ValGenerator(output_size=[cfg.DATASET.SIZE, cfg.DATASET.SIZE])
    test_text_file = f"Test_text_{cfg.prompt_design}.xlsx"
    test_text = read_text(cfg.DATASET.TEXT_PROMPT_PATH + test_text_file)

    test_dataset = DatasetSegmentation(
        cfg.DATASET.TEST_PATH,
        cfg.DATASET.NAME,
        test_text,
        test_tf,
        image_size=cfg.DATASET.SIZE
    )
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    save_dir = os.path.join(
        results_root,
        results_name + f"_Prompt-{cfg.prompt_design}"
    )
    save_unc_dir = os.path.join(
        unc_root,
        results_name + f"_Prompt-{cfg.prompt_design}"
    )

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_unc_dir, exist_ok=True)

    with torch.no_grad():
        for batch in tqdm(test_dataloader):

            seg_samples = model(
                image=batch["image"].to(cfg.MODEL.DEVICE),
                text=batch["text_prompt"],
                num_samples=cfg.TEST.NUM_SAMPLES
            )

            seg_samples = torch.sigmoid(seg_samples)

            seg_logits = seg_samples.mean(dim=0)
            seg_unc = -(
                    seg_logits * torch.log(seg_logits + 1e-8) +
                    (1 - seg_logits) * torch.log(1 - seg_logits + 1e-8)
            )

            mask_preds = seg_logits > 0.5

            dataset_names = batch["dataset_name"]
            mask_names = batch["mask_name"]

            for i in range(len(dataset_names)):
                pred_mask = mask_preds[i].cpu().numpy().astype(np.uint8)
                mask_name = mask_names[i]

                binary_pred = np.uint8(pred_mask > 0)
                cv2.imwrite(os.path.join(save_dir, mask_name), binary_pred * 255)

                u_map = seg_unc[i].cpu().numpy()
                u_map = normalize(u_map)
                colormap = plt.get_cmap('nipy_spectral')
                u_map_color = (colormap(u_map)[:, :, :3] * 255).astype(np.uint8)
                u_map_colored = cv2.cvtColor(u_map_color, cv2.COLOR_RGB2BGR)

                cv2.imwrite(os.path.join(save_unc_dir, mask_name), u_map_colored)

if __name__ == "__main__":
    main()
