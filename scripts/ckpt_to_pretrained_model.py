import argparse
import sys

import tensorflow as tf
from transformers import BartConfig, TFBartForConditionalGeneration

from transformers_bart_pretrain.utils import get_logger

parser = argparse.ArgumentParser(description="convert tensorflow checkpoint to huggingface pretrained format")
parser.add_argument("--model-config-path", type=str, required=True, help="model config file")
parser.add_argument("--pretrained-checkpoint", type=str, required=True, help="pretrained tensorflow checkpoint path")
parser.add_argument("--output-dir", type=str, required=True, help="output pretrained model directory")
parser.add_argument("--with-torch", action="store_true", help="save converted torch weight together")


def main(args: argparse.Namespace):
    logger = get_logger(__name__)

    logger.info("[+] Load config")
    config = BartConfig.from_pretrained(args.model_config_path)

    logger.info("[+] Initialize model")
    model = TFBartForConditionalGeneration(config)
    model(input_ids=tf.keras.Input([None], dtype=tf.int32))

    logger.info("[+] Load weights")
    model.load_weights(args.pretrained_checkpoint)

    logger.info(f"[+] Save pretrained format to {args.output_dir}")
    model.save_pretrained(args.output_dir)

    if args.with_torch:
        from transformers import BartForConditionalGeneration

        logger.info(f"[+] Save converted torch pretrained model to {args.output_dir}")
        model_pt = BartForConditionalGeneration.from_pretrained(args.output_dir, from_tf=True)
        model_pt.save_pretrained(args.output_dir)


if __name__ == "__main__":
    sys.exit(main(parser.parse_args()))
