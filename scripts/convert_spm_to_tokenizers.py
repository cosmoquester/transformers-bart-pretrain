import argparse
import sys

import httpimport
import tokenizers
from transformers import PreTrainedTokenizerFast

SENTENCEPIECE_URI = "https://raw.githubusercontent.com/google/sentencepiece/master/python/src/sentencepiece"

parser = argparse.ArgumentParser(description="convert sentencepiece unigram to tokenizers modl")
parser.add_argument("--sp-model-path", type=str, required=True)
parser.add_argument("--output-path", type=str, required=True)
parser.add_argument("--bos-token", type=str, default="[BOS]")
parser.add_argument("--eos-token", type=str, default="[EOS]")
parser.add_argument("--unk-token", type=str, default="[UNK]")
parser.add_argument("--sep-token", type=str, default="[SEP]")
parser.add_argument("--pad-token", type=str, default="[PAD]")
parser.add_argument("--mask-token", type=str, default="[MASK]")


def main(args: argparse.Namespace):
    with httpimport.remote_repo(["sentencepiece_model_pb2"], SENTENCEPIECE_URI):
        import sentencepiece_model_pb2

        tokenizer = tokenizers.SentencePieceUnigramTokenizer.from_spm(args.sp_model_path)

    pretrained_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token=args.bos_token,
        eos_token=args.eos_token,
        unk_token=args.unk_token,
        sep_token=args.sep_token,
        pad_token=args.pad_token,
        mask_token=args.mask_token,
    )
    pretrained_tokenizer.save_pretrained(args.output_path)
    print(f"Saved to {args.output_path}")


if __name__ == "__main__":
    sys.exit(main(parser.parse_args()))
