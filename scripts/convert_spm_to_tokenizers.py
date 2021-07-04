import argparse
import sys

import httpimport
import tokenizers

SENTENCEPIECE_URI = "https://raw.githubusercontent.com/google/sentencepiece/master/python/src/sentencepiece"

parser = argparse.ArgumentParser(description="convert sentencepiece unigram to tokenizers modl")
parser.add_argument("--sp-model-path", type=str, required=True)
parser.add_argument("--output-path", type=str, required=True)


def main(args: argparse.Namespace):
    with httpimport.remote_repo(["sentencepiece_model_pb2"], SENTENCEPIECE_URI):
        import sentencepiece_model_pb2

        tokenizer = tokenizers.SentencePieceUnigramTokenizer.from_spm(args.sp_model_path)
        tokenizer.save(args.output_path)


if __name__ == "__main__":
    sys.exit(main(parser.parse_args()))
