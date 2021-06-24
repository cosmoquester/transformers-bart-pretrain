import argparse
import glob
import os
import sys

import tensorflow as tf
import tensorflow_text as text
from tqdm import tqdm

# fmt: off
parser = argparse.ArgumentParser()
parser.add_argument("--input-path", type=str, required=True, help="Input File glob pattern")
parser.add_argument("--output-dir", type=str, help="Output path file or directory")
parser.add_argument("--sp-model-path", type=str, default="resources/sp-model/sp_model_unigram_16K.model", help="Sentencepiece model path")
parser.add_argument("--auto-encoding", action="store_true", help="If use autoencoding, dataset is .txt format else .tsv format")
# fmt: on


def read_data(file_path: str, tokenizer: text.SentencepieceTokenizer, auto_encoding: bool):
    @tf.function
    def tokenize_fn(source_text, target_text):
        # Tokenize & Add bos, eos
        source_tokens = tokenizer.tokenize(source_text)
        target_tokens = tokenizer.tokenize(target_text)
        return source_tokens, target_tokens

    if auto_encoding:
        duplicate = tf.function(lambda text: (text, text))
        dataset = tf.data.TextLineDataset(
            file_path,
            num_parallel_reads=tf.data.experimental.AUTOTUNE,
        ).map(duplicate)
    else:
        dataset = tf.data.experimental.CsvDataset(file_path, [tf.string, tf.string], field_delim="\t")

    serialize = tf.function(
        lambda source, target: tf.stack([tf.io.serialize_tensor(source), tf.io.serialize_tensor(target)])
    )
    dataset = (
        dataset.map(tokenize_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .map(serialize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .map(tf.io.serialize_tensor)
    )
    return dataset


def main(args: argparse.Namespace):
    input_files = glob.glob(args.input_path)

    # Load Sentencepiece model
    with open(args.sp_model_path, "rb") as f:
        tokenizer = text.SentencepieceTokenizer(f.read(), add_bos=True, add_eos=True)

    for file_path in tqdm(input_files):
        output_dir = args.output_dir if args.output_dir else os.path.dirname(file_path)
        file_name = os.path.basename(file_path)
        output_path = os.path.join(output_dir, os.path.splitext(file_name)[0] + ".tfrecord")

        # Write TFRecordFile
        dataset = read_data(file_path, tokenizer, args.auto_encoding)
        writer = tf.data.experimental.TFRecordWriter(output_path, "GZIP")
        writer.write(dataset)


if __name__ == "__main__":
    sys.exit(main(parser.parse_args()))
