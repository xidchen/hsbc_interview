import collections
import csv
import datetime
import json
import os
import zipfile

import click
import ftfy
import numpy as np
import sentencepiece as spm
import tensorflow as tf
import tqdm


ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(ROOT_DIR, 'wiki_zh.zip')
DATA_ROOT_PATH = os.path.join(ROOT_DIR, 'data')
PROCESSED_DATA_PATH = os.path.join(DATA_ROOT_PATH, 'processed.txt')
BPE_TSV_PATH = os.path.join(DATA_ROOT_PATH, 'bpe_spm.tsv')
BPE_MODEL_PATH = os.path.join(DATA_ROOT_PATH, 'bpe_model')
TF_RECORDS_PATH = os.path.join(DATA_ROOT_PATH, 'tf_records')
BOS_ID = 3
EOS_ID = 4


def parse_zip_jsonl_files(input_zip_file_path: str) -> None:
    print('Pre-processing the text data.....')
    if not os.path.exists(DATA_ROOT_PATH):
        os.makedirs(DATA_ROOT_PATH)
    with open(PROCESSED_DATA_PATH, 'w', encoding='utf-8') as fw:
        with zipfile.ZipFile(input_zip_file_path) as zf:
            lof = [f for f in sorted(zf.namelist())
                   if not f.endswith(('/', 'e')) and not f.startswith('_')]
            for file in tqdm.tqdm(lof):
                with zf.open(file) as fr:
                    fw.writelines([ftfy.fix_text(
                        json.loads(line)['text'], normalization='NFKC')
                        for line in fr])


def train_byte_pair_encoding(vocab_size):
    print('Training BytePair encoding......')
    token_dict = collections.Counter()
    with open(PROCESSED_DATA_PATH, 'r', encoding='utf-8') as fr:
        for line in tqdm.tqdm(fr):
            token_dict.update(line.lower().split())

    with open(BPE_TSV_PATH, 'w', newline='', encoding='utf-8') as fw:
        tsv_output = csv.writer(fw, delimiter='\t')
        for word in token_dict:
            tsv_output.writerow([word, token_dict[word]])

    spmcmd = f'--input={BPE_TSV_PATH} ' \
             f'--model_prefix={BPE_MODEL_PATH} ' \
             f'--input_format=tsv ' \
             f'--vocab_size={vocab_size} ' \
             f'--user_defined_symbols=[SEP],[BOS],[EOS] ' \
             f'--hard_vocab_limit=false ' \
             f'--model_type=bpe ' \
             f'--pad_id=0 ' \
             f'--unk_id=1 ' \
             f'--bos_id=-1 ' \
             f'--eos_id=-1 ' \
             f'--pad_piece=[PAD] ' \
             f'--unk_piece=[UNK]'
    spm.SentencePieceTrainer.Train(spmcmd)


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def serialize_example(inputs, targets):
    feature = {
        'inputs': _int64_feature(inputs),
        'targets': _int64_feature(targets)
    }
    example_proto = tf.train.Example(
        features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def create_tf_records(min_seq_len, max_seq_len, per_file_limit=5000):
    print('Creating TF Records...............')
    s = spm.SentencePieceProcessor()
    s.Load(BPE_MODEL_PATH + '.model')
    if not os.path.exists(TF_RECORDS_PATH):
        os.makedirs(TF_RECORDS_PATH)
    filename = os.path.join(
        TF_RECORDS_PATH,
        str(datetime.datetime.now().timestamp()) + '.tfrecord')
    tf_writer = tf.io.TFRecordWriter(filename)
    doc_counts = 0
    with open(PROCESSED_DATA_PATH, 'r', encoding='utf-8') as f:
        for line in tqdm.tqdm(f):
            encoded_id = s.EncodeAsIds(line)
            if max_seq_len > len(encoded_id) > min_seq_len:
                inputs = np.array([BOS_ID] + encoded_id)
                targets = np.array(encoded_id + [EOS_ID])

                example = serialize_example(inputs, targets)
                tf_writer.write(example)
                doc_counts += 1
            if doc_counts >= per_file_limit:
                tf_writer.write(example)
                doc_counts = 0
                tf_writer.close()
                filename = os.path.join(
                    TF_RECORDS_PATH,
                    str(datetime.datetime.now().timestamp()) + '.tfrecord')
                tf_writer = tf.io.TFRecordWriter(filename)


@click.command()
@click.option('--data-dir', type=str, default=DATA_PATH, show_default=True, help='training data path')
@click.option('--vocab-size', type=int, default=24512, show_default=True, help='byte pair vocab size')
@click.option('--min-seq-len', type=int, default=15, show_default=True, help='minimum sequence length')
@click.option('--max-seq-len', type=int, default=512, show_default=True, help='maximum sequence length')
def train(data_dir, vocab_size, min_seq_len, max_seq_len):
    parse_zip_jsonl_files(data_dir)
    train_byte_pair_encoding(vocab_size)
    create_tf_records(min_seq_len, max_seq_len)
    print('Pre-processing is done............')


if __name__ == '__main__':
    train()
