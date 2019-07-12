#!/bin/bash

export BERT_BASE_DIR="/home/gzhenfun/bert/uncased_L-12_H-768_A-12"

python create_pretraining_data.py \
  --input_file=/home/gzhenfun/raw_tweets/$1 \
  --output_file=/home/gzhenfun/raw_tweets/tfrecords_uncased_wwm/$1.tfrecord \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --do_whole_word_mask=True \
  --do_lower_case=True \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5

exit
