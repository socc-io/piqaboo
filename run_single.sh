#!/bin/bash

SQUAD_DIR=/home/becxer/data/squad
BERT_DIR=/home/becxer/models/uncased_L-12_H-768_A-12
OUTPUT_DIR=./output

python run_squad.py \
  --vocab_file=$BERT_DIR/vocab.txt \
  --bert_config_file=$BERT_DIR/bert_config.json \
  --init_checkpoint=$BERT_DIR/bert_model.ckpt \
  --do_train=True \
  --train_file=$SQUAD_DIR/train-v1.1.json \
  --do_predict=True \
  --predict_file=$SQUAD_DIR/dev-v1.1.json \
  --train_batch_size=6 \
  --predict_batch_size=6 \
  --learning_rate=3e-5 \
  --num_train_epochs=3.0 \
  --max_seq_length=384 \
  --doc_stride=128 \
  --output_dir=$OUTPUT_DIR/ \
  --debug=True
