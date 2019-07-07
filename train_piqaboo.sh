#!/bin/bash

SQUAD_DIR=/home/becxer/data/squad
BERT_DIR=/home/becxer/model/uncased_L-12_H-768_A-12
OUTPUT_DIR=/home/becxer/model/piqaboo

python run_piqaboo.py \
  --vocab_file=$BERT_DIR/vocab.txt \
  --bert_config_file=$BERT_DIR/bert_config.json \
  --init_checkpoint=$BERT_DIR/bert_model.ckpt \
  --input_type=train \
  --do_train=True \
  --train_file=$SQUAD_DIR/dev-v1.1.json \
  --do_predict=False \
  --predict_file=$SQUAD_DIR/dev-v1.1.json \
  --train_batch_size=6 \
  --predict_batch_size=6 \
  --learning_rate=3e-5 \
  --num_train_epochs=3.0 \
  --max_phrase_context_seq_length=384 \
  --max_question_seq_length=64 \
  --output_dir=$OUTPUT_DIR/ \
  --debug=True
