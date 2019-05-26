# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run BERT on SQuAD 1.1 and SQuAD 2.0."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import math
import os
import random
import modeling
import optimization
import tokenization
import six
import tensorflow as tf
import numpy as np
from tqdm import tqdm

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string("train_file", None,
                    "SQuAD json for training. E.g., train-v1.1.json")

flags.DEFINE_string(
    "predict_file", None,
    "SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "doc_stride", 128,
    "When splitting up a long document into chunks, how much stride to "
    "take between chunks.")

flags.DEFINE_integer(
    "max_phrase_context_seq_length", 384,
    "The maximum number of tokens for the question. Questions longer than "
    "this will be truncated to this length.")

flags.DEFINE_integer(
    "max_question_seq_length", 64,
    "The maximum number of tokens for the question. Questions longer than "
    "this will be truncated to this length.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_predict", False, "Whether to run eval on the dev set.")

flags.DEFINE_string("input_type", "train", "set this to train | context | question")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("predict_batch_size", 8,
                     "Total batch size for predictions.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer("keep_checkpoint_max", 5,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer(
    "n_best_size", 20,
    "The total number of n-best predictions to generate in the "
    "nbest_predictions.json output file.")

flags.DEFINE_integer(
    "max_answer_length", 6,
    "The maximum length of an answer that can be generated. This is needed "
    "because the start and end predictions are not conditioned on one another.")


flags.DEFINE_float(
    "train_false_case_ratio", 0.5,
    "")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_bool(
    "verbose_logging", False,
    "If true, all of the warnings related to data processing will be printed. "
    "A number of warnings are expected for a normal SQuAD evaluation.")

flags.DEFINE_bool(
    "version_2_with_negative", False,
    "If true, the SQuAD examples contain some that do not have an answer.")

flags.DEFINE_bool(
    "debug", False,
    "Debugging mode")

flags.DEFINE_float(
    "null_score_diff_threshold", 0.0,
    "If null_score - best_non_null is greater than the threshold predict null.")


class SquadExample(object):
  """A single training/test example for simple sequence classification.

     For examples without an answer, the start and end position are -1.
  """

  def __init__(self,
               qas_id,
               article_title,
               para_idx,
               question_text,
               doc_tokens,
               orig_answer_text=None,
               start_position=None,
               end_position=None,
               is_impossible=False):
    self.qas_id = qas_id
    self.article_title = article_title
    self.para_idx = para_idx
    self.question_text = question_text
    self.doc_tokens = doc_tokens
    self.orig_answer_text = orig_answer_text
    self.start_position = start_position
    self.end_position = end_position
    self.is_impossible = is_impossible

  def __str__(self):
    return self.__repr__()

  def __repr__(self):
    s = ""
    s += "qas_id: %s" % (tokenization.printable_text(self.qas_id))
    s += ", question_text: %s" % (
        tokenization.printable_text(self.question_text))
    s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
    if self.start_position:
      s += ", start_position: %d" % (self.start_position)
    if self.start_position:
      s += ", end_position: %d" % (self.end_position)
    if self.start_position:
      s += ", is_impossible: %r" % (self.is_impossible)
    return s


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               unique_id,
               article_title,
               para_idx,
               qid,
               phrase_text,
               example_index,
               doc_span_index,
               tokens,
               token_is_max_context,
               phrase_context_input_ids,
               phrase_context_input_mask,
               phrase_context_segment_ids,
               question_input_ids,
               question_input_mask,
               question_segment_ids,
               label_sim):
    self.unique_id = unique_id
    self.article_title = article_title
    self.para_idx = para_idx
    self.qid = qid
    self.phrase_text = phrase_text
    self.example_index = example_index
    self.doc_span_index = doc_span_index
    self.tokens = tokens
    self.token_is_max_context = token_is_max_context
    self.phrase_context_input_ids = phrase_context_input_ids
    self.phrase_context_input_mask = phrase_context_input_mask
    self.phrase_context_segment_ids = phrase_context_segment_ids
    self.question_input_ids = question_input_ids
    self.question_input_mask = question_input_mask
    self.question_segment_ids = question_segment_ids
    self.label_sim = label_sim


def read_squad_examples(input_file, is_training):
  """Read a SQuAD json file into a list of SquadExample."""
  with tf.gfile.Open(input_file, "r") as reader:
    input_data = json.load(reader)["data"]

  def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
      return True
    return False

  examples = []
  for entry in input_data:
    article_title = entry["title"]
    for idx, paragraph in enumerate(entry["paragraphs"]):
      paragraph_text = paragraph["context"]
      para_idx = idx
      doc_tokens = []
      char_to_word_offset = []
      prev_is_whitespace = True
      for c in paragraph_text:
        if is_whitespace(c):
          prev_is_whitespace = True
        else:
          if prev_is_whitespace:
            doc_tokens.append(c)
          else:
            doc_tokens[-1] += c
          prev_is_whitespace = False
        char_to_word_offset.append(len(doc_tokens) - 1)

      for qa in paragraph["qas"]:
        qas_id = qa["id"]
        question_text = qa["question"]
        start_position = None
        end_position = None
        orig_answer_text = None
        is_impossible = False
        if is_training:

          if FLAGS.version_2_with_negative:
            is_impossible = qa["is_impossible"]
          if not is_impossible:
            answer = qa["answers"][0]
            orig_answer_text = answer["text"]
            answer_offset = answer["answer_start"]
            answer_length = len(orig_answer_text)
            start_position = char_to_word_offset[answer_offset]
            end_position = char_to_word_offset[answer_offset + answer_length -
                                               1]
            # Only add answers where the text can be exactly recovered from the
            # document. If this CAN'T happen it's likely due to weird Unicode
            # stuff so we will just skip the example.
            #
            # Note that this means for training mode, every example is NOT
            # guaranteed to be preserved.
            actual_text = " ".join(
                doc_tokens[start_position:(end_position + 1)])
            cleaned_answer_text = " ".join(
                tokenization.whitespace_tokenize(orig_answer_text))
            if actual_text.find(cleaned_answer_text) == -1:
              tf.logging.warning("Could not find answer: '%s' vs. '%s'",
                                 actual_text, cleaned_answer_text)
              continue
          else:
            start_position = -1
            end_position = -1
            orig_answer_text = ""

        example = SquadExample(
            qas_id=qas_id,
            article_title=article_title,
            para_idx= idx,
            question_text=question_text,
            doc_tokens=doc_tokens,
            orig_answer_text=orig_answer_text,
            start_position=start_position,
            end_position=end_position,
            is_impossible=is_impossible)
        examples.append(example)

  return examples


def convert_examples_to_features(examples, tokenizer, max_doc_phrase_input_length,
                                 doc_stride, max_question_input_length, is_training,
                                 output_fn):
  """Loads a data file into a list of `InputBatch`s."""

  unique_id = 1000000000

  for (example_index, example) in enumerate(tqdm(examples)):
    article_title = example.article_title
    para_idx = example.para_idx
    qid = example.qas_id
    # question 인풋
    question_tokens = ["[CLS]"]
    question_tokens.extend(tokenizer.tokenize(example.question_text))
    question_tokens.append("[SEP]")
    if len(question_tokens) > max_question_input_length:
      question_tokens = question_tokens[0:max_question_input_length]

    question_input_ids = tokenizer.convert_tokens_to_ids(question_tokens)
    question_input_mask = [1] * len(question_input_ids)
    while len(question_input_ids) < max_question_input_length:
        question_input_ids.append(0)
        question_input_mask.append(0)

    question_segment_ids = [0] * max_question_input_length
    # phrase + doc 인풋
    for i, token in enumerate(example.doc_tokens) :
        # 문서의 모든 토큰에 대해 케이스를 만듬.
        for window_size in range(FLAGS.max_answer_length):
            # 현재 토큰에 대해서 윈도우 사이즈별로 케이스를 만든다. 정답인 경우에만 label_sim=1 이다
            start_idx = i
            end_idx = i + window_size
            label_sim = 0.0
            answer_tokens = tokenization.whitespace_tokenize(example.orig_answer_text)
            if example.doc_tokens[start_idx:(end_idx+1)] == answer_tokens:
                label_sim = 1.0

            phrase_text = " ".join(example.doc_tokens[start_idx:(end_idx+1)])
            phrase_tokens = tokenizer.tokenize(" ".join(example.doc_tokens[start_idx:(end_idx+1)]));
            all_doc_tokens = tokenizer.tokenize(" ".join(example.doc_tokens))

            phrase_context_tokens = []
            phrase_context_segment_ids = []
            phrase_context_tokens.append("[CLS]")
            phrase_context_segment_ids.append(0)
            for phrase_token in phrase_tokens:
                phrase_context_tokens.append(phrase_token)
                phrase_context_segment_ids.append(0)

            phrase_context_tokens.append("[SEP]")
            phrase_context_segment_ids.append(0)

            # The -3 accounts for [CLS], [SEP] and [SEP]
            max_tokens_for_doc = max_doc_phrase_input_length - len(phrase_tokens) - 3
            phrase_context_tokens.extend(all_doc_tokens[0:max_tokens_for_doc])
            phrase_context_segment_ids.extend([1]*max_tokens_for_doc)

            phrase_context_tokens.append("[SEP]")
            phrase_context_segment_ids.append(1)

            phrase_context_input_ids = tokenizer.convert_tokens_to_ids(phrase_context_tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            phrase_context_input_mask = [1] * len(phrase_context_input_ids)

            # Zero-pad up to the sequence length.
            while len(phrase_context_input_ids) < max_doc_phrase_input_length:
                phrase_context_input_ids.append(0)
                phrase_context_input_mask.append(0)
                phrase_context_segment_ids.append(0)

            assert len(phrase_context_input_ids) == max_doc_phrase_input_length
            assert len(phrase_context_input_mask) == max_doc_phrase_input_length
            assert len(phrase_context_segment_ids) == max_doc_phrase_input_length


            if example_index < 3 and label_sim==1.0 and FLAGS.debug:
                tf.logging.info("*** Example ***")
                tf.logging.info("unique_id: %s" % (unique_id))
                tf.logging.info("example_index: %s" % (example_index))
                tf.logging.info("question_tokens: %s" % " ".join(
                    [tokenization.printable_text(x) for x in question_tokens]))
                tf.logging.info("question_input_ids: %s" % " ".join([str(x) for x in question_input_ids]))
                tf.logging.info(
                    "question_input_mask: %s" % " ".join([str(x) for x in question_input_mask]))
                tf.logging.info(
                    "question_segment_ids: %s" % " ".join([str(x) for x in question_segment_ids]))
                tf.logging.info("doc_span_index: %s" % (doc_span_index))
                tf.logging.info("phrase_context_tokens: %s" % " ".join(
                    [tokenization.printable_text(x) for x in phrase_context_tokens]))
                tf.logging.info("token_is_max_context: %s" % " ".join([
                    "%d:%s" % (x, y) for (x, y) in six.iteritems(token_is_max_context)
                ]))
                tf.logging.info("phrase_context_input_ids: %s" % " ".join([str(x) for x in phrase_context_input_ids]))
                tf.logging.info(
                    "phrase_context_input_mask: %s" % " ".join([str(x) for x in phrase_context_input_mask]))
                tf.logging.info(
                    "phrase_context_segment_ids: %s" % " ".join([str(x) for x in phrase_context_segment_ids]))
                if is_training and example.is_impossible:
                  tf.logging.info("impossible example")

            feature = InputFeatures(
              unique_id=unique_id,
              example_index=example_index,
              tokens=phrase_context_tokens,
              phrase_context_input_ids=phrase_context_input_ids,
              phrase_context_input_mask=phrase_context_input_mask,
              phrase_context_segment_ids=phrase_context_segment_ids,
              question_input_ids=question_input_ids,
              question_input_mask=question_input_mask,
              question_segment_ids=question_segment_ids,
              label_sim=label_sim
            )

            # Run callback
            output_fn(feature)

            unique_id += 1


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
  """Returns tokenized answer spans that better match the annotated answer."""

  # The SQuAD annotations are character based. We first project them to
  # whitespace-tokenized words. But then after WordPiece tokenization, we can
  # often find a "better match". For example:
  #
  #   Question: What year was John Smith born?
  #   Context: The leader was John Smith (1895-1943).
  #   Answer: 1895
  #
  # The original whitespace-tokenized answer will be "(1895-1943).". However
  # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
  # the exact answer, 1895.
  #
  # However, this is not always possible. Consider the following:
  #
  #   Question: What country is the top exporter of electornics?
  #   Context: The Japanese electronics industry is the lagest in the world.
  #   Answer: Japan
  #
  # In this case, the annotator chose "Japan" as a character sub-span of
  # the word "Japanese". Since our WordPiece tokenizer does not split
  # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
  # in SQuAD, but does happen.
  tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

  for new_start in range(input_start, input_end + 1):
    for new_end in range(input_end, new_start - 1, -1):
      text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
      if text_span == tok_answer_text:
        return (new_start, new_end)

  return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
  """Check if this is the 'max context' doc span for the token."""

  # Because of the sliding window approach taken to scoring documents, a single
  # token can appear in multiple documents. E.g.
  #  Doc: the man went to the store and bought a gallon of milk
  #  Span A: the man went to the
  #  Span B: to the store and bought
  #  Span C: and bought a gallon of
  #  ...
  #
  # Now the word 'bought' will have two scores from spans B and C. We only
  # want to consider the score with "maximum context", which we define as
  # the *minimum* of its left and right context (the *sum* of left and
  # right context will always be the same, of course).
  #
  # In the example the maximum context for 'bought' would be span C since
  # it has 1 left context and 3 right context, while span B has 4 left context
  # and 0 right context.
  best_score = None
  best_span_index = None
  for (span_index, doc_span) in enumerate(doc_spans):
    end = doc_span.start + doc_span.length - 1
    if position < doc_span.start:
      continue
    if position > end:
      continue
    num_left_context = position - doc_span.start
    num_right_context = end - position
    score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
    if best_score is None or score > best_score:
      best_score = score
      best_span_index = span_index

  return cur_span_index == best_span_index


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 use_one_hot_embeddings, scope):
  """Creates a classification model."""

  with tf.variable_scope('bert', reuse=tf.AUTO_REUSE) as real_scope:
    model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings,
      scope=scope
    )

  final_hidden = model.get_sequence_output()
  final_hidden_shape = modeling.get_shape_list(final_hidden, expected_rank=3)
  return final_hidden[:,0,:]

def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    unique_ids = features["unique_ids"]

    if FLAGS.input_type == "context" or FLAGS.input_type == "train":
      phrase_context_input_ids = features["phrase_context_input_ids"]
      phrase_context_input_mask = features["phrase_context_input_mask"]
      phrase_context_segment_ids = features["phrase_context_segment_ids"]

    if FLAGS.input_type == "question" or FLAGS.input_type == "train":
      question_input_ids = features["question_input_ids"]
      question_input_mask = features["question_input_mask"]
      question_segment_ids = features["question_segment_ids"]
    
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    if is_training:
      label_sim = features["label_sim"]

    if FLAGS.input_type == "context" or FLAGS.input_type == "train":
      with tf.variable_scope('context', reuse=tf.AUTO_REUSE) as scope:
        phrase_embedding = create_model(
              bert_config=bert_config,
              is_training=is_training,
              input_ids=phrase_context_input_ids,
              input_mask=phrase_context_input_mask,
              segment_ids=phrase_context_segment_ids,
              use_one_hot_embeddings=use_one_hot_embeddings,
              scope=scope)
        norm_pe = tf.nn.l2_normalize(phrase_embedding)

    if FLAGS.input_type == "question" or FLAGS.input_type == "train":
      with tf.variable_scope('question', reuse=tf.AUTO_REUSE) as scope:
        question_embedding = create_model(
              bert_config=bert_config,
              is_training=is_training,
              input_ids=question_input_ids,
              input_mask=question_input_mask,
              segment_ids=question_segment_ids,
              use_one_hot_embeddings=use_one_hot_embeddings,
              scope=scope)
      norm_qe = tf.nn.l2_normalize(question_embedding)

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, init_var) \
        = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint, "context/")
      initialized_variable_names.update(init_var)
      tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

      (assignment_map, init_var) \
        = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint, "question/")
      initialized_variable_names.update(init_var)
      tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:
      similarity = tf.reduce_sum(norm_pe * norm_qe, axis=1)
      similarity_scaled = (similarity - 0.5) * 32
      loss_sim = tf.nn.sigmoid_cross_entropy_with_logits(logits=similarity_scaled, labels=label_sim)
      total_loss = tf.reduce_sum(loss_sim)

      global_step = tf.train.get_or_create_global_step()
      optimizer = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, global_step)
      optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)
      train_op = optimizer.minimize(total_loss, global_step=global_step)

      output_spec = tf.estimator.EstimatorSpec(mode=mode, loss=total_loss, train_op=train_op)

    elif mode == tf.estimator.ModeKeys.PREDICT:
      predictions = {"unique_ids": unique_ids}
      if FLAGS.input_type == "context" or FLAGS.input_type == "train":
        predictions["phrase_embeddings"] =  phrase_embedding
      if FLAGS.input_type == "question" or FLAGS.input_type == "train":
        predictions["question_embeddings"] = question_embedding
      output_spec = tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    else:
      raise ValueError(
          "Only TRAIN and PREDICT modes are supported: %s" % (mode))

    return output_spec

  return model_fn


def input_fn_builder(input_file, phrase_context_seq_length, question_seq_length, is_training, drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "unique_ids": tf.FixedLenFeature([], tf.int64)  
  }

  if FLAGS.input_type == "context" or FLAGS.input_type == "train":
    name_to_features["phrase_context_input_ids"] = tf.FixedLenFeature([phrase_context_seq_length], tf.int64)
    name_to_features["phrase_context_input_mask"] = tf.FixedLenFeature([phrase_context_seq_length], tf.int64)
    name_to_features["phrase_context_segment_ids"] = tf.FixedLenFeature([phrase_context_seq_length], tf.int64)

  if FLAGS.input_type == "question" or FLAGS.input_type == "train":
    name_to_features["question_input_ids"] = tf.FixedLenFeature([question_seq_length], tf.int64)
    name_to_features["question_input_mask"] = tf.FixedLenFeature([question_seq_length], tf.int64)
    name_to_features["question_segment_ids"] = tf.FixedLenFeature([question_seq_length], tf.int64)  

  if is_training:
    name_to_features["label_sim"] = tf.FixedLenFeature([], tf.float32)

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""
    batch_size = FLAGS.train_batch_size if is_training else FLAGS.predict_batch_size

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn


RawResult = collections.namedtuple("RawResult",
            ["unique_id", "phrase_embedding", "question_embedding"])


def write_predictions_piqa(all_features, all_results, output_context_dir, output_question_dir):
  """Output format for PIQA
  each .npz in context_emb is '%s_%d'.npz % (article_title, para_idx)
  each .npz in context_emb is N phrase vectors of d-dim: N x d
  each .json is a list of N phrases
  each.npz in question_emb is '%s'.npz % (question_id)
  each.npz in question_emb must be 1 x d matrix

  * feature.unique_id -> (article_title, para_idx, phrase_idx, qid)

  * all_context_embedding = {
   "title": { # article
     "12" : [ # para
       {
         "embedding" : [1,1,1,1,1,1],
         "phrase-text" : "phrase text"
       }
     ]
   }}

  * all_question_embedding = {
   "qid": [1,1,1,1,1]
  }
  """
  unique_id_to_result = {}
  for result in all_results:
    unique_id_to_result[result.unique_id] = result

  all_context_embedding = {}
  all_question_embedding = {}

  for (feature_index, feature) in enumerate(all_features):
    if FLAGS.debug and (not feature.unique_id in unique_id_to_result.keys()):
      print(str(feature.unique_id) + " not found")
      continue
    result_entry = unique_id_to_result[feature.unique_id]
    if FLAGS.input_type == "context" or FLAGS.input_type == "train":
      article_title = feature.article_title
      para_idx = str(feature.para_idx)
      phrase_text = feature.phrase_text
      phrase_embedding = result_entry.phrase_embedding
      context_embedding_entry = all_context_embedding.get(article_title, {})
      paragraph_entry = context_embedding_entry.get(para_idx, [])
      paragraph_entry.append({
        "embedding" : list(phrase_embedding),
        "phrase-text" : phrase_text
      })
      context_embedding_entry[para_idx] = paragraph_entry
      all_context_embedding[article_title] = context_embedding_entry

    if FLAGS.input_type == "question" or FLAGS.input_type == "train":
      qid = feature.qid
      question_embedding = result_entry.question_embedding
      all_question_embedding[qid] = list(question_embedding)

  for title in all_context_embedding.keys():
    para_entry = all_context_embedding[title]
    for para_key in para_entry.keys():
      phrase_list = para_entry[para_key]
      phrase_embedding_list = []
      phrase_text_list = []
      for phrase in phrase_list:
        phrase_embedding_list.append(phrase["embedding"])
        phrase_text_list.append(phrase["phrase-text"])
      phrase_embedding_np = np.array(phrase_embedding_list)

      filename_np = os.path.join(output_context_dir, "%s_%s" % (title, para_key))
      np.savez(filename_np, phrase_embedding_np)

      filename_json = os.path.join(output_context_dir, "%s_%s" % (title, para_key))
      with open(filename_json, "w") as fp:
        fp.write(json.dumps(phrase_text_list))

  for qid in all_question_embedding.keys():
    print(qid)
    question_embedding_np = np.array(all_question_embedding[qid])
    filename_np = os.path.join(output_question_dir, "%s" % (qid))
    np.savez(filename_np, question_embedding_np)

  tf.logging.info("***** Write prediction compelete *****")

class FeatureWriter(object):
  """Writes InputFeature to TF example file."""

  def __init__(self, filename, is_training):
    self.filename = filename
    self.is_training = is_training
    self.num_features = 0
    self._writer = tf.python_io.TFRecordWriter(filename)

  def process_feature(self, feature):
    """Write a InputFeature to the TFRecordWriter as a tf.train.Example."""
    self.num_features += 1

    def create_int_feature(values):
      feature = tf.train.Feature(
          int64_list=tf.train.Int64List(value=list(values)))
      return feature

    def create_float_feature(values):
      feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
      return feature

    features = collections.OrderedDict()
    features["unique_ids"] = create_int_feature([feature.unique_id])

    if FLAGS.input_type == "context" or FLAGS.input_type == "train":
      features["phrase_context_input_ids"] = create_int_feature(feature.phrase_context_input_ids)
      features["phrase_context_input_mask"] = create_int_feature(feature.phrase_context_input_mask)
      features["phrase_context_segment_ids"] = create_int_feature(feature.phrase_context_segment_ids)

    if FLAGS.input_type == "question" or FLAGS.input_type == "train":
      features["question_input_ids"] = create_int_feature(feature.question_input_ids)
      features["question_input_mask"] = create_int_feature(feature.question_input_mask)
      features["question_segment_ids"] = create_int_feature(feature.question_segment_ids)

    if self.is_training:
      features["label_sim"] = create_float_feature([feature.label_sim])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    self._writer.write(tf_example.SerializeToString())

  def close(self):
    self._writer.close()


def validate_flags_or_throw(bert_config):
  """Validate the input FLAGS or throw an exception."""
  tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                FLAGS.init_checkpoint)

  if not FLAGS.do_train and not FLAGS.do_predict:
    raise ValueError("At least one of `do_train` or `do_predict` must be True.")

  if FLAGS.do_train:
    if not FLAGS.train_file:
      raise ValueError(
          "If `do_train` is True, then `train_file` must be specified.")
  if FLAGS.do_predict:
    if not FLAGS.predict_file:
      raise ValueError(
          "If `do_predict` is True, then `predict_file` must be specified.")

  if FLAGS.max_phrase_context_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_phrase_context_seq_length, bert_config.max_position_embeddings))

  if FLAGS.max_phrase_context_seq_length <= FLAGS.max_question_seq_length + 3:
    raise ValueError(
        "The max_phrase_context_seq_length (%d) must be greater than max_question_seq_length "
        "(%d) + 3" % (FLAGS.max_phrase_context_seq_length, FLAGS.max_question_seq_length))

  ngpu = len(os.getenv('CUDA_VISIBLE_DEVICES', '0').split(','))

  if FLAGS.predict_batch_size % ngpu != 0 or FLAGS.train_batch_size % ngpu != 0:
    raise ValueError(
        "Batch size should be able to divide with number of gpu")


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  validate_flags_or_throw(bert_config)

  tf.gfile.MakeDirs(FLAGS.output_dir)

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  run_config = tf.estimator.RunConfig(
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      keep_checkpoint_max=FLAGS.keep_checkpoint_max)

  train_examples = None
  num_train_steps = None
  num_warmup_steps = None
  if FLAGS.do_train:
    train_examples = read_squad_examples(
        input_file=FLAGS.train_file, is_training=True)
    num_train_steps = int(len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    # Pre-shuffle the input to avoid having to make a very large shuffle
    # buffer in in the `input_fn`.
    rng = random.Random(12345)
    rng.shuffle(train_examples)

  model_fn = model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.estimator.Estimator(
      model_fn=tf.contrib.estimator.replicate_model_fn(
          model_fn, loss_reduction=tf.losses.Reduction.MEAN),
      config=run_config)

  if FLAGS.do_train:
    # We write to a temporary file to avoid storing very large constant tensors
    # in memory.
    train_writer = FeatureWriter(
        filename=os.path.join(FLAGS.output_dir, "train.tf_record"),
        is_training=True)
    convert_examples_to_features(
        examples=train_examples,
        tokenizer=tokenizer,
        max_doc_phrase_input_length=FLAGS.max_phrase_context_seq_length,
        doc_stride=FLAGS.doc_stride,
        max_question_input_length=FLAGS.max_question_seq_length,
        is_training=True,
        output_fn=train_writer.process_feature)
    train_writer.close()

    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num orig examples = %d", len(train_examples))
    tf.logging.info("  Num split examples = %d", train_writer.num_features)
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)
    del train_examples

    train_input_fn = input_fn_builder(
        input_file=train_writer.filename,
        phrase_context_seq_length = FLAGS.max_phrase_context_seq_length,
        question_seq_length = FLAGS.max_question_seq_length,
        is_training=True,
        drop_remainder=True)
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

  if FLAGS.do_predict:
    eval_examples = read_squad_examples(
        input_file=FLAGS.predict_file, is_training=False)

    eval_writer = FeatureWriter(
        filename=os.path.join(FLAGS.output_dir, "eval.tf_record"),
        is_training=False)
    eval_features = []

    def append_feature(feature):
      eval_features.append(feature)
      eval_writer.process_feature(feature)

    convert_examples_to_features(
        examples=eval_examples,
        tokenizer=tokenizer,
        max_doc_phrase_input_length=FLAGS.max_phrase_context_seq_length,
        doc_stride=FLAGS.doc_stride,
        max_question_input_length=FLAGS.max_question_seq_length,
        is_training=False,
        output_fn=append_feature)

    if len(eval_features) % FLAGS.predict_batch_size != 0:
        for i in range(len(eval_features) % FLAGS.predict_batch_size):
            dummy = eval_features[-1]
            dummy.unique_id = dummy.unique_id + 1
            append_feature(dummy)

    eval_writer.close()
    tf.logging.info("***** Running predictions *****")
    tf.logging.info("  Num orig examples = %d", len(eval_examples))
    tf.logging.info("  Num split examples = %d", len(eval_features))
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

    predict_input_fn = input_fn_builder(
        input_file=eval_writer.filename,
        phrase_context_seq_length=FLAGS.max_phrase_context_seq_length,
        question_seq_length=FLAGS.max_question_seq_length,
        is_training=False,
        drop_remainder=False)

    # If running eval on the TPU, you will need to specify the number of
    # steps.
    all_results = []
    for result in estimator.predict(
        predict_input_fn, yield_single_examples=True):
      if len(all_results) % 1000 == 0:
        tf.logging.info("Processing example: %d" % (len(all_results)))
        if FLAGS.debug and len(all_results) >= 2000:
          break
        
      unique_id = int(result["unique_ids"])
      phrase_embedding = None
      question_embedding = None
      if FLAGS.input_type == "context" or FLAGS.input_type == "train":
          phrase_embedding = result["phrase_embeddings"]
      if FLAGS.input_type == "question" or FLAGS.input_type == "train":
          question_embedding = result["question_embeddings"]
      all_results.append(
          RawResult(
              unique_id=unique_id,
              phrase_embedding=phrase_embedding,
              question_embedding=question_embedding))

    output_context_dir = os.path.join(FLAGS.output_dir, "context_emb")
    output_question_dir = os.path.join(FLAGS.output_dir, "question_emb")
    tf.gfile.MakeDirs(output_context_dir)
    tf.gfile.MakeDirs(output_question_dir)

    write_predictions_piqa(eval_features, all_results, output_context_dir, output_question_dir)

if __name__ == "__main__":
  print("Current options : ", tf.app.flags.FLAGS.flag_values_dict())
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()
