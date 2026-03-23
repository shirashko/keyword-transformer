# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Simple speech recognition to spot a limited number of keywords.

It is based on tensorflow/examples/speech_commands
This is a self-contained example script that will train a very basic audio
recognition model in TensorFlow. It downloads the necessary training data and
runs with reasonable defaults to train within a few hours even only using a CPU.

It is intended as an introduction to using neural networks for audio
recognition, and is not a full speech recognition system. This network uses a
keyword detection style to spot discrete words from a small vocabulary,
consisting of
"yes", "no", "up", "down", "left", "right", "on", "off", "stop", and "go".

To run the training process, use:

bazel run model_train_eval.py

This will write out checkpoints to /tmp/speech_commands_train/, and will
download over 1GB of open source training data, so you'll need enough free space
and a good internet connection. The default data is a collection of thousands of
one-second .wav files, each containing one spoken word. This data set is
collected from https://aiyprojects.withgoogle.com/open_speech_recording, please
consider contributing to help improve this and other models!

As training progresses, it will print out its accuracy metrics, which should
rise above 90% by the end. Once it's complete, it will produce
Keras, SavedModel, TFLite and graphdef representations.

If you want to train on your own data, you'll need to create .wavs with your
recordings, all at a consistent length, and then arrange them into subfolders
organized by label. For example, here's a possible file structure:

data >
  up >
    audio_0.wav
    audio_1.wav
  down >
    audio_2.wav
    audio_3.wav
  other>
    audio_4.wav
    audio_5.wav

You'll also need to tell the script what labels to look for, using the
`--wanted_words` argument. In this case, 'up,down' might be what you want, and
the audio in the 'other' folder would be used to train an 'unknown' category.

To pull this all together, you'd run:

bazel run tensorflow/examples/speech_commands:train --
--data_dir /data --wanted_words up,down

Above script will automatically split data into training/validation and testing.

If you prefer to split the data on your own, then you should set flag
"--split_data 0" and prepare folders with structure:

data >
  training >
    up >
      audio_0.wav
      audio_1.wav
    down >
      audio_2.wav
      audio_3.wav
  validation >
    up >
      audio_6.wav
      audio_7.wav
    down >
      audio_8.wav
      audio_9.wav
  testing >
    up >
      audio_12.wav
      audio_13.wav
    down >
      audio_14.wav
      audio_15.wav
  _background_noise_ >
    audio_18.wav

To pull this all together, you'd run:

bazel run tensorflow/examples/speech_commands:train --
--data_dir /data --wanted_words up,down --split_data 0

"""
import json
import os
import sys
from absl import logging
import tensorflow.compat.v1 as tf
from kws_streaming.layers import modes
from kws_streaming.models import model_flags
from kws_streaming.models import utils
import kws_streaming.models.att_mh_rnn as att_mh_rnn
import kws_streaming.models.kws_transformer as kws_transformer
from kws_streaming.train import base_parser
from kws_streaming.train import train
import kws_streaming.train.test as test

FLAGS = None


def main(_):
  # Update flags
  flags = model_flags.update_flags(FLAGS)

  if flags.train:
    # Create model folders where logs and model will be stored
    os.makedirs(flags.train_dir, exist_ok=True)
    if not os.path.exists(flags.summaries_dir):
        os.makedirs(flags.summaries_dir, exist_ok=True)

    # Model training
    train.train(flags)
  else:
    if not os.path.isdir(flags.train_dir):
      raise ValueError('model is not trained set "--train 1" and retrain it')

  flags.training = False

  # write all flags settings into json
  with open(os.path.join(flags.train_dir, 'flags.json'), 'wt') as f:
    json.dump(flags.__dict__, f)

  # convert to SavedModel
  test.convert_model_saved(flags, 'non_stream',
                           modes.Modes.NON_STREAM_INFERENCE)
  try:
    test.convert_model_saved(flags, 'stream_state_internal',
                             modes.Modes.STREAM_INTERNAL_STATE_INFERENCE)
  except (ValueError, IndexError) as e:
    logging.info('FAILED to run TF streaming: %s', e)

  logging.info('run TF non streaming model accuracy evaluation')
  # with TF
  folder_name = 'tf'
  test.tf_non_stream_model_accuracy(flags, folder_name)

  # with TF.
  # We can apply non stream model on stream data, by running inference
  # every 200ms (for example), so that total latency will be similar with
  # streaming model which is executed every 20ms.
  # To measure the impact of sampling on model accuracy,
  # we introduce time_shift_ms during accuracy evaluation.
  # Convert milliseconds to samples:
  time_shift_samples = int(
      (flags.time_shift_ms * flags.sample_rate) / model_flags.MS_PER_SECOND)
  test.tf_non_stream_model_accuracy(
      flags,
      folder_name,
      time_shift_samples,
      accuracy_name='tf_non_stream_model_sampling_stream_accuracy.txt')

  name2opt = {
      '': None,
      'quantize_opt_for_size_': [tf.lite.Optimize.OPTIMIZE_FOR_SIZE],
  }

  for opt_name, optimizations in name2opt.items():

    if (opt_name and flags.feature_type == 'mfcc_tf' and
        flags.preprocess == 'raw'):
      logging.info('feature type mfcc_tf needs quantization aware training '
                   'for quantization - it is not implemented')
      continue

    folder_name = opt_name + 'tflite_non_stream'
    file_name = 'non_stream.tflite'
    mode = modes.Modes.NON_STREAM_INFERENCE
    test.convert_model_tflite(flags, folder_name, mode, file_name,
                              optimizations=optimizations)
    test.tflite_non_stream_model_accuracy(flags, folder_name, file_name)

    # these models are using bi-rnn, so they are non streamable by default
    # also models using striding or pooling are not supported for streaming now
    non_streamable_models = {'att_mh_rnn', 'kws_transformer'}

    model_is_streamable = True
    if flags.model_name in non_streamable_models:
      model_is_streamable = False

    # if model can be streamed, then run conversion/evaluation in streaming mode
    if model_is_streamable:
      # ---------------- TF streaming model accuracy evaluation ----------------
      # Streaming model with external state evaluation using TF with state reset
      if not opt_name:
        logging.info('run TF evalution only without optimization/quantization')
        try:
          folder_name = 'tf'
          test.tf_stream_state_external_model_accuracy(
              flags,
              folder_name,
              accuracy_name='stream_state_external_model_accuracy_sub_set_reset1.txt',
              reset_state=True)  # with state reset between test sequences

          # Streaming (with external state) evaluation using TF no state reset
          test.tf_stream_state_external_model_accuracy(
              flags,
              folder_name,
              accuracy_name='stream_state_external_model_accuracy_sub_set_reset0.txt',
              reset_state=False)  # without state reset

          # Streaming (with internal state) evaluation using TF no state reset
          test.tf_stream_state_internal_model_accuracy(flags, folder_name)
        except (ValueError, IndexError) as e:
          logging.info('FAILED to run TF streaming: %s', e)

      logging.info('run TFlite streaming model accuracy evaluation')
      try:
        # convert model to TFlite
        folder_name = opt_name + 'tflite_stream_state_external'
        file_name = 'stream_state_external.tflite'
        mode = modes.Modes.STREAM_EXTERNAL_STATE_INFERENCE
        test.convert_model_tflite(flags, folder_name, mode, file_name,
                                  optimizations=optimizations)

        # Streaming model accuracy evaluation with TFLite with state reset
        test.tflite_stream_state_external_model_accuracy(
            flags,
            folder_name,
            file_name,
            accuracy_name='tflite_stream_state_external_model_accuracy_reset1.txt',
            reset_state=True)

        # Streaming model accuracy evaluation with TFLite without state reset
        test.tflite_stream_state_external_model_accuracy(
            flags,
            folder_name,
            file_name,
            accuracy_name='tflite_stream_state_external_model_accuracy_reset0.txt',
            reset_state=False)
      except (ValueError, IndexError) as e:
        logging.info('FAILED to run TFLite streaming: %s', e)

if __name__ == '__main__':
  # parser for training/testing data and speach feature flags
  parser = base_parser.base_parser()

  # sub parser for model settings
  subparsers = parser.add_subparsers(dest='model_name', help='NN model name')

  # ATT MH RNN model settings (teacher for distillation)
  parser_att_mh_rnn = subparsers.add_parser('att_mh_rnn')
  att_mh_rnn.model_parameters(parser_att_mh_rnn)

  # kws_transformer settings
  parser_kws_transformer = subparsers.add_parser('kws_transformer')
  kws_transformer.model_parameters(parser_kws_transformer)

  FLAGS, unparsed = parser.parse_known_args()
  if unparsed and tuple(unparsed) != ('--alsologtostderr',):
    raise ValueError('Unknown argument: {}'.format(unparsed))
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
