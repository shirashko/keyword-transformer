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

"""Models parameters (with toy values, for testing)."""

from absl import logging


class Params(object):
  """Default parameters for data and feature settings.

     These parameters are compatible with command line flags
     and discribed in /train/base_parser.py
  """

  def __init__(self):
    # default parameters
    self.start_step = 0
    self.data_url = ''
    self.train_dir = ''
    self.wanted_words = 'yes,no,up,down,left,right,on,off,stop,go'
    self.train = 0
    self.split_data = 1
    self.sample_rate = 16000
    self.clip_duration_ms = 1000
    self.window_size_ms = 40.0
    self.window_stride_ms = 20.0
    self.preprocess = 'raw'
    self.feature_type = 'mfcc_tf'
    self.preemph = 0.0
    self.window_type = 'hann'
    self.mel_num_bins = 40
    self.mel_lower_edge_hertz = 20.0
    self.mel_upper_edge_hertz = 7000.0
    self.log_epsilon = 1e-12
    self.dct_num_features = 20
    self.use_tf_fft = 0
    self.mel_non_zero_only = 1
    self.fft_magnitude_squared = False
    self.use_spec_augment = 0
    self.time_masks_number = 2
    self.time_mask_max_size = 10
    self.frequency_masks_number = 2
    self.frequency_mask_max_size = 5
    self.use_spec_cutout = 0
    self.spec_cutout_masks_number = 3
    self.spec_cutout_time_mask_size = 10
    self.spec_cutout_frequency_mask_size = 5
    self.optimizer = 'adam'
    self.lr_schedule = 'linear'
    self.background_volume = 0.1
    self.l2_weight_decay = 0.0
    self.background_frequency = 0.8
    self.silence_percentage = 10.0
    self.unknown_percentage = 10.0
    self.time_shift_ms = 100.0
    self.testing_percentage = 10
    self.validation_percentage = 10
    self.how_many_training_steps = '10000,10000,10000'
    self.eval_step_interval = 400
    self.learning_rate = '0.0005,0.0001,0.00002'
    self.batch_size = 100
    self.optimizer_epsilon = 1e-08
    self.resample = 0.15
    self.volume_resample = 0.0
    self.return_softmax = 0
    self.sp_time_shift_ms = 0.0
    self.sp_resample = 0.0
    self.pick_deterministically = 0
    self.verbosity = logging.INFO
    self.causal_data_frame_padding = 0


def att_mh_rnn_params():
  """Parameters for toy multihead attention model."""
  params = Params()
  params.model_name = 'att_rnn'
  params.cnn_filters = '3,1'
  params.cnn_kernel_size = '(3,1),(3,1)'
  params.cnn_act = "'relu','relu'"
  params.cnn_dilation_rate = '(1,1),(1,1)'
  params.cnn_strides = '(1,1),(1,1)'
  params.rnn_layers = 2
  params.rnn_type = 'gru'
  params.rnn_units = 2
  params.heads = 4
  params.dropout1 = 0.1
  params.units2 = '2,2'
  params.act2 = "'relu','linear'"
  return params


# these are toy hotword model parameters
# with reduced dims for unit test only
HOTWORD_MODEL_PARAMS = {
    'att_mh_rnn': att_mh_rnn_params(),
}
