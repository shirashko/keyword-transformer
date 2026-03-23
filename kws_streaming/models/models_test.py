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

"""Tests for models."""
import numpy as np
from kws_streaming.layers.compat import tf
from kws_streaming.layers.compat import tf1
from kws_streaming.layers.modes import Modes
from kws_streaming.models import model_flags
from kws_streaming.models import model_params
from kws_streaming.models import models
tf1.disable_eager_execution()


class AttMhRnnTest(tf.test.TestCase):
  """Test att_mh_rnn model can be built with toy params."""

  def setUp(self):
    super(AttMhRnnTest, self).setUp()
    config = tf1.ConfigProto()
    config.gpu_options.allow_growth = True
    self.sess = tf1.Session(config=config)
    tf1.keras.backend.set_session(self.sess)
    tf.keras.backend.set_learning_phase(0)

    model_name = 'att_mh_rnn'
    self.params = model_params.HOTWORD_MODEL_PARAMS[model_name]
    self.params = model_flags.update_flags(self.params)
    self.params.batch_size = 1

  def test_att_mh_rnn_build(self):
    model = models.MODELS[self.params.model_name](self.params)
    model.summary()
    input_data = np.random.rand(self.params.batch_size,
                                self.params.desired_samples)
    output = model.predict(input_data)
    self.assertEqual(output.shape[0], self.params.batch_size)


if __name__ == '__main__':
  tf.test.main()
