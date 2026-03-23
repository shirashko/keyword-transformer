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

"""Tests for kws_streaming.models.utils."""

from absl.testing import parameterized
import numpy as np

from kws_streaming.layers import modes
from kws_streaming.layers.compat import tf
from kws_streaming.layers.compat import tf1
from kws_streaming.models import model_flags
from kws_streaming.models import model_params
from kws_streaming.models import models
from kws_streaming.models import utils
tf1.disable_eager_execution()


class SequentialModel(tf.keras.Model):
  """Dummy sequential model to test conversion to functional model."""

  def __init__(self,
               num_outputs,):
    """Initialize dummy model.

    Args:
      num_outputs: Number of outputs.
    """
    super().__init__()

    self._model = {}
    self._model['model1'] = tf.keras.Sequential(
        layers=[tf.keras.layers.Dense(units=num_outputs, activation=None)],
        name='model1')
    self._model['model2'] = tf.keras.Sequential(
        layers=[tf.keras.layers.GlobalMaxPooling2D()], name='model2')

  def call(self, inputs):
    net = inputs
    net = self._model['model1'](net)
    net = self._model['model2'](net)
    return net


# two models are tested with all cobinations of speech frontend
# and all models are tested with one frontend
class UtilsTest(tf.test.TestCase, parameterized.TestCase):

  def _testTFLite(self,
                  preprocess='raw',
                  feature_type='mfcc_op',
                  model_name='att_mh_rnn'):
    params = model_params.HOTWORD_MODEL_PARAMS[model_name]
    params.clip_duration_ms = 100  # make it shorter for testing

    # set parameters to test
    params.preprocess = preprocess
    params.feature_type = feature_type
    params = model_flags.update_flags(params)

    # create model
    model = models.MODELS[params.model_name](params)

    # convert TF non streaming model to TFLite non streaming inference
    self.assertTrue(
        utils.model_to_tflite(self.sess, model, params,
                              modes.Modes.NON_STREAM_INFERENCE))

  def setUp(self):
    super(UtilsTest, self).setUp()
    tf1.reset_default_graph()
    config = tf1.ConfigProto()
    config.gpu_options.allow_growth = True
    self.sess = tf1.Session(config=config)
    tf1.keras.backend.set_session(self.sess)

  @parameterized.named_parameters([
      {
          'testcase_name': 'raw with mfcc_tf',
          'preprocess': 'raw',
          'feature_type': 'mfcc_tf'
      },
      {
          'testcase_name': 'raw with mfcc_op',
          'preprocess': 'raw',
          'feature_type': 'mfcc_op'
      },
      {
          'testcase_name': 'mfcc',
          'preprocess': 'mfcc',
          'feature_type': 'mfcc_op'
      },  # feature_type will be ignored
      {
          'testcase_name': 'micro',
          'preprocess': 'micro',
          'feature_type': 'mfcc_op'
      },  # feature_type will be ignored
  ])
  def testPreprocessNonStreamInferenceTFandTFLite(self,
                                                  preprocess,
                                                  feature_type,
                                                  model_name='att_mh_rnn'):
    # Validate that model with different preprocessing
    # can be converted to non stream inference mode.
    self._testTFLite(preprocess, feature_type, model_name)

  def testNextPowerOfTwo(self):
    self.assertEqual(utils.next_power_of_two(11), 16)

  def testNonStreamInferenceTFandTFLite_att_mh_rnn(self):
    # Validate att_mh_rnn with selected preprocessing
    # can be converted to non stream inference mode.
    self._testTFLite(model_name='att_mh_rnn')

  def test_sequential_to_functional(self):
    # prepare input data
    np.random.seed(1)
    tf.random.set_seed(1)
    batch_input_shape = (1, 4, 2, 2)
    input_data = np.random.rand(np.prod(batch_input_shape))
    input_data = np.reshape(input_data, batch_input_shape)

    # create sequential model
    inputs = tf.keras.Input(batch_input_shape=batch_input_shape)
    net = SequentialModel(2)(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=net)
    model.summary()

    # convert keras sequential model to functional and compare them
    func_model = utils.sequential_to_functional(model)
    func_model.summary()
    self.assertAllClose(
        model.predict(input_data), func_model.predict(input_data))


if __name__ == '__main__':
  tf.test.main()
