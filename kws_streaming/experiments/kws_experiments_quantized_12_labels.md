# Models from [paper](https://arxiv.org/abs/2005.06720) with quantization
======================================================================================

To enable post training model quantization, we uses \
--feature_type 'mfcc_op' which is numerically different with 'mfcc_tf' (the last one was used in [paper](https://arxiv.org/abs/2005.06720)). We did not run hyperparameters optimization with 'mfcc_op' feature extractor, so there can be some accuracy reduction. mfcc_op calls audio_spectrogram() and mfcc(). The last one expects squared fft magnitude, so we set fft_magnitude_squared 1.

All below models are trained with \
--feature_type 'mfcc_op' (speech mfcc feature extractor is using internal TFLite op ) and \
--preprocess 'raw' (so that  model is built end to end: speech feature extractor is part of the model)


## Set up python kws_streaming.

Set main folder.
```shell
# create main folder
mkdir test

# set path to a main folder
KWS_PATH=$PWD/test

cd $KWS_PATH
```

```shell
# copy content of kws_streaming to a folder /tmp/test/kws_streaming
git clone https://github.com/google-research/google-research.git
mv google-research/kws_streaming .
```

## Install tensorflow with deps.
```shell
# set up virtual env
pip install virtualenv
virtualenv --system-site-packages -p python3 ./venv3
source ./venv3/bin/activate

# install TensorFlow, correct TensorFlow version is important
pip install --upgrade pip
pip install tf_nightly==2.4.0-dev20200917
pip install tensorflow_addons
# was tested on tf_nightly-2.3.0.dev20200515-cp36-cp36m-manylinux2010_x86_64.whl

# install libs:
pip install pydot
pip install graphviz
pip install numpy
pip install absl-py
```

## Set up data sets:

There are two versions of data sets for training KWS which are well described
in [paper](https://arxiv.org/pdf/1804.03209.pdf)
[data sets V1 2017](http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz)
[data sets V2 2018](https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz)

```shell
# download and set up path to data set V2 and set it up
wget https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz
mkdir data2
mv ./speech_commands_v0.02.tar.gz ./data2
cd ./data2
tar -xf ./speech_commands_v0.02.tar.gz
cd ../

# path to data sets V2
DATA_PATH=$KWS_PATH/data2
```

## Models training and evaluation:

There are two options of running python script. One with bazel and another by calling python directly shown below:
```shell
# CMD_TRAIN="bazel run -c opt --copt=-mavx2 kws_streaming/train:model_train_eval --"
CMD_TRAIN="python -m kws_streaming.train.model_train_eval"
```

### att_mh_rnn

parameters: 700K \
float accuracy: 97.9 model size: 3400KB; latency 8ms \
quant accuracy: 97.8 model size: 1300KB; latency 4ms

```shell
$CMD_TRAIN \
--data_url '' \
--data_dir $DATA_PATH/ \
--train_dir $MODELS_PATH/att_mh_rnn/ \
--mel_upper_edge_hertz 8000 \
--how_many_training_steps 20000,20000,20000,20000 \
--learning_rate 0.001,0.0005,0.0001,0.00002 \
--window_size_ms 40.0 \
--window_stride_ms 20.0 \
--mel_num_bins 40 \
--dct_num_features 20 \
--resample 0.15 \
--alsologtostderr \
--train 0 \
--lr_schedule 'exp' \
--use_spec_augment 1 \
--time_masks_number 2 \
--time_mask_max_size 10 \
--frequency_masks_number 2 \
--frequency_mask_max_size 5 \
--feature_type 'mfcc_op' \
--fft_magnitude_squared 1 \
att_mh_rnn \
--cnn_filters '10,1' \
--cnn_kernel_size '(5,1),(5,1)' \
--cnn_act "'relu','relu'" \
--cnn_dilation_rate '(1,1),(1,1)' \
--cnn_strides '(1,1),(1,1)' \
--rnn_layers 2 \
--rnn_type 'gru' \
--rnn_units 128 \
--heads 4 \
--dropout1 0.2 \
--units2 '64,32' \
--act2 "'relu','linear'"
```
