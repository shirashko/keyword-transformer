"""Quick eval script that loads name-based checkpoints via tf.train.Saver."""
import sys
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import tensorflow.compat.v1 as tf
import numpy as np
from kws_streaming.models import models, model_flags
from kws_streaming.models import kws_transformer
from kws_streaming.data import input_data
from kws_streaming.train import base_parser

def evaluate(train_dir, data_dir, distill, model_args):
    parser = base_parser.base_parser()
    subparsers = parser.add_subparsers(dest='model_name', help='NN model name')
    parser_kwt = subparsers.add_parser('kws_transformer')
    kws_transformer.model_parameters(parser_kwt)

    # distill_teacher_json must go BEFORE the subcommand
    base_args = ['--data_dir', data_dir, '--train_dir', train_dir, '--train', '0']
    if distill:
        base_args += ['--distill_teacher_json', distill]
    # Find subcommand position in model_args
    args = base_args + model_args
    flags, _ = parser.parse_known_args(args)
    flags = model_flags.update_flags(flags)
    flags.training = False
    flags.batch_size = 100

    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    tf.keras.backend.set_session(sess)
    tf.keras.backend.set_learning_phase(0)

    audio_processor = input_data.AudioProcessor(flags)
    model = models.MODELS[flags.model_name](flags)

    is_distilled = distill is not None
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    if is_distilled:
        model.compile(loss=loss, loss_weights=[0.5, 0.5, 0.0], metrics=['accuracy'])
    else:
        model.compile(loss=loss, metrics=['accuracy'])

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    ckpt_path = os.path.join(train_dir, 'best_weights.ckpt')
    saver.restore(sess, ckpt_path)
    print(f"Loaded checkpoint: {ckpt_path}")

    set_size = audio_processor.set_size('testing')
    set_size = int(set_size / flags.batch_size) * flags.batch_size
    total_acc_ensemble = 0.0
    total_acc_label = 0.0
    total_acc_distill = 0.0
    total_accuracy = 0.0
    count = 0.0

    for i in range(0, set_size, flags.batch_size):
        test_fingerprints, test_ground_truth = audio_processor.get_data(
            flags.batch_size, i, flags, 0.0, 0.0, 0, 'testing', 0.0, 0.0, sess)
        one_hot = tf.keras.utils.to_categorical(test_ground_truth, num_classes=flags.label_count)
        if is_distilled:
            targets = [one_hot, one_hot, one_hot]
        else:
            targets = one_hot
        result = model.test_on_batch(test_fingerprints, targets)
        if is_distilled:
            total_acc_label += result[4]
            total_acc_distill += result[5]
            total_acc_ensemble += result[6]
        else:
            total_accuracy += result[1]
        count += 1.0

    if is_distilled:
        print(f"\n*** Test accuracy (label head):   {total_acc_label / count * 100:.2f}% ***")
        print(f"*** Test accuracy (distill head): {total_acc_distill / count * 100:.2f}% ***")
        print(f"*** Test accuracy (ensemble):     {total_acc_ensemble / count * 100:.2f}% (N={set_size}) ***\n")
    else:
        print(f"\n*** Test accuracy: {total_accuracy / count * 100:.2f}% (N={set_size}) ***\n")

if __name__ == '__main__':
    argv = sys.argv[1:]
    distill = None
    if '--distill' in argv:
        idx = argv.index('--distill')
        distill = argv[idx + 1]
        argv = argv[:idx] + argv[idx+2:]
    train_dir = argv[0]
    data_dir = argv[1]
    model_args = argv[2:]
    evaluate(train_dir, data_dir, distill, model_args)
