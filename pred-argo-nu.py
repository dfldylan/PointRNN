import os
import sys
import io
from datetime import datetime
import argparse
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from PIL import Image
import models.argo_nu as models
from datasets.datasets_loop import Datasets
import time

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='/root/datasets/normal', help='Dataset directory [default: data/argo-5m]')
parser.add_argument('--dataset', default='argo', help='Dataset. argo or nu [default: argo]')
parser.add_argument('--batch-size', type=int, default=1, help='Batch Size during training [default: 4]')
parser.add_argument('--num-iters', type=int, default=200000, help='Iterations to run [default: 200000]')
parser.add_argument('--save-iters', type=int, default=1000, help='Iterations to save checkpoints [default: 1000]')
parser.add_argument('--learning-rate', type=float, default=1e-5, help='Learning rate [default: 1e-5]')
parser.add_argument('--max-gradient-norm', type=float, default=5.0, help='Clip gradients to this norm [default: 5.0].')
parser.add_argument('--seq-length', type=int, default=1, help='Length of sequence [default: 10]')
parser.add_argument('--num-points', type=int, default=1024, help='Number of points [default: 1024]')
parser.add_argument('--num-samples', type=int, default=8, help='Number of samples [default: 8]')
parser.add_argument('--unit', type=str, default='pointrnn',
                    help='Unit. pointrnn, pointgru or pointlstm [default: pointrnn]')
parser.add_argument('--alpha', type=float, default=1.0, help='Weigh on CD loss [default: 1.0]')
parser.add_argument('--beta', type=float, default=1.0, help='Weigh on EMD loss [default: 1.0]')
parser.add_argument('--log-dir', default='outputs', help='Log dir [default: outputs]')

args = parser.parse_args()
np.random.seed(999)
tf.set_random_seed(999)

args.log_dir += '/%s-%s' % (args.dataset, args.unit)

# if args.dataset == 'argo':
#     from datasets.argo_nu import Argoverse as Dataset
# if args.dataset == 'nu':
#     from datasets.argo_nu import nuScenes as Dataset

train_dataset = Datasets(seq_length=args.seq_length)

point_size = 5
axes_limits = [[-5, 5], [-5, 5], [-5, 5]]  # X axis range  # Y axis range  # Z axis range
axes_str = ["X", "Y", "Z"]
axes = [1, 0, 2]


def get_batch(dataset, batch_size):
    for i in range(len(dataset)):
        yield np.expand_dims(dataset[i], axis=0)


model_name = 'Point' + args.unit[5:].upper()
Model = getattr(models, model_name)
model = Model(batch_size=1,
              seq_length=args.seq_length,
              num_points=args.num_points,
              num_samples=args.num_samples,
              knn=True,
              is_training=False)


checkpoint_dir = os.path.join(args.log_dir, 'checkpoints')
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
checkpoint_path = os.path.join(checkpoint_dir, 'ckpt')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter(os.path.join(args.log_dir, 'summary'), sess.graph)
    batch_data_generate = get_batch(dataset=train_dataset, batch_size=args.batch_size)
    for i in range(args.num_iters):
        batch_data = batch_data_generate.__next__()
        train_dataset.write_csv(batch_data[0, 0])
        for step in range(train_dataset.pred_loop_num):
            feed_dict = {model.inputs: batch_data}
            [predictions] = sess.run([model.predicted_frames], feed_dict=feed_dict)
            train_dataset.fps += 1
            train_dataset.write_csv(predictions[0,0])
            batch_data = predictions

