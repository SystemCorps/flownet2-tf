import os
from src.net import Mode
from src.flownet2.flownet2 import FlowNet2
from src.dataloader import load_batch
from src.dataset_configs import FLYING_CHAIRS_DATASET_CONFIG
import keras
import tensorflow as tf

net = FlowNet2()
input_a, input_b, flow = load_batch(FLYING_CHAIRS_DATASET_CONFIG, 'sample', net.global_step)

a = 0
"""
with tf.Session() as sess:
    saver = tf.train.import_meta_graph('./checkpoints/FlowNet2/flownet-2.ckpt-0.meta')
    saver.restore(sess, './checkpoints/FlowNet2/flownet-2.ckpt-0')
    saver.
"""