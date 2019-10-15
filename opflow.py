import keras
import tensorflow as tf
import numpy as np
import cv2
from glob import glob
import os
from src.flowlib import flow_to_image
from src.flownet2.flownet2 import FlowNet2
from src.training_schedules import LONG_SCHEDULE

img_dir = [y for x in os.walk('./test') for y in glob(os.path.join(x[0], '*.png'))]
img_dir.sort()


a = cv2.imread(img_dir[0]) / 255.0
b = cv2.imread(img_dir[1]) / 255.0
inputs = {'input_a': tf.expand_dims(tf.constant(a, dtype=tf.float32), 0),
         'input_b': tf.expand_dims(tf.constant(b, dtype=tf.float32), 0)}

model = FlowNet2(mode=Mode.TEST, debug=False)

predictions = model(inputs, LONG_SCHEDULE)
pred_flow = predictions['flow']
saver = tf.train.Saver()

with tf.Session() as sess:
    #saver = tf.train.import_meta_graph('./checkpoints/FlowNet2/flownet-2.ckpt-0.meta')
    saver.restore(sess, './checkpoints/FlowNet2/flownet-2.ckpt-0')
    pred = sess.run(pred_flow)
    print(pred.keys())
    pred_flow = pred['flow']
    fimg = flow_to_image(pred_flow)
    
    cv2.imwrite('aaa.png',fimg)