from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import math
import time
import collections
import os
import json

import scipy.io as sio
def conv(batch_input, kernel_size, out_channels, stride):
    with tf.variable_scope("conv"):
        in_channels = batch_input.get_shape()[3]
        filter = tf.get_variable("filter", [kernel_size, kernel_size, in_channels, out_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        conv = tf.nn.conv2d(batch_input, filter, [1,stride, stride, 1], padding='SAME')
        return conv

def lrelu(x, a):
    with tf.name_scope("lrelu"):
        x=tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)

def strided_conv(batch_input, kernel_size, out_channels):
    with tf.variable_scope("strided_conv"):
        batch, in_height, in_width, in_channels = [int(d) for d in batch_input.get_shape()]
        filter = tf.get_variable("filter", [kernel_size, kernel_size, out_channels, in_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        strided_conv = tf.nn.conv2d_transpose(batch_input, filter, [batch, in_height*2, in_width*2, out_channels], [1,2,2,1], padding='SAME')
        return strided_conv

def create_generator(generator_inputs1, generator_inputs2, generator_outputs_channels):
    layers=[]
    with tf.variable_scope("encoder_1_1"):
        output = conv(generator_inputs1, 3, 32, 1)
        layers.append(output)
    with tf.variable_scope("encoder_2_1"):
        rectified = lrelu(layers[-1], 0.2)
        convolved = conv(rectified, 3, 32, 1)
        layers.append(convolved)
    with tf.variable_scope("encoder_3_1"):
        rectified = lrelu(layers[-1], 0.2)
        convolved = conv(rectified, 2, 64, 2)
        layers.append(convolved)

    with tf.variable_scope("encoder_1_2"):
        output = conv(generator_inputs2, 3, 32, 1)
        layers.append(output)
    with tf.variable_scope("encoder_2_2"):
        rectified = lrelu(layers[-1], 0.2)
        convolved = conv(rectified, 3, 32, 1)
        layers.append(convolved)
    with tf.variable_scope("encoder_3_2"):
        rectified = lrelu(layers[-1], 0.2)
        convolved = conv(rectified, 2, 64, 2)
        layers.append(convolved)

    concat1 = tf.concat([layers[-1], layers[-1-3]], 3)
    
    with tf.variable_scope("encoder_4"):
        rectified = lrelu(concat1, 0.2)
        convolved = conv(rectified, 3, 128, 1)
        layers.append(convolved)
    
    with tf.variable_scope("encoder_5"):
        rectified = lrelu(layers[-1], 0.2)
        convolved = conv(rectified, 3, 128, 1)
        layers.append(convolved)
    
    
    with tf.variable_scope("encoder_6"):
        rectified = lrelu(layers[-1], 0.2)
        convolved = conv(rectified, 3, 256, 1)
        layers.append(convolved)
    
    
    with tf.variable_scope("decoder_7"):
        rectified = lrelu(layers[-1], 0.2)
        convolved = conv(rectified, 1, 256, 1)
        layers.append(convolved)
    
   
    with tf.variable_scope("decoder_8"):
        rectified = lrelu(layers[-1], 0.2)
        convolved = conv(rectified, 3, 256, 1)
        layers.append(convolved)
    
    sum1 = layers[-1] + layers[-1-2]
    with tf.variable_scope("decoder_9"):
        rectified = lrelu(sum1, 0.2)
        convolved = conv(rectified, 3, 128, 1)
        layers.append(convolved)

   
    sum2 = layers[-1] + layers[-1-4] 
    with tf.variable_scope("decoder_10"):
        rectified = lrelu(sum2, 0.2)
        convolved = conv(rectified, 3, 128, 1)
        layers.append(convolved)
    
    sum3 = layers[-1] + layers[-1-6]
    with tf.variable_scope("decoder_11"):
        rectified = lrelu(sum3, 0.2)
        strided_convolved = strided_conv(rectified, 2, 128)
        layers.append(strided_convolved)

    concat3 = tf.concat([layers[-1], layers[-1-9], layers[-1-12]], 3)

    with tf.variable_scope("decoder_12"):
        rectified = lrelu(concat3, 0.2)
        convolved = conv(rectified, 3, 64, 1)
        layers.append(convolved)

    with tf.variable_scope("decoder_13"):
        rectified = lrelu(layers[-1], 0.2)
        convolved = conv(rectified, 3, generator_outputs_channels, 1)
        output = tf.nn.relu(convolved)
        layers.append(output)

    return layers[-1]

