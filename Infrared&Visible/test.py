# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import scipy.misc
import time
import os
import glob
import cv2
import scipy.io as scio




def imread(path, is_grayscale=True):
  """
  Read image using its path.
  Default value is gray-scale, and image is read by YCbCr format as the paper said.
  """
  if is_grayscale:
    return scipy.misc.imread(path, flatten=True, mode='YCbCr').astype(np.float)
  else:
    return scipy.misc.imread(path, mode='YCbCr').astype(np.float)

def imsave(image, path):
  return scipy.misc.imsave(path, image)
  
  
def prepare_data(dataset):
    data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)))
    data = glob.glob(os.path.join(data_dir, "*.bmp"))
    data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
    data.sort(key=lambda x:int(x[len(data_dir)+1:-4]))
    return data

def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)

def encoder_ir(img):
    with tf.variable_scope('encoder_ir'):
        with tf.variable_scope('layer1_ir'):
            weights = tf.get_variable("w1_ir", initializer=tf.constant(reader.get_tensor('encoder_ir/layer1_ir/w1_ir')))
            # weights=weights_spectral_norm(weights)
            bias = tf.get_variable("b1_fg", initializer=tf.constant(reader.get_tensor('encoder_ir/layer1_ir/b1_ir')))
            conv1 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(img, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            conv1 = lrelu(conv1)

        with tf.variable_scope('layer2_ir'):
            weights = tf.get_variable("w2_ir", initializer=tf.constant(reader.get_tensor('encoder_ir/layer2_ir/w2_ir')))
            # weights=weights_spectral_norm(weights)
            bias = tf.get_variable("b2_ir", initializer=tf.constant(reader.get_tensor('encoder_ir/layer2_ir/b2_ir')))
            conv2 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(conv1, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            conv2 = lrelu(conv2)

        flag1 = tf.concat([conv1, conv2], axis=-1)

        with tf.variable_scope('layer3_ir'):
            weights = tf.get_variable("w3_ir", initializer=tf.constant(reader.get_tensor('encoder_ir/layer3_ir/w3_ir')))
            # weights=weights_spectral_norm(weights)
            bias = tf.get_variable("b3_ir", initializer=tf.constant(reader.get_tensor('encoder_ir/layer3_ir/b3_ir')))
            conv3 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(flag1, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            conv3 = lrelu(conv3)

        flag2 = tf.concat([flag1, conv3], axis=-1)

        with tf.variable_scope('layer4_ir'):
            weights = tf.get_variable("w4_ir", initializer=tf.constant(reader.get_tensor('encoder_ir/layer4_ir/w4_ir')))
            # weights=weights_spectral_norm(weights)
            bias = tf.get_variable("b4_ir", initializer=tf.constant(reader.get_tensor('encoder_ir/layer4_ir/b4_ir')))
            conv4 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(flag2, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            conv4 = lrelu(conv4)

        flag3 = tf.concat([flag2, conv4], axis=-1)
        return flag3

    #     with tf.variable_scope('layer5_ir'):
    #         weights=tf.get_variable("w5_ir",initializer=tf.constant(reader.get_tensor('encoder_ir/layer5_ir/w5_ir')))
    #         #weights=weights_spectral_norm(weights)
    #         bias=tf.get_variable("b5_ir",initializer=tf.constant(reader.get_tensor('encoder_ir/layer5_ir/b5_ir')))
    #         conv5= tf.contrib.layers.batch_norm(tf.nn.conv2d(flag3, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
    #         conv5 = lrelu(conv5)
    # return conv5


def encoder_vi(img):
    with tf.variable_scope('encoder_vi'):
        with tf.variable_scope('layer1_vi'):
            weights = tf.get_variable("w1_vi", initializer=tf.constant(reader.get_tensor('encoder_vi/layer1_vi/w1_vi')))
            # weights=weights_spectral_norm(weights)
            bias = tf.get_variable("b1_vi", initializer=tf.constant(reader.get_tensor('encoder_vi/layer1_vi/b1_vi')))
            conv1 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(img, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            conv1 = lrelu(conv1)

        with tf.variable_scope('layer2_vi'):
            weights = tf.get_variable("w2_vi", initializer=tf.constant(reader.get_tensor('encoder_vi/layer2_vi/w2_vi')))
            # weights=weights_spectral_norm(weights)
            bias = tf.get_variable("b2_vi", initializer=tf.constant(reader.get_tensor('encoder_vi/layer2_vi/b2_vi')))
            conv2 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(conv1, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            conv2 = lrelu(conv2)

        flag1 = tf.concat([conv1, conv2], axis=-1)

        with tf.variable_scope('layer3_vi'):
            weights = tf.get_variable("w3_vi", initializer=tf.constant(reader.get_tensor('encoder_vi/layer3_vi/w3_vi')))
            # weights=weights_spectral_norm(weights)
            bias = tf.get_variable("b3_vi", initializer=tf.constant(reader.get_tensor('encoder_vi/layer3_vi/b3_vi')))
            conv3 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(flag1, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            conv3 = lrelu(conv3)

        flag2 = tf.concat([flag1, conv3], axis=-1)

        with tf.variable_scope('layer4_vi'):
            weights = tf.get_variable("w4_vi", initializer=tf.constant(reader.get_tensor('encoder_vi/layer4_vi/w4_vi')))
            # weights=weights_spectral_norm(weights)
            bias = tf.get_variable("b4_vi", initializer=tf.constant(reader.get_tensor('encoder_vi/layer4_vi/b4_vi')))
            conv4 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(flag2, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            conv4 = lrelu(conv4)

        flag3 = tf.concat([flag2, conv4], axis=-1)
        return flag3

    #     with tf.variable_scope('layer5_vi'):
    #         weights = tf.get_variable("w5_vi", initializer=tf.constant(reader.get_tensor('encoder_vi/layer5_vi/w5_vi')))
    #         # weights=weights_spectral_norm(weights)
    #         bias = tf.get_variable("b5_vi", initializer=tf.constant(reader.get_tensor('encoder_vi/layer5_vi/b5_vi')))
    #         conv5 = tf.contrib.layers.batch_norm(
    #             tf.nn.conv2d(flag3, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
    #             updates_collections=None, epsilon=1e-5, scale=True)
    #         conv5 = lrelu(conv5)
    # return conv5



def decoder(img):
    # Flag1 = tf.concat([ir,vi],axis=-1)
    with tf.variable_scope('decoder'):
        with tf.variable_scope('Layer1'):
            weights = tf.get_variable("W1", initializer=tf.constant(reader.get_tensor('decoder/Layer1/W1')))
            # weights=weights_spectral_norm(weights)
            bias = tf.get_variable("B1", initializer=tf.constant(reader.get_tensor('decoder/Layer1/B1')))
            conv1 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(img, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            conv1 = lrelu(conv1)

        with tf.variable_scope('Layer2'):
            weights = tf.get_variable("W2", initializer=tf.constant(reader.get_tensor('decoder/Layer2/W2')))
            # weights=weights_spectral_norm(weights)
            bias = tf.get_variable("B2", initializer=tf.constant(reader.get_tensor('decoder/Layer2/B2')))
            conv2 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(conv1, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            conv2 = lrelu(conv2)

        with tf.variable_scope('Layer3'):
            weights = tf.get_variable("W3", initializer=tf.constant(reader.get_tensor('decoder/Layer3/W3')))
            # weights=weights_spectral_norm(weights)
            bias = tf.get_variable("B3", initializer=tf.constant(reader.get_tensor('decoder/Layer3/B3')))
            conv3 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(conv2, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            conv3 = lrelu(conv3)

        with tf.variable_scope('Layer4'):
            weights = tf.get_variable("W4", initializer=tf.constant(reader.get_tensor('decoder/Layer4/W4')))
            # weights=weights_spectral_norm(weights)
            bias = tf.get_variable("B4", initializer=tf.constant(reader.get_tensor('decoder/Layer4/B4')))
            conv4 = tf.nn.conv2d(conv3, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
            conv4 = tf.nn.tanh(conv4)
    return conv4



def input_setup(index):
    padding=0
    sub_near_sequence = []
    sub_far_sequence = []
    input_near=(imread(data_near[index])-127.5)/127.5
    input_near=np.lib.pad(input_near,((padding,padding),(padding,padding)),'edge')
    w,h=input_near.shape
    input_near=input_near.reshape([w,h,1])
    input_far=(imread(data_far[index])-127.5)/127.5
    input_far=np.lib.pad(input_far,((padding,padding),(padding,padding)),'edge')
    w,h=input_far.shape
    input_far=input_far.reshape([w,h,1])
    sub_near_sequence.append(input_near)
    sub_far_sequence.append(input_far)
    train_data_near= np.asarray(sub_near_sequence)
    train_data_far= np.asarray(sub_far_sequence)
    return train_data_near,train_data_far

for idx_num in range(0,25):
  num_epoch=idx_num
  while(num_epoch==idx_num):
      reader = tf.train.NewCheckpointReader('./checkpoint/CGAN_60/CGAN.model-'+ str(num_epoch))

      with tf.name_scope('IR_input'):
          images_near = tf.placeholder(tf.float32, [1,None,None,None], name='images_near')
      with tf.name_scope('VI_input'):
          images_far = tf.placeholder(tf.float32, [1,None,None,None], name='images_far')
      with tf.name_scope('input'):
          input_image_near =images_near
          input_image_far =images_far

      with tf.name_scope('fusion'):
          # fusion_image=fusion_model(input_image_near,input_image_far)
          image_ir_encoder = encoder_ir(input_image_near)
          image_vi_encoder = encoder_vi(input_image_far)
          fusion_image_encoder = tf.concat([image_ir_encoder, image_vi_encoder], axis=-1)
          fusion_image = decoder(fusion_image_encoder)

      with tf.Session() as sess:
          init_op=tf.global_variables_initializer()
          sess.run(init_op)
          data_near=prepare_data('Test_near')
          data_far=prepare_data('Test_far')
          for i in range(len(data_near)):
              start=time.time()
              train_data_near,train_data_far=input_setup(i)
              result =sess.run(fusion_image,feed_dict={images_near: train_data_near,images_far: train_data_far})
              print("result:",result.shape)
              result=result*127.5+127.5
              result = result.squeeze()
              image_path = os.path.join(os.getcwd(), 'result','epoch'+str(num_epoch))
              if not os.path.exists(image_path):
                  os.makedirs(image_path)
              end=time.time()
              image_path = os.path.join(image_path,str(i+1)+".jpg")
              imsave(result, image_path)
              # scio.savemat(image_path, {'I':result})
              print("Testing [%d] success,Testing time is [%f]"%(i,end-start))
      tf.reset_default_graph()
      num_epoch=num_epoch+1
