# -*- coding: utf-8 -*-
from utils import (
  read_data, 
  input_setup, 
  imsave,
  merge,
  gradient,
  lrelu,
  weights_spectral_norm,
  l2_norm,
  tf_ms_ssim,
  tf_ssim,
  blur_2th,
  mean_filter
)

# from  filter import *

import time
import os
import matplotlib.pyplot as plt
import cv2 

import numpy as np
import tensorflow as tf

class CGAN(object):

  def __init__(self, 
               sess, 
               image_size=72,
               batch_size=32,
               c_dim=1, 
               checkpoint_dir=None, 
               sample_dir=None):

    self.sess = sess
    self.is_grayscale = (c_dim == 1)
    self.image_size = image_size
    self.batch_size = batch_size

    self.c_dim = c_dim

    self.checkpoint_dir = checkpoint_dir
    self.sample_dir = sample_dir
    self.build_model()

  def build_model(self):
    with tf.name_scope('IR_input'):
        self.images_near = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='images_near')
    with tf.name_scope('VI_input'):
        self.images_far = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='images_far')


    with tf.name_scope('input'):
        self.input_image_near =self.images_near
        self.input_image_far =self.images_far


    with tf.name_scope('fusion'): 
        # self.fusion_image=self.fusion_model(self.input_image_near,self.input_image_far)
        self.image_ir_encoder=self.encoder_ir(self.input_image_near)
        self.image_vi_encoder=self.encoder_vi(self.input_image_far)
        self.fusion_image_encoder=tf.concat([self.image_ir_encoder,self.image_vi_encoder],axis=-1)
        self.fusion_image=self.decoder(self.fusion_image_encoder)
    with tf.name_scope('grad_bin'):
        self.Image_far_grad=tf.abs(gradient(self.images_far))
        self.Image_near_grad=tf.abs(gradient(self.images_near))
        self.Image_fused_grad=tf.abs(gradient(self.fusion_image))
        self.Image_far_weight=tf.abs(blur_2th(self.images_far))
        self.Image_near_weight=tf.abs(blur_2th(self.images_near))

        self.Image_near_intensity = mean_filter(self.images_near)
        self.Image_fused_intensity = mean_filter(self.fusion_image)
        
        self.Image_far_score=tf.sign(self.Image_far_weight-tf.minimum(self.Image_far_weight,self.Image_near_weight))
        self.Image_near_score=1-self.Image_far_score

        self.Image_far_score_ave=tf.reduce_mean(tf.sign((self.Image_far_weight-tf.minimum(self.Image_far_weight,self.Image_near_weight))))        
        self.Image_near_score_ave= 1- self.Image_far_score_ave
 
                      
        # self.Image_far_near_grad_bin=tf.maximum(self.Image_far_grad,self.Image_near_grad)
        # self.Image_fused_grad_bin=self.Image_fused_grad
    


    with tf.name_scope('image'):
        tf.summary.image('input_near',tf.expand_dims(self.images_near[1,:,:,:],0))  
        tf.summary.image('input_far',tf.expand_dims(self.images_far[1,:,:,:],0))  
        tf.summary.image('fusion_image',tf.expand_dims(self.fusion_image[1,:,:,:],0)) 
        tf.summary.image('Image_far_grad',tf.expand_dims(self.Image_far_grad[1,:,:,:],0)) 
        tf.summary.image('Image_near_grad',tf.expand_dims(self.Image_near_grad[1,:,:,:],0))
        # tf.summary.image('Image_far_near_grad_bin',tf.expand_dims(self.Image_far_near_grad_bin[1,:,:,:],0))
        # tf.summary.image('Image_fused_grad_bin',tf.expand_dims(self.Image_fused_grad_bin[1,:,:,:],0))
        tf.summary.image('self.Image_near_intensity', tf.expand_dims(self.Image_near_intensity[1, :, :, :], 0))
        tf.summary.image('self.Image_fused_intensity', tf.expand_dims(self.Image_fused_intensity[1, :, :, :], 0))
        


          
    with tf.name_scope('d_loss_grad'):
        pos_grad=self.discriminator_grad(self.Image_far_grad,reuse=False)
        neg_grad=self.discriminator_grad(self.Image_fused_grad,reuse=True,update_collection='NO_OPS')
        pos_loss_grad=tf.reduce_mean(tf.square(pos_grad-tf.random_uniform(shape=[self.batch_size,1],minval=0.7,maxval=1.2,dtype=tf.float32)))
        neg_loss_grad=tf.reduce_mean(tf.square(neg_grad-tf.random_uniform(shape=[self.batch_size,1],minval=0,maxval=0.3,dtype=tf.float32)))

        self.d_loss_grad=neg_loss_grad+pos_loss_grad

        tf.summary.scalar('pos_grad', tf.reduce_mean(pos_grad))
        tf.summary.scalar('neg_grad', tf.reduce_mean(neg_grad))
        tf.summary.scalar('loss_d_grad',self.d_loss_grad)

    with tf.name_scope('d_loss_int'):
        pos_int=self.discriminator_int(self.Image_near_intensity,reuse=False)
        neg_int=self.discriminator_int(self.Image_fused_intensity,reuse=True,update_collection='NO_OPS')
        pos_loss_int=tf.reduce_mean(tf.square(pos_int-tf.random_uniform(shape=[self.batch_size,1],minval=0.7,maxval=1.2,dtype=tf.float32)))
        neg_loss_int=tf.reduce_mean(tf.square(neg_int-tf.random_uniform(shape=[self.batch_size,1],minval=0,maxval=0.3,dtype=tf.float32)))

        self.d_loss_int=neg_loss_int+pos_loss_int

        tf.summary.scalar('pos_int', tf.reduce_mean(pos_int))
        tf.summary.scalar('neg_int', tf.reduce_mean(neg_int))
        tf.summary.scalar('loss_d_int',self.d_loss_int)
        
        
        
    with tf.name_scope('g_loss'):
        self.g_loss_grad=tf.reduce_mean(tf.square(neg_grad-tf.random_uniform(shape=[self.batch_size,1],minval=0.7,maxval=1.2,dtype=tf.float32)))
        tf.summary.scalar('g_loss_grad',self.g_loss_grad)

        self.g_loss_int = tf.reduce_mean(tf.square(neg_int - tf.random_uniform(shape=[self.batch_size, 1], minval=0.7, maxval=1.2, dtype=tf.float32)))
        tf.summary.scalar('g_loss_int', self.g_loss_int)
        
        # self.g_loss_int=tf.reduce_mean(self.Image_far_score*tf.square(self.fusion_image-self.images_far))+tf.reduce_mean(self.Image_near_score*tf.square(self.fusion_image-self.images_near))
        # self.g_loss_grad=tf.reduce_mean(self.Image_far_score*tf.square(gradient(self.fusion_image)-gradient(self.images_far)))+tf.reduce_mean(self.Image_near_score*tf.square(gradient(self.fusion_image)-gradient(self.images_near)))
        # self.g_loss_2 = 5 * self.g_loss_grad + self.g_loss_int
        # self.g_loss_content_int = tf.reduce_mean(1 * tf.square(self.fusion_image - self.images_far)) + tf.reduce_mean(10 * tf.square(self.fusion_image - self.images_near))
        # self.g_loss_content_grad = tf.reduce_mean(5 * tf.square(gradient(self.fusion_image) - gradient(self.images_far))) + tf.reduce_mean(1 * tf.square(gradient(self.fusion_image) - gradient(self.images_near)))
        # self.g_loss_content = 2 * self.g_loss_grad + 5 *self.g_loss_int

        # self.g_loss_content_int = 1 * tf.reduce_mean(tf.square(self.fusion_image - self.images_far)) + 10 * tf.reduce_mean(tf.square(self.fusion_image - self.images_near))
        # self.g_loss_content_grad = 5 * tf.reduce_mean(tf.square(gradient(self.fusion_image) - gradient(self.images_far))) + 1 * tf.reduce_mean(tf.square(gradient(self.fusion_image) - gradient(self.images_near)))
        # self.g_loss_content = 1 * self.g_loss_content_grad + 1 * self.g_loss_content_int

        self.g_loss_content_int = tf.reduce_mean(self.Image_far_score * tf.square(self.fusion_image - self.images_far)) + tf.reduce_mean(self.Image_near_score * tf.square(self.fusion_image - self.images_near))
        self.g_loss_content_grad = tf.reduce_mean(self.Image_far_score * tf.square(gradient(self.fusion_image) - gradient(self.images_far))) + tf.reduce_mean(self.Image_near_score * tf.square(gradient(self.fusion_image) - gradient(self.images_near)))
        self.g_loss_content = 10 * self.g_loss_content_grad  + 1*self.g_loss_content_int


        tf.summary.scalar('self.g_loss_content_int', self.g_loss_content_int)
        tf.summary.scalar('self.g_loss_content_grad', self.g_loss_content_grad)
        tf.summary.scalar('g_loss_content',self.g_loss_content)
        self.g_loss_total=1*self.g_loss_grad+1*self.g_loss_int+10*self.g_loss_content
        tf.summary.scalar('loss_g',self.g_loss_total)        
    self.saver = tf.train.Saver(max_to_keep=50)
    
  def train(self, config):
    if config.is_train:
      input_setup(self.sess, config,"Train_near")
      input_setup(self.sess,config,"Train_far")
    else:
      nx_near, ny_near = input_setup(self.sess, config,"Test_near")
      nx_far,ny_far=input_setup(self.sess, config,"Test_far")

    if config.is_train:     
      data_dir_near = os.path.join('./{}'.format(config.checkpoint_dir), "Train_near","train.h5")
      data_dir_far = os.path.join('./{}'.format(config.checkpoint_dir), "Train_far","train.h5")
    else:
      data_dir_near = os.path.join('./{}'.format(config.checkpoint_dir),"Test_near", "test.h5")
      data_dir_far = os.path.join('./{}'.format(config.checkpoint_dir),"Test_far", "test.h5")

    train_data_near= read_data(data_dir_near)
    train_data_far= read_data(data_dir_far)

    t_vars = tf.trainable_variables()
    self.d_vars_grad = [var for var in t_vars if 'discriminator_grad' in var.name]
    print(self.d_vars_grad)
    self.d_vars_int = [var for var in t_vars if 'discriminator_int' in var.name]
    print(self.d_vars_int)
    self.encoder_ir_vars = [var for var in t_vars if 'encoder_ir' in var.name]
    print(self.encoder_ir_vars)
    self.encoder_vi_vars = [var for var in t_vars if 'encoder_vi' in var.name]
    print(self.encoder_vi_vars)
    self.decoder_vars = [var for var in t_vars if 'decoder' in var.name]
    # self.g_vars = [var for var in t_vars if 'fusion_model' in var.name]
    # print(self.g_vars)

    with tf.name_scope('train_step'):
        # self.train_fusion_op = tf.train.AdamOptimizer(config.learning_rate).minimize(self.g_loss_total,var_list=self.g_vars)
        self.train_encoder_ir_op = tf.train.AdamOptimizer(config.learning_rate).minimize(self.g_loss_total,var_list=self.encoder_ir_vars)
        self.train_encoder_vi_op = tf.train.AdamOptimizer(config.learning_rate).minimize(self.g_loss_total,var_list=self.encoder_vi_vars)
        self.train_decoder_op = tf.train.AdamOptimizer(config.learning_rate).minimize(self.g_loss_total,var_list=self.decoder_vars)
        self.train_discriminator_grad_op=tf.train.AdamOptimizer(config.learning_rate).minimize(self.d_loss_grad,var_list=self.d_vars_grad)
        self.train_discriminator_int_op = tf.train.AdamOptimizer(config.learning_rate).minimize(self.d_loss_int,var_list=self.d_vars_int)
    self.summary_op = tf.summary.merge_all()

    self.train_writer = tf.summary.FileWriter(config.summary_dir + '/train',self.sess.graph,flush_secs=60)
    
    tf.initialize_all_variables().run()
    
    counter = 0
    start_time = time.time()



    if config.is_train:
      print("Training...")

      for ep in range(config.epoch):
        # Run by batch images
        batch_idxs = len(train_data_near) // config.batch_size
        for idx in range(0, batch_idxs):
          batch_images_near = train_data_near[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_images_far = train_data_far[idx*config.batch_size : (idx+1)*config.batch_size]

          counter += 1
          for i in range(2):
            _, err_d_grad= self.sess.run([self.train_discriminator_grad_op, self.d_loss_grad], feed_dict={self.images_near: batch_images_near, self.images_far: batch_images_far})
            _, err_d_int = self.sess.run([self.train_discriminator_int_op, self.d_loss_int],feed_dict={self.images_near: batch_images_near, self.images_far: batch_images_far})
          # _, err_g,summary_str= self.sess.run([self.train_fusion_op, self.g_loss_total,self.summary_op], feed_dict={self.images_near: batch_images_near, self.images_far: batch_images_far})
          _,_,_, err_encoder_ir,err_encoder_vi,err_decoder,summary_str_encoder_ir,summary_str_encoder_vi,summary_str_decoder= self.sess.run([self.train_encoder_ir_op,self.train_encoder_vi_op,self.train_decoder_op,self.g_loss_total,self.g_loss_total,self.g_loss_total,self.summary_op,self.summary_op,self.summary_op], feed_dict={self.images_near: batch_images_near, self.images_far: batch_images_far})

          self.train_writer.add_summary(summary_str_decoder,counter)

          if counter % 10 == 0:
            print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss_d_grad: [%.8f],loss_d_int: [%.8f],loss_g:[%.8f]" \
              % ((ep+1), counter, time.time()-start_time, err_d_grad, err_d_int, err_decoder))


        self.save(config.checkpoint_dir, ep)

    else:
      print("Testing...")

      result = self.fusion_image.eval(feed_dict={self.images_near: train_data_near, self.images_far: train_data_far})
      result=result*127.5+127.5
      result = merge(result, [nx_near, ny_near])
      result = result.squeeze()
      image_path = os.path.join(os.getcwd(), config.sample_dir)
      image_path = os.path.join(image_path, "test_image.png")
      imsave(result, image_path)

  def encoder_ir(self,img):
    with tf.variable_scope('encoder_ir'):
        with tf.variable_scope('layer1_ir'):
            weights=tf.get_variable("w1_ir",[3,3,1,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            #weights=weights_spectral_norm(weights)
            bias=tf.get_variable("b1_ir",[16],initializer=tf.constant_initializer(0.0))
            conv1= tf.contrib.layers.batch_norm(tf.nn.conv2d(img, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv1 = lrelu(conv1)
        
        with tf.variable_scope('layer2_ir'):
            weights=tf.get_variable("w2_ir",[3,3,16,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            #weights=weights_spectral_norm(weights)
            bias=tf.get_variable("b2_ir",[16],initializer=tf.constant_initializer(0.0))
            conv2= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv1, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv2 = lrelu(conv2)

        flag1 = tf.concat([conv1,conv2],axis=-1)

        with tf.variable_scope('layer3_ir'):
            weights=tf.get_variable("w3_ir",[3,3,32,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            #weights=weights_spectral_norm(weights)
            bias=tf.get_variable("b3_ir",[16],initializer=tf.constant_initializer(0.0))
            conv3= tf.contrib.layers.batch_norm(tf.nn.conv2d(flag1, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv3 = lrelu(conv3)

        flag2 = tf.concat([flag1,conv3],axis=-1)

        with tf.variable_scope('layer4_ir'):
            weights=tf.get_variable("w4_ir",[3,3,48,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            #weights=weights_spectral_norm(weights)
            bias=tf.get_variable("b4_ir",[16],initializer=tf.constant_initializer(0.0))
            conv4= tf.contrib.layers.batch_norm(tf.nn.conv2d(flag2, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv4 = lrelu(conv4)
        
        flag3 = tf.concat([flag2,conv4],axis=-1)

        return flag3


  def encoder_vi(self,img):
    with tf.variable_scope('encoder_vi'):
        with tf.variable_scope('layer1_vi'):
            weights=tf.get_variable("w1_vi",[3,3,1,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            #weights=weights_spectral_norm(weights)
            bias=tf.get_variable("b1_vi",[16],initializer=tf.constant_initializer(0.0))
            conv1= tf.contrib.layers.batch_norm(tf.nn.conv2d(img, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv1 = lrelu(conv1)
        
        with tf.variable_scope('layer2_vi'):
            weights=tf.get_variable("w2_vi",[3,3,16,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            #weights=weights_spectral_norm(weights)
            bias=tf.get_variable("b2_vi",[16],initializer=tf.constant_initializer(0.0))
            conv2= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv1, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv2 = lrelu(conv2)

        flag1 = tf.concat([conv1,conv2],axis=-1)

        with tf.variable_scope('layer3_vi'):
            weights=tf.get_variable("w3_vi",[3,3,32,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            #weights=weights_spectral_norm(weights)
            bias=tf.get_variable("b3_vi",[16],initializer=tf.constant_initializer(0.0))
            conv3= tf.contrib.layers.batch_norm(tf.nn.conv2d(flag1, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv3 = lrelu(conv3)

        flag2 = tf.concat([flag1,conv3],axis=-1)

        with tf.variable_scope('layer4_vi'):
            weights=tf.get_variable("w4_vi",[3,3,48,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            #weights=weights_spectral_norm(weights)
            bias=tf.get_variable("b4_vi",[16],initializer=tf.constant_initializer(0.0))
            conv4= tf.contrib.layers.batch_norm(tf.nn.conv2d(flag2, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv4 = lrelu(conv4)
        
        flag3 = tf.concat([flag2,conv4],axis=-1)

        return flag3


  def decoder(self,img):
    #Flag1 = tf.concat([ir,vi],axis=-1)
    with tf.variable_scope('decoder'):
        with tf.variable_scope('Layer1'):
            weights=tf.get_variable("W1",[3,3,128,64],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            #weights=weights_spectral_norm(weights)
            bias=tf.get_variable("B1",[64],initializer=tf.constant_initializer(0.0))
            conv1= tf.contrib.layers.batch_norm(tf.nn.conv2d(img, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv1 = lrelu(conv1)

        with tf.variable_scope('Layer2'):
            weights=tf.get_variable("W2",[3,3,64,32],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            #weights=weights_spectral_norm(weights)
            bias=tf.get_variable("B2",[32],initializer=tf.constant_initializer(0.0))
            conv2= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv1, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv2 = lrelu(conv2)

        with tf.variable_scope('Layer3'):
            weights=tf.get_variable("W3",[3,3,32,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            #weights=weights_spectral_norm(weights)
            bias=tf.get_variable("B3",[16],initializer=tf.constant_initializer(0.0))
            conv3= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv2, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv3 = lrelu(conv3)

        with tf.variable_scope('Layer4'):
            weights=tf.get_variable("W4",[1,1,16,1],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            #weights=weights_spectral_norm(weights)
            bias=tf.get_variable("B4",[1],initializer=tf.constant_initializer(0.0))
            conv4= tf.nn.conv2d(conv3, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv4 = tf.nn.tanh(conv4)
    return conv4
    
  def discriminator_int(self,img,reuse,update_collection=None):
    with tf.variable_scope('discriminator_int',reuse=reuse):
        print(img.shape)
        with tf.variable_scope('layer_1'):
            weights=tf.get_variable("w_1",[3,3,1,32],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            weights=weights_spectral_norm(weights,update_collection=update_collection)
            bias=tf.get_variable("b_1",[32],initializer=tf.constant_initializer(0.0))
            conv1_int=tf.nn.conv2d(img, weights, strides=[1,2,2,1], padding='VALID') + bias
            conv1_int = lrelu(conv1_int)
            #print(conv1_int.shape)
        with tf.variable_scope('layer_2'):
            weights=tf.get_variable("w_2",[3,3,32,64],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            weights=weights_spectral_norm(weights,update_collection=update_collection)
            bias=tf.get_variable("b_2",[64],initializer=tf.constant_initializer(0.0))
            conv2_int= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv1_int, weights, strides=[1,2,2,1], padding='VALID') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv2_int = lrelu(conv2_int)
            #print(conv2_int.shape)
        with tf.variable_scope('layer_3'):
            weights=tf.get_variable("w_3",[3,3,64,128],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            weights=weights_spectral_norm(weights,update_collection=update_collection)
            bias=tf.get_variable("b_3",[128],initializer=tf.constant_initializer(0.0))
            conv3_int= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv2_int, weights, strides=[1,2,2,1], padding='VALID') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv3_int=lrelu(conv3_int)
            #print(conv3_int.shape)
        with tf.variable_scope('layer_4'):
            weights=tf.get_variable("w_4",[3,3,128,256],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            weights=weights_spectral_norm(weights,update_collection=update_collection)
            bias=tf.get_variable("b_4",[256],initializer=tf.constant_initializer(0.0))
            conv4_int= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv3_int, weights, strides=[1,2,2,1], padding='VALID') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv4_int=lrelu(conv4_int)
            print(conv4_int.shape)
            conv4_int = tf.reshape(conv4_int,[self.batch_size,2*2*256])
        with tf.variable_scope('line_5'):
            weights=tf.get_variable("w_5",[2*2*256,1],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            weights=weights_spectral_norm(weights,update_collection=update_collection)
            bias=tf.get_variable("b_5",[1],initializer=tf.constant_initializer(0.0))
            line_5=tf.matmul(conv4_int, weights) + bias
            #conv3_int= tf.contrib.layers.batch_norm(conv3_int, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
    return line_5

  def discriminator_grad(self,img,reuse,update_collection=None):
    with tf.variable_scope('discriminator_grad',reuse=reuse):
        print(img.shape)
        with tf.variable_scope('Layer_1'):
            weights=tf.get_variable("W_1",[3,3,1,32],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            weights=weights_spectral_norm(weights,update_collection=update_collection)
            bias=tf.get_variable("B_1",[32],initializer=tf.constant_initializer(0.0))
            conv1_grad=tf.nn.conv2d(img, weights, strides=[1,2,2,1], padding='VALID') + bias
            conv1_grad = lrelu(conv1_grad)
            #print(conv1_vi.shape)
        with tf.variable_scope('Layer_2'):
            weights=tf.get_variable("W_2",[3,3,32,64],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            weights=weights_spectral_norm(weights,update_collection=update_collection)
            bias=tf.get_variable("B_2",[64],initializer=tf.constant_initializer(0.0))
            conv2_grad= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv1_grad, weights, strides=[1,2,2,1], padding='VALID') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv2_grad = lrelu(conv2_grad)
            #print(conv2_vi.shape)
        with tf.variable_scope('Layer_3'):
            weights=tf.get_variable("W_3",[3,3,64,128],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            weights=weights_spectral_norm(weights,update_collection=update_collection)
            bias=tf.get_variable("B_3",[128],initializer=tf.constant_initializer(0.0))
            conv3_grad= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv2_grad, weights, strides=[1,2,2,1], padding='VALID') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv3_grad=lrelu(conv3_grad)
            #print(conv3_vi.shape)
        with tf.variable_scope('Layer_4'):
            weights=tf.get_variable("W_4",[3,3,128,256],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            weights=weights_spectral_norm(weights,update_collection=update_collection)
            bias=tf.get_variable("B_4",[256],initializer=tf.constant_initializer(0.0))
            conv4_grad= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv3_grad, weights, strides=[1,2,2,1], padding='VALID') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv4_grad=lrelu(conv4_grad)
            # print("conv4_grad_sape:",conv4_grad.shape)
            conv4_grad = tf.reshape(conv4_grad,[self.batch_size,2*2*256])
        with tf.variable_scope('Line_5'):
            weights=tf.get_variable("W_5",[2*2*256,1],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            weights=weights_spectral_norm(weights,update_collection=update_collection)
            bias=tf.get_variable("B_5",[1],initializer=tf.constant_initializer(0.0))
            line_5=tf.matmul(conv4_grad, weights) + bias
            #conv3_vi= tf.contrib.layers.batch_norm(conv3_vi, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
    return line_5
    
    
  # def discriminator_grad(self,img,reuse,update_collection=None):
  #   with tf.variable_scope('discriminator_grad',reuse=reuse):
  #       print(img.shape)
  #       with tf.variable_scope('layer_grad_1'):
  #           weights=tf.get_variable("w_grad_1",[3,3,1,32],initializer=tf.truncated_normal_initializer(stddev=1e-3))
  #           bias=tf.get_variable("b_grad_1",[32],initializer=tf.constant_initializer(0.0))
  #           conv1_grad=tf.nn.conv2d(img, weights, strides=[1,2,2,1], padding='VALID') + bias
  #           conv1_grad = lrelu(conv1_grad)
  #       with tf.variable_scope('layer_grad_2'):
  #           weights=tf.get_variable("w_grad_2",[3,3,32,64],initializer=tf.truncated_normal_initializer(stddev=1e-3))
  #           bias=tf.get_variable("b_grad_2",[64],initializer=tf.constant_initializer(0.0))
  #           conv2_grad= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv1_grad, weights, strides=[1,2,2,1], padding='VALID') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
  #           conv2_grad = lrelu(conv2_grad)
  #       with tf.variable_scope('layer_grad_3'):
  #           weights=tf.get_variable("w_grad_3",[3,3,64,128],initializer=tf.truncated_normal_initializer(stddev=1e-3))
  #           bias=tf.get_variable("b_grad_3",[128],initializer=tf.constant_initializer(0.0))
  #           conv3_grad= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv2_grad, weights, strides=[1,2,2,1], padding='VALID') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
  #           conv3_grad=lrelu(conv3_grad)
  #       with tf.variable_scope('layer_grad_4'):
  #           weights=tf.get_variable("w_grad_4",[3,3,128,256],initializer=tf.truncated_normal_initializer(stddev=1e-3))
  #           bias=tf.get_variable("b_grad_4",[256],initializer=tf.constant_initializer(0.0))
  #           conv4_grad= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv3_grad, weights, strides=[1,2,2,1], padding='VALID') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
  #           conv4_grad=lrelu(conv4_grad)
  #           [B,H,W,C]=conv4_grad.get_shape().as_list()
  #
  #           conv4_grad = tf.reshape(conv4_grad,[self.batch_size,H*H*256])
  #       with tf.variable_scope('line_grad_5'):
  #           weights=tf.get_variable("w_grad_5",[H*H*256,1],initializer=tf.truncated_normal_initializer(stddev=1e-3))
  #           bias=tf.get_variable("b_grad_5",[1],initializer=tf.constant_initializer(0.0))
  #           line_5=tf.matmul(conv4_grad, weights) + bias
  #   return line_5
  #
  # def discriminator_int(self,img,reuse,update_collection=None):
  #   with tf.variable_scope('discriminator_int',reuse=reuse):
  #       print(img.shape)
  #       with tf.variable_scope('layer_int_1'):
  #           weights=tf.get_variable("w_int_1",[3,3,1,32],initializer=tf.truncated_normal_initializer(stddev=1e-3))
  #           bias=tf.get_variable("b_int_1",[32],initializer=tf.constant_initializer(0.0))
  #           conv1_int=tf.nn.conv2d(img, weights, strides=[1,2,2,1], padding='VALID') + bias
  #           conv1_int = lrelu(conv1_int)
  #       with tf.variable_scope('layer_int_2'):
  #           weights=tf.get_variable("w_int_2",[3,3,32,64],initializer=tf.truncated_normal_initializer(stddev=1e-3))
  #           bias=tf.get_variable("b_int_2",[64],initializer=tf.constant_initializer(0.0))
  #           conv2_int= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv1_int, weights, strides=[1,2,2,1], padding='VALID') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
  #           conv2_int = lrelu(conv2_int)
  #       with tf.variable_scope('layer_int_3'):
  #           weights=tf.get_variable("w_int_3",[3,3,64,128],initializer=tf.truncated_normal_initializer(stddev=1e-3))
  #           bias=tf.get_variable("b_int_3",[128],initializer=tf.constant_initializer(0.0))
  #           conv3_int= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv2_int, weights, strides=[1,2,2,1], padding='VALID') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
  #           conv3_int=lrelu(conv3_int)
  #       with tf.variable_scope('layer_int_4'):
  #           weights=tf.get_variable("w_int_4",[3,3,128,256],initializer=tf.truncated_normal_initializer(stddev=1e-3))
  #           bias=tf.get_variable("b_int_4",[256],initializer=tf.constant_initializer(0.0))
  #           conv4_int= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv3_int, weights, strides=[1,2,2,1], padding='VALID') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
  #           conv4_int=lrelu(conv4_int)
  #           [B,H,W,C]=conv4_int.get_shape().as_list()
  #
  #           conv4_int = tf.reshape(conv4_int,[self.batch_size,H*H*256])
  #       with tf.variable_scope('line_int_5'):
  #           weights=tf.get_variable("w_int_5",[H*H*256,1],initializer=tf.truncated_normal_initializer(stddev=1e-3))
  #           bias=tf.get_variable("b_int_5",[1],initializer=tf.constant_initializer(0.0))
  #           line_5=tf.matmul(conv4_int, weights) + bias
  #   return line_5

  def save(self, checkpoint_dir, step):
    model_name = "CGAN.model"
    model_dir = "%s_%s" % ("CGAN", self.image_size)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
                    os.path.join(checkpoint_dir, model_name),
                    global_step=step)

  def load(self, checkpoint_dir):
    print(" [*] Reading checkpoints...")
    model_dir = "%s_%s" % ("CGAN", self.image_size)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        print(ckpt_name)
        self.saver.restore(self.sess, os.path.join(checkpoint_dir,ckpt_name))
        return True
    else:
        return False


