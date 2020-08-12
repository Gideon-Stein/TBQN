"""
Different Encoder Variants:
    - Classic
    - NO Embedding
    - No Embedding, No Dropout
    - No initial layer ( Positinal encoding will be added to the input. The input dimension has to be of the size d_model)
    - No initial layersyer No Dropout
"""
 

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import time
import numpy as np
import keras
import random
import copy
tf.__version__
import warnings
warnings.filterwarnings('ignore')

from transformers_enc_layers import *
# So we can import the encoder easily with a variable
encoderLayer = {1:EncoderLayer_1,2:EncoderLayer_2,3:EncoderLayer_3,4:EncoderLayer_4,5:EncoderLayer_5,6:EncoderLayer_6,7:EncoderLayer_7}


class Encoder_1(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,max_horizon,
                   rate,target_vocab_size=2,enc_layer_type=1):
        super(Encoder_1, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(max_horizon, self.d_model)



        self.enc_layers = [encoderLayer[enc_layer_type](d_model, num_heads, dff, rate,custom_layer) 
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    
    def call(self, x, training, mask):  
        attnStack = []
        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, attn= self.enc_layers[i](x, training, mask)
            attnStack.append(attn)
        x = self.final_layer(x)

        return x, np.array(attnStack)  # (batch_size, input_seq_len, d_model)

    
class Encoder_2(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, max_horizon,rate,
                   custom_layer,custom_last_layer,target_vocab_size=2 ,enc_layer_type=1):
        super(Encoder_2, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pseudo_embedding = tf.keras.layers.Dense(d_model)   # in order to bring the states on the d_model size
        self.pos_encoding = positional_encoding(max_horizon,self.d_model)

        if custom_layer == None:
            self.enc_layers = [encoderLayer[enc_layer_type](d_model, num_heads, dff,rate,None,None) 
                            for _ in range(num_layers)]
        else:
            self.enc_layers = [encoderLayer[enc_layer_type](d_model, num_heads, dff,rate,custom_layer,x+1) 
                            for x in range(num_layers)] 

        self.dropout = tf.keras.layers.Dropout(rate)
        
        if custom_layer != None and custom_last_layer == True:
            kernel_init = glorot_adapted(num_layers+1,custom_layer)
            self.final_layer = tf.keras.layers.Dense(target_vocab_size,kernel_initializer=kernel_init)
        else:
            self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    
    def call(self, x, training, mask):
        attnStack = []
        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.pseudo_embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x,attn = self.enc_layers[i](x, training, mask)
            attnStack.append(attn)            
        x = self.final_layer(x)
        

        return x, np.array(attnStack)  # (batch_size, input_seq_len, d_model)


class Encoder_3(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, max_horizon,rate,
                   custom_layer,custom_last_layer,target_vocab_size=2 ,enc_layer_type=1):
        super(Encoder_3, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pseudo_embedding = tf.keras.layers.Dense(d_model)   # in order to bring the states on the d_model size
        self.pos_encoding = positional_encoding(max_horizon,self.d_model)


        if custom_layer == None:
            self.enc_layers = [encoderLayer[enc_layer_type](d_model, num_heads, dff,rate,None,None) 
                            for _ in range(num_layers)]
        else:
            self.enc_layers = [encoderLayer[enc_layer_type](d_model, num_heads, dff,rate,custom_layer,x+1) 
                            for x in range(num_layers)] 

        if custom_layer != None and custom_last_layer == True:
            kernel_init = glorot_adapted(num_layers+1,custom_layer)
            self.final_layer = tf.keras.layers.Dense(target_vocab_size,kernel_initializer=kernel_init)
        else:
            self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    
    def call(self, x, training, mask):
        attnStack = []
        seq_len = tf.shape(x)[1]
        # adding embedding and position encoding.
        x = self.pseudo_embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        for i in range(self.num_layers):
            x, attn = self.enc_layers[i](x, training, mask)
            attnStack.append(attn)
        x = self.final_layer(x)

        return x, np.array(attnStack)  # (batch_size, input_seq_len, d_model)


class Encoder_4(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, max_horizon,rate,
                   custom_layer,custom_last_layer,target_vocab_size=2 ,enc_layer_type=1):
        super(Encoder_4, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_encoding = positional_encoding(max_horizon, self.d_model)

        if custom_layer == None:
            self.enc_layers = [encoderLayer[enc_layer_type](d_model, num_heads, dff,rate,None,None) 
                            for _ in range(num_layers)]
        else:
            self.enc_layers = [encoderLayer[enc_layer_type](d_model, num_heads, dff,rate,custom_layer,x+1) 
                            for x in range(num_layers)] 

        self.dropout = tf.keras.layers.Dropout(rate)
        if custom_layer != None and custom_last_layer == True:
            kernel_init = glorot_adapted(num_layers+1,custom_layer)
            self.final_layer = tf.keras.layers.Dense(target_vocab_size,kernel_initializer=kernel_init)
        else:
            self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, x, training, mask):
        attnStack = []

        seq_len = tf.shape(x)[1]

        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x ,attn= self.enc_layers[i](x, training, mask)
            attnStack.append(attn)
        x = self.final_layer(x)

        return x, np.array(attnStack)  # (batch_size, input_seq_len, d_model)

class Encoder_5(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, max_horizon,rate,
                   custom_layer,custom_last_layer,target_vocab_size=2 ,enc_layer_type=1):
        super(Encoder_5, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_encoding = positional_encoding(max_horizon, self.d_model)

        if custom_layer == None:
            self.enc_layers = [encoderLayer[enc_layer_type](d_model, num_heads, dff,rate,None,None) 
                            for _ in range(num_layers)]
        else:
            self.enc_layers = [encoderLayer[enc_layer_type](d_model, num_heads, dff,rate,custom_layer,x+1) 
                            for x in range(num_layers)] 

        if custom_layer != None and custom_last_layer == True:
            kernel_init = glorot_adapted(num_layers+1,custom_layer)
            self.final_layer = tf.keras.layers.Dense(target_vocab_size,kernel_initializer=kernel_init)
        else:
            self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, x, training, mask):
        attnStack = []

        seq_len = tf.shape(x)[1]
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        for i in range(self.num_layers):
            x,attn = self.enc_layers[i](x, training, mask)
            attnStack.append(attn)
        x = self.final_layer(x)
        return x, np.array(attnStack)  # (batch_size, input_seq_len, d_model)
