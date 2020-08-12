""" 
Different Encoder Layer variants:
    - Standard Encoder Layer
    - No Dropout Layer
    - Swapped Normalization Layer
    - Gated + swapped norm Layer with GRU hack
    - prenorm without relu activation
    - Gated with out1put sigmoid
    - New GRU gate
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

from transformer_build_essentials import *


class EncoderLayer_1(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff,rate,custom_layer,layerN):
        super(EncoderLayer_1, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads,custom_layer,layerN)
        self.ffn = point_wise_feed_forward_network(d_model,dff,custom_layer,layerN)


        self.layernorm1 = tf.keras.layers.BatchNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.BatchNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, attn = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
        return out2, attn


class EncoderLayer_2(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff,rate,custom_layer,layerN):
        super(EncoderLayer_2, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads,custom_layer,layerN)
        self.ffn = point_wise_feed_forward_network(d_model,dff,custom_layer,layerN)

        self.layernorm1 = tf.keras.layers.BatchNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.BatchNormalization(epsilon=1e-6)



    def call(self, x, training, mask):
        attn_output, attn = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2, attn


class EncoderLayer_3(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff,rate,custom_layer,layerN):
        super(EncoderLayer_3, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads,custom_layer,layerN)
        self.ffn = point_wise_feed_forward_network(d_model,dff,custom_layer,layerN)


        self.layernorm1 = tf.keras.layers.BatchNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.BatchNormalization(epsilon=1e-6)

        #self.dropout1 = tf.keras.layers.Dropout(rate)
        #self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):

        out1 = self.layernorm1(x)  # (batch_size, input_seq_len, d_model)
        attn_output, attn = self.mha(out1, out1, out1, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = tf.nn.relu(attn_output)
        sum1 = attn_output + x

        out2 = self.layernorm2(sum1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.ffn(out2)  # (batch_size, input_seq_len, d_model)
        ffn_output = tf.nn.relu(ffn_output)
        sum2 = sum1 + ffn_output 

        #attn_output = self.dropout1(attn_output, training=training)
        #ffn_output = self.dropout2(ffn_output, training=training)
        return sum2 , attn

    

class EncoderLayer_4(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff,rate,custom_layer,layerN):
        super(EncoderLayer_4, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads,custom_layer,layerN)
        self.ffn = point_wise_feed_forward_network(d_model,dff,custom_layer,layerN)


        self.layernorm1 = tf.keras.layers.BatchNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.BatchNormalization(epsilon=1e-6)

        self.gru1 = tf.keras.layers.GRU(d_model)
        self.gru2 = tf.keras.layers.GRU(d_model)

        #self.dropout1 = tf.keras.layers.Dropout(rate)
        #self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):

        out1 = self.layernorm1(x)  # (batch_size, input_seq_len, d_model)
        attn_output, attn = self.mha(out1, out1, out1, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = tf.nn.relu(attn_output)

        # I think this is the best way. Might need some revisiting to check if it works as intended. 
        # -> We collapse the time dimension into the batch dimension in order to be able to use the GRU layer without it calculating multiple timesteps
        ##
        ## Encountered a crazy Bug here. When the tf-agent is initalized, it runs some tensors with None dimensions (No Idea why). This will crash the model. 
        ## In order to prevent this, we include a small hack here. As soon as initializing is over, this should have no purpose

        if None in x.shape:
            print("Non Dimension found in input tensor. Forward will be ignored. This might break your model!")
            return x

        else:
            format_x = tf.expand_dims(tf.reshape(x,shape=(x.shape[0]*x.shape[1],x.shape[2])),1)
            format_attn_output = tf.reshape(attn_output,shape=(attn_output.shape[0]*attn_output.shape[1],x.shape[2]))

            gate1 = self.gru1(format_x,format_attn_output )
            gate1 = tf.reshape(gate1,shape=(x.shape[0],x.shape[1],x.shape[2]))
        ##

            out2 = self.layernorm2(gate1)  # (batch_size, input_seq_len, d_model)
            ffn_output = self.ffn(out2)  # (batch_size, input_seq_len, d_model)
            ffn_output = tf.nn.relu(ffn_output)

        ##

            format_gate1 = tf.expand_dims(tf.reshape(gate1,shape=(gate1.shape[0]*gate1.shape[1],gate1.shape[2])),1)
            format_ffn_output = tf.reshape(ffn_output,shape=(ffn_output.shape[0]*ffn_output.shape[1],ffn_output.shape[2]))

            gate2 = self.gru2(format_gate1,format_ffn_output )
            gate2 = tf.reshape(gate2,shape=(gate1.shape[0],gate1.shape[1],gate1.shape[2]))
        ##
        
        #attn_output = self.dropout1(attn_output, training=training)
        #ffn_output = self.dropout2(ffn_output, training=training)
            return  gate2, attn



class EncoderLayer_5(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff,rate,custom_layer,layerN):
        super(EncoderLayer_5, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads,custom_layer,layerN)
        self.ffn = point_wise_feed_forward_network(d_model,dff,custom_layer,layerN)


        self.layernorm1 = tf.keras.layers.BatchNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.BatchNormalization(epsilon=1e-6)

        #self.dropout1 = tf.keras.layers.Dropout(rate)
        #self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):

        out1 = self.layernorm1(x)  # (batch_size, input_seq_len, d_model)
        attn_output, attn = self.mha(out1, out1, out1, mask)  # (batch_size, input_seq_len, d_model)
        #attn_output = tf.nn.relu(attn_output)
        sum1 = attn_output + x
        out2 = self.layernorm2(sum1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.ffn(out2)  # (batch_size, input_seq_len, d_model)
        #ffn_output = tf.nn.relu(ffn_output)
        sum2 = sum1 + ffn_output 

        #attn_output = self.dropout1(attn_output, training=training)
        #ffn_output = self.dropout2(ffn_output, training=training)
        return sum2 , attn

class EncoderLayer_6(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff,rate,custom_layer,layerN):
        super(EncoderLayer_6, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads,custom_layer,layerN)
        self.ffn = point_wise_feed_forward_network(d_model,dff,custom_layer,layerN)


        self.layernorm1 = tf.keras.layers.BatchNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.BatchNormalization(epsilon=1e-6)

        self.gate1 = gr_output(d_model)
        self.gate2 = gr_output(d_model)

    def call(self, x, training, mask):

        out1 = self.layernorm1(x)  # (batch_size, input_seq_len, d_model)
        attn_output, attn = self.mha(out1, out1, out1, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = tf.nn.relu(attn_output)
        sum1 = self.gate1([x,attn_output])
        out2 = self.layernorm2(sum1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.ffn(out2)  # (batch_size, input_seq_len, d_model)
        ffn_output = tf.nn.relu(ffn_output)
        sum2 = self.gate1([sum1,ffn_output])

        return sum2 , attn


class EncoderLayer_7(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff,rate,custom_layer,layerN):
        super(EncoderLayer_7, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads,custom_layer,layerN)
        self.ffn = point_wise_feed_forward_network(d_model,dff,custom_layer,layerN)


        self.layernorm1 = tf.keras.layers.BatchNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.BatchNormalization(epsilon=1e-6)

        self.gate1 = gru(d_model)
        self.gate2 = gru(d_model)

    def call(self, x, training, mask):

        out1 = self.layernorm1(x)  # (batch_size, input_seq_len, d_model)
        attn_output, attn = self.mha(out1, out1, out1, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = tf.nn.relu(attn_output)
        sum1 = self.gate1([x,attn_output])

        out2 = self.layernorm2(sum1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.ffn(out2)  # (batch_size, input_seq_len, d_model)
        ffn_output = tf.nn.relu(ffn_output)
        sum2 = self.gate1([sum1,ffn_output])

        return sum2 , attn


