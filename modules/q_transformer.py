import numpy as np 
import tensorflow as tf
from tf_agents.networks import network
from transformers_encoders import * 



encoderList = {1:Encoder_1,2:Encoder_2,3:Encoder_3,4:Encoder_4,5:Encoder_5}

class QTransformer(network.Network):

    def __init__(self,
    input_tensor_spec,
    output_n,
    num_layers,
    d_model,
    num_heads, 
    dff,
    rate,
    encoderType,
    enc_layer_type,
    max_horizon,
    custom_layer,
    custom_last_layer,
    name='QTrans',):
        
        #Creates an instance of `QNetwork`.

        super(QTransformer, self).__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=(),
            name=name)
        self.max_horizon = max_horizon
        self._q_value_layer = encoderList[encoderType](num_layers, d_model, num_heads, 
                         dff,max_horizon,rate,target_vocab_size=output_n,
                         enc_layer_type=enc_layer_type,custom_layer=custom_layer,custom_last_layer=custom_last_layer)
    
    def reshape_history(self,observation):
        if observation.shape[0] == 0 or observation.shape[0] == None :
            print("Initializing stuff.Weird Tensor Catch to prevent the model to break Case 2 (Empty Tensor)")
            b = int(len(self.input_tensor_spec.maximum)  /self.max_horizon)
            return tf.fill([1,self.max_horizon,b],1.)

        elif observation.shape[-1] != self.input_tensor_spec.shape:
            print("Tensor without history detected. This should not happen. Added a History Dimension")
            return tf.expand_dims(observation,1) 

        elif len(observation.shape) == 1:   # handels pyEnv steps in order to render
            a = self.max_horizon
            b = int(observation.shape[0]  /self.max_horizon)
            observation = tf.reshape(observation,[1,a,b])
            return observation
        else:
            a = self.max_horizon
            b = int(observation.shape[1]  /self.max_horizon)
            c = observation.shape[0]
            observation = tf.reshape(observation,[c,a,b])
            return observation
        
    
    def ignore_empty_steps(self,observation):
        check = tf.reduce_sum(tf.abs(observation),2)
        return tf.math.count_nonzero(check,1) 
    #since the last timestep is not always real due to padding, we need to get the last real env state in order to find the correct q values.
    # (Qvalues are based on the last env step index)
  

    def call(self, observation, step_type=None,network_state=(), training=False,attention_out = False):
        """Runs the given observation through the network.
        Args:
          observation: The observation to provide to the network.
          step_type: The step type for the given observation. See `StepType` in
            time_step.py.
          network_state: A state tuple to pass to the network, mainly used by RNNs.
          training: Whether the output is being used for training.
        Returns:
          A tuple `(logits, network_state)`.
        """
        observation = self.reshape_history(observation)
        time_limit = self.ignore_empty_steps(observation)
        q_value, attn = self._q_value_layer(observation, training=training, mask=None)
        if attention_out == False:
            return tf.gather(q_value,time_limit-1,axis=1,batch_dims=1), network_state
        else:
            return tf.gather(q_value,time_limit-1,axis=1,batch_dims=1), attn