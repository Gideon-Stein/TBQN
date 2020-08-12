from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# GPU stuff
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"  

import tensorflow as tf

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../modules/')

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


import argparse
import pickle
import gym


import os
import time

from absl import app
from absl import flags
from absl import logging

import gin
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.agents.ppo import ppo_clip_agent
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import parallel_py_environment
from tf_agents.environments import suite_mujoco,suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import actor_distribution_rnn_network
from tf_agents.networks import value_network
from tf_agents.networks import value_rnn_network
from tf_agents.policies import policy_saver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
from q_transformer import *

from env_wrappers import *
from train_eval_ppo import *
from utils import *


def main():

    logging.set_verbosity(logging.INFO)
    tf.compat.v1.enable_v2_behavior()
    parser = argparse.ArgumentParser()

    ## Essential parameters
    parser.add_argument("--output_dir", default=None, type=str, required=True,help="The output directory where the model stats and checkpoints will be written.")
    parser.add_argument("--env", default=None, type=str, required=True,help="The environment to train the agent on")
    parser.add_argument("--max_horizon", default=4, type=int)
    parser.add_argument("--atari", default=False, type=bool, help = "Gets some data Types correctly")


    ##agent parameters
    parser.add_argument("--reward_scale_factor", default=1.0, type=float)
    parser.add_argument("--debug_summaries", default=False, type=bool)
    parser.add_argument("--summarize_grads_and_vars", default=False, type=bool)

    ##transformer parameters
    parser.add_argument("--d_model", default=64, type=int)
    parser.add_argument("--num_layers", default=3, type=int)
    parser.add_argument("--dff", default=256, type=int)

    ##Training parameters
    parser.add_argument('--num_iterations', type=int, default=100000,help="steps in the env")
    parser.add_argument('--num_parallel', type=int, default=30,help="how many envs should run in parallel")
    parser.add_argument("--collect_episodes_per_iteration", default=1, type=int)
    parser.add_argument('--num_epochs', type=int, default = 25,help = 'Number of epochs for computing policy updates.')


    ## Other parameters
    parser.add_argument("--num_eval_episodes", default=10, type=int)
    parser.add_argument("--eval_interval", default=1000, type=int)
    parser.add_argument("--log_interval", default=10, type=int)
    parser.add_argument("--summary_interval", default=1000, type=int)
    parser.add_argument("--run_graph_mode", default=True, type=bool)
    parser.add_argument("--checkpoint_interval", default=1000, type=int)
    parser.add_argument("--summary_flush", default=10, type=int)   #what does this exactly do? 

    # HP opt params
    #parser.add_argument("--doubleQ", default=True, type=bool,help="Whether to use a  DoubleQ agent")
    parser.add_argument("--custom_last_layer", default=True, type=bool)
    parser.add_argument("--custom_layer_init", default=1.0,type=    float)
    parser.add_argument("--initial_collect_steps", default=5000, type=int)
    #parser.add_argument("--loss_function", default="element_wise_huber_loss", type=str)
    parser.add_argument("--num_heads", default=4, type=int)
    parser.add_argument("--normalize_env", default=False, type=bool)  
    parser.add_argument('--custom_lr_schedule',default="No",type=str,help = "whether to use a custom LR schedule")
    #parser.add_argument("--epsilon_greedy", default=0.3, type=float)
    #parser.add_argument("--target_update_period", default=1000, type=int)
    parser.add_argument("--rate", default=0.1, type=float)  # dropout rate  (might be not used depending on the q network)  #Setting this to 0.0 somehow break the code. Not relevant tho just select a network without dropout
    parser.add_argument("--gradient_clipping", default=True, type=bool)
    parser.add_argument("--replay_buffer_max_length", default=1001, type=int)
    #parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--encoder_type", default=3, type=int,help="Which Type of encoder is used for the model")
    parser.add_argument("--layer_type", default=3, type=int,help="Which Type of layer is used for the encoder")
    #parser.add_argument("--target_update_tau", default=1, type=float)
    #parser.add_argument("--gamma", default=0.99, type=float)


    
    args = parser.parse_args()
    global_step = tf.compat.v1.train.get_or_create_global_step()
    
    baseEnv = gym.make(args.env)
    
    eval_tf_env = tf_py_environment.TFPyEnvironment(PyhistoryWrapper(suite_gym.load(args.env),args.max_horizon,args.atari))
        #[lambda: PyhistoryWrapper(suite_gym.load(args.env),args.max_horizon,args.atari)] * args.num_parallel)
    tf_env = tf_py_environment.TFPyEnvironment(
        parallel_py_environment.ParallelPyEnvironment(
            #[lambda: PyhistoryWrapper(suite_gym.load(args.env),args.max_horizon,args.atari)] * args.num_parallel))
            [lambda: PyhistoryWrapper(suite_gym.load(args.env),args.max_horizon,args.atari)] * args.num_parallel))
    
    
    actor_net = actor_distribution_network.ActorDistributionNetwork(
        tf_env.observation_spec(),
        tf_env.action_spec(),
        fc_layer_params=(200, 100),
        activation_fn=tf.keras.activations.tanh)
    value_net = value_network.ValueNetwork(
        tf_env.observation_spec(),
        fc_layer_params=(200, 100),
        activation_fn=tf.keras.activations.tanh)
    
    
    
    actor_net = QTransformer(
        tf_env.observation_spec(),
        baseEnv.action_space.n,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads, 
        dff=args.dff,
        rate = args.rate,
        encoderType = args.encoder_type,
        enc_layer_type=args.layer_type,
        max_horizon=args.max_horizon,
        custom_layer = args.custom_layer_init, 
        custom_last_layer = args.custom_last_layer)

    if args.custom_lr_schedule == "Transformer":    # builds a lr schedule according to the original usage for the transformer
        learning_rate = CustomSchedule(args.d_model,int(args.num_iterations/10))
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    elif args.custom_lr_schedule == "Transformer_low":    # builds a lr schedule according to the original usage for the transformer
        learning_rate = CustomSchedule(int(args.d_model/2),int(args.num_iterations/10)) # --> same schedule with lower general lr
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    elif args.custom_lr_schedule == "Linear": 
        lrs = LinearCustomSchedule(learning_rate,args.num_iterations)
        optimizer = tf.keras.optimizers.Adam(lrs, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    else:
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=args.learning_rate)




    tf_agent = ppo_clip_agent.PPOClipAgent(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        optimizer,
        actor_net=actor_net,
        value_net=value_net,
        entropy_regularization=0.0,
        importance_ratio_clipping=0.2,
        normalize_observations=False,
        normalize_rewards=False,
        use_gae=True,
        num_epochs=args.num_epochs,
        debug_summaries=args.debug_summaries,
        summarize_grads_and_vars=args.summarize_grads_and_vars,
        train_step_counter=global_step)
    tf_agent.initialize()


    
    train_eval(
    args.output_dir,
    0, # ??
    # TODO(b/127576522): rename to policy_fc_layers.
    tf_agent,
    eval_tf_env,
    tf_env,
    # Params for collect
    args.num_iterations,
    args.collect_episodes_per_iteration,
    args.num_parallel,
    args.replay_buffer_max_length,  # Per-environment
    # Params for train
    args.num_epochs,
    args.learning_rate,
    # Params for eval
    args.num_eval_episodes,
    args.eval_interval,
    # Params for summaries and logging
    args.checkpoint_interval,
    args.checkpoint_interval,
    args.checkpoint_interval,
    args.log_interval,
    args.summary_interval,
    args.summary_flush,
    args.debug_summaries,
    args.summarize_grads_and_vars,
    args.run_graph_mode,
    None)
    

    
    pickle.dump(args,open(args.output_dir + "/training_args.p","wb"))
    print("Successfully trained and evaluation.")
    

if __name__ == '__main__':
  main()