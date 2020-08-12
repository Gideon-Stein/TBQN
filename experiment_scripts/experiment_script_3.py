# GPU stuff
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"  

import tensorflow as tf

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


import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../modules/')


import argparse
import pickle
import gym

from tf_agents.environments import tf_environment
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.networks import q_network

from tf_agents.utils.common import element_wise_squared_loss, element_wise_huber_loss

from train_eval import *
from utils import *



def main():
    parser = argparse.ArgumentParser()

    ## Essential parameters
    parser.add_argument("--output_dir", default=None, type=str, required=True,help="The output directory where the model stats and checkpoints will be written.")
    parser.add_argument("--env", default=None, type=str, required=True,help="The environment to train the agent on")
    parser.add_argument("--approx_env_boundaries", default=False, type=bool,help="Whether to get the env boundaries approximately")  
    parser.add_argument("--max_horizon", default=5, type=int)
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
    parser.add_argument('--num_iterations', type=int, default=150000,help="steps in the env")
    parser.add_argument('--num_iparallel', type=int, default=1,help="how many envs should run in parallel")
    parser.add_argument("--collect_steps_per_iteration", default=1, type=int)
    parser.add_argument("--train_steps_per_iteration", default=1, type=int)


    ## Other parameters
    parser.add_argument("--num_eval_episodes", default=10, type=int)
    parser.add_argument("--eval_interval", default=1000, type=int)
    parser.add_argument("--log_interval", default=1000, type=int)
    parser.add_argument("--summary_interval", default=1000, type=int)
    parser.add_argument("--run_graph_mode", default=True, type=bool)
    parser.add_argument("--checkpoint_interval", default=10000, type=int)
    parser.add_argument("--summary_flush", default=10, type=int)   #what does this exactly do? 

    # HP opt params
    parser.add_argument("--doubleQ", default=True, type=bool,help="Whether to use a  DoubleQ agent")
    parser.add_argument("--custom_last_layer", default=False, type=bool)
    parser.add_argument("--custom_layer_init", default=1,type=    float)
    parser.add_argument("--initial_collect_steps", default=500, type=int)
    parser.add_argument("--loss_function", default="element_wise_squared_loss", type=str)
    parser.add_argument("--num_heads", default=4, type=int)
    parser.add_argument("--normalize_env", default=False, type=bool)  
    parser.add_argument('--custom_lr_schedule',default="No",type=str,help = "whether to use a custom LR schedule")
    parser.add_argument("--epsilon_greedy", default=0.1, type=float)
    parser.add_argument("--target_update_period", default=10, type=int)
    parser.add_argument("--rate", default=0.1, type=float)  # dropout rate  (might be not used depending on the q network)  #Setting this to 0.0 somehow break the code. Not relevant tho just select a network without dropout
    parser.add_argument("--gradient_clipping", default=True, type=bool)
    parser.add_argument("--replay_buffer_max_length", default=100000, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--encoder_type", default=3, type=int,help="Which Type of encoder is used for the model")
    parser.add_argument("--layer_type", default=3, type=int,help="Which Type of layer is used for the encoder")
    parser.add_argument("--target_update_tau", default=1, type=float)
    parser.add_argument("--gamma", default=0.95, type=float)


    
    args = parser.parse_args()
    # List of encoder modules which we can use to change encoder based on a variable 
    global_step = tf.compat.v1.train.get_or_create_global_step()

    baseEnv = gym.make(args.env)
    env = suite_gym.load(args.env)
    eval_env = suite_gym.load(args.env)
    if args.normalize_env == True:
        env = NormalizeWrapper(env,args.approx_env_boundaries,args.env)
        eval_env = NormalizeWrapper(eval_env,args.approx_env_boundaries,args.env)
    env = PyhistoryWrapper(env,args.max_horizon,args.atari)
    eval_env = PyhistoryWrapper(eval_env,args.max_horizon,args.atari)
    tf_env = tf_py_environment.TFPyEnvironment(env)
    eval_tf_env = tf_py_environment.TFPyEnvironment(eval_env)


    q_net = QTransformer(
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

    if args.loss_function == "element_wise_huber_loss" :
        lf = element_wise_huber_loss
    elif args.loss_function == "element_wise_squared_loss":
        lf = element_wise_squared_loss




    if args.doubleQ == False:          # global step count
        agent = dqn_agent.DqnAgent(
            tf_env.time_step_spec(),
            tf_env.action_spec(),
            q_network=q_net,
            epsilon_greedy=args.epsilon_greedy,
            #boltzmann_temperature = 1,
            target_update_tau=args.target_update_tau,
            target_update_period=args.target_update_period,
            td_errors_loss_fn =lf,
            optimizer=optimizer,
            gamma=args.gamma,
            reward_scale_factor=args.reward_scale_factor,
            gradient_clipping=args.gradient_clipping,
            debug_summaries=args.debug_summaries,
            summarize_grads_and_vars=args.summarize_grads_and_vars,
            train_step_counter=global_step)
    else:
        agent = dqn_agent.DdqnAgent(
            tf_env.time_step_spec(),
            tf_env.action_spec(),
            q_network=q_net,
            epsilon_greedy=args.epsilon_greedy,
            #boltzmann_temperature = 1,
            target_update_tau=args.target_update_tau,
            td_errors_loss_fn = lf,
            target_update_period=args.target_update_period,
            optimizer=optimizer,
            gamma=args.gamma,
            reward_scale_factor=args.reward_scale_factor,
            gradient_clipping=args.gradient_clipping,
            debug_summaries=args.debug_summaries,
            summarize_grads_and_vars=args.summarize_grads_and_vars,
            train_step_counter=global_step)
    agent.initialize()


    count_weights(q_net)


    train_eval(
    root_dir=args.output_dir,
    tf_env= tf_env,
    eval_tf_env = eval_tf_env,
    agent = agent, 
    num_iterations=args.num_iterations,
    initial_collect_steps= args.initial_collect_steps,
    collect_steps_per_iteration= args.collect_steps_per_iteration,
    replay_buffer_capacity= args.replay_buffer_max_length,
    train_steps_per_iteration=args.train_steps_per_iteration,
    batch_size = args.batch_size,
    use_tf_functions=args.run_graph_mode,
    num_eval_episodes= args.num_eval_episodes,
    eval_interval= args.eval_interval,
    train_checkpoint_interval=args.checkpoint_interval,
    policy_checkpoint_interval=args.checkpoint_interval,
    rb_checkpoint_interval=args.checkpoint_interval,
    log_interval = args.log_interval,
    summary_interval=args.summary_interval,
    summaries_flush_secs=args.summary_flush
    )


    pickle.dump(args,open(args.output_dir + "/training_args.p","wb"))
    print("Successfully trained and evaluation.")

if __name__ == "__main__":
    main()
