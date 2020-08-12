# GPU stuff
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"

import tensorflow as tf
import optuna

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

from tf_agents.environments import tf_environment
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.networks import q_network
from tf_agents.utils.common import element_wise_squared_loss ,element_wise_huber_loss

from train_eval import *
from utils import *


class objective(object):
    def __init__(self,argus):
        # Hold this implementation specific arguments as the fields of the class.
	    self.args = argus
	    self.trial = 0

    def __call__(self, trial):
	    outs = []
	    self.trial+= 1 
	    # parameters
	    # grey at the end were filtered by the first study 

	    batch_size = 32
	    custom_last_layer = False
	    doubleQ = True
	    encoder_type = 3
	    epsilon_greedy = 0.1
	    gamma = 0.95
	    gradient_clipping = True
	    normalize_env = False
	    rate = 0.1
	    replay_buffer_max_length = 100000
	    target_update_tau = 1
	    learning_rate = 1e-4
	    initial_collect_steps = 500

	    num_heads = trial.suggest_categorical('num_heads', [ 4])
	    custom_layer_init = trial.suggest_categorical('custom_layer_init', [ 1]) 
	    custom_lr_schedule = trial.suggest_categorical('custom_lr_schedule', ["No","Transformer"]) #["Linear","No","Transformer","Transformer_low"]
	    layer_type = trial.suggest_categorical('layer_type', [3]) # [1,2,3,5,6,7]		
	    loss_function = trial.suggest_categorical('loss_function', ["element_wise_huber_loss"])     
	    target_update_period = trial.suggest_categorical('target_update_period', [5]) #[5, 10,15]


	    for x in range(self.args.n_trys):

		    global_step = tf.Variable(0, trainable=False,dtype="int64",name= "global_step")
		    baseEnv = gym.make(self.args.env)
		    env = suite_gym.load(self.args.env)
		    eval_env = suite_gym.load(self.args.env)
		    if normalize_env == True:
		        env = NormalizeWrapper(env,self.args.approx_env_boundaries,self.args.env)
		        eval_env = NormalizeWrapper(eval_env,self.args.approx_env_boundaries,self.args.env)
		    env = PyhistoryWrapper(env,self.args.max_horizon,self.args.atari)
		    eval_env = PyhistoryWrapper(eval_env,self.args.max_horizon,self.args.atari)
		    tf_env = tf_py_environment.TFPyEnvironment(env)
		    eval_tf_env = tf_py_environment.TFPyEnvironment(eval_env)

		    q_net = QTransformer(
		        tf_env.observation_spec(),
		        baseEnv.action_space.n,
		        num_layers=self.args.num_layers,
		        d_model=self.args.d_model,
		        num_heads=num_heads, 
		        dff=self.args.dff,
		        rate = rate,
		        encoderType = encoder_type,
		        enc_layer_type=layer_type,
		        max_horizon=self.args.max_horizon,
		        custom_layer = custom_layer_init, 
		        custom_last_layer = custom_last_layer)

		    if custom_lr_schedule == "Transformer":    # builds a lr schedule according to the original usage for the transformer
		        learning_rate = CustomSchedule(self.args.d_model,int(self.args.num_iterations/10))
		        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

		    elif custom_lr_schedule == "Transformer_low":    # builds a lr schedule according to the original usage for the transformer
		        learning_rate = CustomSchedule(int(self.args.d_model/2),int(self.args.num_iterations/10)) # --> same schedule with lower general lr
		        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

		    elif custom_lr_schedule == "Linear": 
		        lrs = LinearCustomSchedule(learning_rate,self.args.num_iterations)
		        optimizer = tf.keras.optimizers.Adam(lrs, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

		    else:
		        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

		    if loss_function == "element_wise_huber_loss" :
		    	lf = element_wise_huber_loss
		    elif loss_function == "element_wise_squared_loss":
		    	lf = element_wise_squared_loss

		    if doubleQ == False:          # global step count
		        agent = dqn_agent.DqnAgent(
		            tf_env.time_step_spec(),
		            tf_env.action_spec(),
		            q_network=q_net,
		            epsilon_greedy=epsilon_greedy,
		            target_update_tau=target_update_tau,
		            target_update_period=target_update_period,
		            optimizer=optimizer,
		            gamma=gamma,
		            td_errors_loss_fn = lf,
		            reward_scale_factor=self.args.reward_scale_factor,
		            gradient_clipping=gradient_clipping,
		            debug_summaries=self.args.debug_summaries,
		            summarize_grads_and_vars=self.args.summarize_grads_and_vars,
		            train_step_counter=global_step)
		    else:
		        agent = dqn_agent.DdqnAgent(
		            tf_env.time_step_spec(),
		            tf_env.action_spec(),
		            q_network=q_net,
		            epsilon_greedy=epsilon_greedy,
		            target_update_tau=target_update_tau,
		            target_update_period=target_update_period,
		            optimizer=optimizer,
		            gamma=gamma,
		            td_errors_loss_fn = lf,
		            reward_scale_factor=self.args.reward_scale_factor,
		            gradient_clipping=gradient_clipping,
		            debug_summaries=self.args.debug_summaries,
		            summarize_grads_and_vars=self.args.summarize_grads_and_vars,
		            train_step_counter=global_step)
		    agent.initialize()


		    metric = train_eval_2(
		    root_dir = os.path.join(self.args.output_dir, str(self.trial) + "_" + str(x)),
		    num_eval_episodes = self.args.num_eval_episodes,
		    tf_env= tf_env,
		    eval_tf_env = eval_tf_env,
		    agent = agent, 
		    eval_interval = self.args.eval_interval,
		    summary_interval = self.args.summary_interval,
		    num_iterations=self.args.num_iterations,
		    initial_collect_steps= initial_collect_steps,
		    collect_steps_per_iteration= self.args.collect_steps_per_iteration,
		    replay_buffer_capacity= replay_buffer_max_length,
		    train_steps_per_iteration=self.args.train_steps_per_iteration,
		    batch_size = batch_size,
		    use_tf_functions=self.args.run_graph_mode,
		    log_interval = self.args.log_interval,
		    global_step = global_step)

		    outs.append(metric)

	    return -np.mean(outs) # since we are minimizing we need to take the negative reward sum

	


def main():
    parser = argparse.ArgumentParser()

    ## Set Parameters
    parser.add_argument("--output_dir", default=None, type=str, required=True,help="The output directory where the model stats and checkpoints will be written.")
    parser.add_argument("--num_eval_episodes", default=5, type=int)
    parser.add_argument("--eval_interval", default=999999, type=int)
    parser.add_argument("--env", default=None, type=str, required=True,help="The environment to train the agent on")
    parser.add_argument("--approx_env_boundaries", default=True, type=bool,help="Whether to get the env boundaries approximately")
    parser.add_argument("--max_horizon", default=5, type=int)
    parser.add_argument("--atari", default=False, type=bool, help = "Gets some data Types correctly")	
    parser.add_argument("--reward_scale_factor", default=1.0, type=float)
    parser.add_argument("--n_trials", default=12, type=int)
    parser.add_argument("--n_trys", default=2, type=int)
    parser.add_argument("--parallel_opt", default=1, type=int)

    ##transformer parameters
    parser.add_argument("--d_model", default=64, type=int)
    parser.add_argument("--num_layers", default=3, type=int)
 	# nheads is a HP
    parser.add_argument("--dff", default=256, type=int)

    ##Training parameters
    parser.add_argument('--num_iterations', type=int, default=75000,help="steps in the env")
    parser.add_argument('--num_iparallel', type=int, default=1,help="how many envs should run in parallel")
    parser.add_argument("--collect_steps_per_iteration", default=1, type=int)
    parser.add_argument("--train_steps_per_iteration", default=1, type=int)
    parser.add_argument("--debug_summaries", default=False, type=bool)
    parser.add_argument("--summarize_grads_and_vars", default=False, type=bool)

    ##Eval parameters

    ## Other parameters
    parser.add_argument("--log_interval", default=1000, type=int)
    parser.add_argument("--summary_interval", default=1000, type=int)
    parser.add_argument("--run_graph_mode", default=True, type=float)

    args = parser.parse_args()


    try:
    # Create target Directory
    	os.mkdir(args.output_dir)
    	print("Directory " , args.output_dir ,  " Created ") 
    except FileExistsError: 
    	print("Directory " , args.output_dir ,  " already exists")


    search_space = {
    'custom_layer_init': [1],
    'custom_lr_schedule': ["No"],
    'layer_type' : [6],
    'loss_function': ["element_wise_squared_loss"], #"element_wise_squared_loss"
    'target_update_period': [5,10],
    'num_heads': [2,4],
	}

    study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space))
    study.optimize(objective(args),n_jobs=args.parallel_opt , n_trials=args.n_trials)
 
    pickle.dump(study,open(args.output_dir + "/study_save.p","wb"))
    pickle.dump(args,open(args.output_dir + "/opt_training_args.p","wb"))
    print("optimization finnished.")

if __name__ == "__main__":
    main()
