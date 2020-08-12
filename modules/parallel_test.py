
from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

from absl import app
from absl import flags
from absl import logging

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.environments import suite_atari
from tf_agents.environments import parallel_py_environment

def train_eval(root_dir,env_name,env_load_fn,num_parallel_environments):
	global_step = tf.compat.v1.train.get_or_create_global_step()

	print("oof")
	eval_py_env = parallel_py_environment.ParallelPyEnvironment(
		[lambda: env_load_fn(env_name)] * num_parallel_environments)
	print("oof") 
train_eval("out",env_name="Breakout-v0",env_load_fn=suite_atari.load,num_parallel_environments=2)


