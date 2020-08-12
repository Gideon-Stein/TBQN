# Script to train a RL agent on baseline environments and save the results

from __future__ import absolute_import, division, print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

 
from tf_agents.drivers import dynamic_step_driver

from env_wrappers import *
from q_transformer import *
from utils import *

import tensorflow as tf
import numpy as np
import os 
import pickle


from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.policies import policy_saver
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tensorflow.python.framework import tensor_spec
from tf_agents.trajectories import trajectory
from tf_agents.trajectories.trajectory import *
from tf_agents.utils import common 

def compute_avg_return(environment, policy, num_episodes=10):

    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


def train_eval(
    root_dir,
    tf_env,
    eval_tf_env,
    agent,
    num_iterations,
    initial_collect_steps,
    collect_steps_per_iteration,
    replay_buffer_capacity,
    train_steps_per_iteration,
    batch_size,
    use_tf_functions,
    num_eval_episodes,
    eval_interval,
    train_checkpoint_interval,
    policy_checkpoint_interval,
    rb_checkpoint_interval,
    log_interval,
    summary_interval,
    summaries_flush_secs):

  """A simple train and eval for DQN."""
  root_dir = os.path.expanduser(root_dir)
  train_dir = os.path.join(root_dir, 'train')
  eval_dir = os.path.join(root_dir, 'eval')



  train_summary_writer = tf.compat.v2.summary.create_file_writer(
      train_dir, flush_millis=summaries_flush_secs * 1000)
  train_summary_writer.set_as_default()


  eval_summary_writer = tf.compat.v2.summary.create_file_writer(
      eval_dir, flush_millis=summaries_flush_secs * 1000)
  eval_metrics = [
    #tf_metrics.ChosenActionHistogram(buffer_size=num_eval_episodes),
      tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
      #tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes)
  ]


  global_step = tf.compat.v1.train.get_or_create_global_step()

  with tf.compat.v2.summary.record_if(lambda: tf.math.equal(global_step % summary_interval, 0)):

    tf_env = tf_env
    eval_tf_env = eval_tf_env

    tf_agent = agent

    train_metrics = [
        #tf_metrics.ChosenActionHistogram(),
        tf_metrics.NumberOfEpisodes(),
        tf_metrics.EnvironmentSteps(),
        tf_metrics.AverageReturnMetric(buffer_size=1),
        #tf_metrics.AverageEpisodeLengthMetric(),
    ]

    diverged = False

    eval_policy = tf_agent.policy
    collect_policy = tf_agent.collect_policy

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=tf_agent.collect_data_spec,
        batch_size=tf_env.batch_size,
        max_length=replay_buffer_capacity)

    collect_driver = dynamic_step_driver.DynamicStepDriver(
        tf_env,
        collect_policy,
        observers=[replay_buffer.add_batch] + train_metrics,
        num_steps=collect_steps_per_iteration)

    train_checkpointer = common.Checkpointer(
        ckpt_dir=train_dir,
        agent=tf_agent,
        global_step=global_step,
        max_to_keep=1,
        metrics=metric_utils.MetricsGroup(train_metrics, 'train_metrics'))

    policy_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(train_dir, 'policy'),
        policy=eval_policy,
        max_to_keep=1,
        global_step=global_step)


    rb_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(train_dir, 'replay_buffer'),
        max_to_keep=1,
        replay_buffer=replay_buffer)


    

    train_checkpointer.initialize_or_restore()
    rb_checkpointer.initialize_or_restore()
    best_policy = -1000
    if use_tf_functions:
      # To speed up collect use common.function.
      collect_driver.run = common.function(collect_driver.run)
      tf_agent.train = common.function(tf_agent.train)

    initial_collect_policy = random_tf_policy.RandomTFPolicy(
        tf_env.time_step_spec(), tf_env.action_spec())

    #Collect initial replay data.
    dynamic_step_driver.DynamicStepDriver(
        tf_env,
        initial_collect_policy,
        observers=[replay_buffer.add_batch] + train_metrics,
        num_steps=initial_collect_steps).run()

    results = metric_utils.eager_compute(
        eval_metrics,
        eval_tf_env,
        eval_policy,
        num_episodes=num_eval_episodes,
        train_step=global_step,
        summary_writer=eval_summary_writer,
        summary_prefix='Metrics',
    )
    metric_utils.log_metrics(eval_metrics)

    time_step = None
    policy_state = collect_policy.get_initial_state(tf_env.batch_size)
    timed_at_step = global_step.numpy()
    time_acc = 0

    # Dataset generates trajectories with shape [Bx2x...]
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=batch_size,
        num_steps=2).prefetch(3)
    iterator = iter(dataset)

    def train_step():
      experience, _ = next(iterator)
      return tf_agent.train(experience)

    if use_tf_functions:
      train_step = common.function(train_step)

    for _ in range(num_iterations):
      start_time = time.time()
      time_step, policy_state = collect_driver.run(
          time_step=time_step,
          policy_state=policy_state,
      )
      for _ in range(train_steps_per_iteration):
        train_loss = train_step()
      time_acc += time.time() - start_time

      if np.isnan(train_loss.loss).any():
        diverged = True
        break
      elif np.isinf(train_loss.loss).any():
        diverged = True
        break 

      if global_step.numpy() % log_interval == 0:
        print('step = {0}, loss = {1}'.format( global_step.numpy(),train_loss.loss))

        steps_per_sec = (global_step.numpy() - timed_at_step) / time_acc
        print('{0} steps/sec'.format( steps_per_sec))
        tf.compat.v2.summary.scalar(
            name='global_steps_per_sec', data=steps_per_sec, step=global_step)
        timed_at_step = global_step.numpy()
        time_acc = 0

      for train_metric in train_metrics:
        train_metric.tf_summaries(
            train_step=global_step, step_metrics=train_metrics[:2])

      if global_step.numpy() % train_checkpoint_interval == 0:
        train_checkpointer.save(global_step=global_step.numpy())


      if global_step.numpy() % rb_checkpoint_interval == 0:
        rb_checkpointer.save(global_step=global_step.numpy())

      if global_step.numpy() % eval_interval == 0:
        results = metric_utils.eager_compute(
            eval_metrics,
            eval_tf_env,
            eval_policy,
            num_episodes=num_eval_episodes,
            train_step=global_step,
            summary_writer=eval_summary_writer,
            summary_prefix='Metrics',
        )

        if results["AverageReturn"].numpy() > best_policy:
            print("New best policy found")
            print(results["AverageReturn"].numpy())
            best_policy = results["AverageReturn"].numpy()
            policy_checkpointer.save(global_step=global_step.numpy())

        metric_utils.log_metrics(eval_metrics)
    return train_loss




def train_eval_2(
root_dir,
tf_env,
eval_tf_env,
agent,
num_iterations,
initial_collect_steps,
collect_steps_per_iteration,
num_eval_episodes,
eval_interval,
summary_interval,
replay_buffer_capacity,
train_steps_per_iteration,
batch_size,
use_tf_functions,
log_interval,
global_step):

  """A simple train and eval for DQN."""

  root_dir = os.path.expanduser(root_dir)
  train_dir = os.path.join(root_dir, 'train')
  eval_dir = os.path.join(root_dir, 'eval')

  train_summary_writer = tf.compat.v2.summary.create_file_writer(train_dir)
  train_summary_writer.set_as_default()
  eval_summary_writer = tf.compat.v2.summary.create_file_writer(eval_dir)
  #q = []

  eval_metrics = [
      tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
  ]

  train_metrics = [
      #tf_metrics.ChosenActionHistogram(),
      tf_metrics.NumberOfEpisodes(),
      tf_metrics.AverageReturnMetric(),
  ]

  diverged = False

  global_step = global_step

  with tf.compat.v2.summary.record_if(lambda: tf.math.equal(global_step % summary_interval, 0)):

    tf_env = tf_env
    eval_tf_env = eval_tf_env
    tf_agent = agent
    eval_policy = tf_agent.policy
    collect_policy = tf_agent.collect_policy

      
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=tf_agent.collect_data_spec,
        batch_size=tf_env.batch_size,
        max_length=replay_buffer_capacity)

    collect_driver = dynamic_step_driver.DynamicStepDriver(
        tf_env,
        collect_policy,
        observers=[replay_buffer.add_batch]+ train_metrics,
        num_steps=collect_steps_per_iteration)


    best_policy = -1000
    if use_tf_functions:
      # To speed up collect use common.function.
      collect_driver.run = common.function(collect_driver.run)
      tf_agent.train = common.function(tf_agent.train)

    initial_collect_policy = random_tf_policy.RandomTFPolicy(
        tf_env.time_step_spec(), tf_env.action_spec())

    # Collect initial replay data.
    dynamic_step_driver.DynamicStepDriver(
        tf_env,
        initial_collect_policy,
        observers=[replay_buffer.add_batch] + train_metrics,
        num_steps=initial_collect_steps).run()

    results = metric_utils.eager_compute(
            eval_metrics,
            eval_tf_env,
            eval_policy,
            num_episodes=num_eval_episodes,
            train_step=global_step,
            summary_writer=eval_summary_writer,
            summary_prefix='Metrics',
        )

    time_step = None
    policy_state = collect_policy.get_initial_state(tf_env.batch_size)
    timed_at_step = global_step.numpy()
    time_acc = 0

    # Dataset generates trajectories with shape [Bx2x...]
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=batch_size,
        num_steps=2).prefetch(3)
    iterator = iter(dataset)

    def train_step():
      experience, _ = next(iterator)
      return tf_agent.train(experience)

    if use_tf_functions:
      train_step = common.function(train_step)

    for _ in range(num_iterations):

      start_time = time.time()
      time_step, policy_state = collect_driver.run(
          time_step=time_step,
          policy_state=policy_state,
      )
      for _ in range(train_steps_per_iteration):
        train_loss = train_step()
      time_acc += time.time() - start_time

      if np.isnan(train_loss.loss).any():
        diverged = True
        break
      elif np.isinf(train_loss.loss).any():
        diverged = True
        break 

      if global_step.numpy() % log_interval == 0:
        print('step = {0}, loss = {1}'.format( global_step.numpy(),train_loss.loss))

        steps_per_sec = (global_step.numpy() - timed_at_step) / time_acc
        print('{0} steps/sec'.format( steps_per_sec))
        tf.compat.v2.summary.scalar(
            name='global_steps_per_sec', data=steps_per_sec, step=global_step)
        timed_at_step = global_step.numpy()
        time_acc = 0


      for train_metric in train_metrics:
          train_metric.tf_summaries(
              train_step=global_step, step_metrics=train_metrics)


      if global_step.numpy() % eval_interval == 0:
          results = metric_utils.eager_compute(
              eval_metrics,
              eval_tf_env,
              eval_policy,
              num_episodes=num_eval_episodes,
              train_step=global_step,
              summary_writer=eval_summary_writer,
              summary_prefix='Metrics',
          )
          #q.append(collect_q (eval_tf_env,agent,1)[0])




    #pickle.dump(q,open(root_dir + "/q.p","wb"))
    return train_metrics[1].result().numpy()     


