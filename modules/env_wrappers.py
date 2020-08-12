from tf_agents.environments.tf_wrappers import TFEnvironmentBaseWrapper
from tf_agents.environments.wrappers import PyEnvironmentBaseWrapper
import numpy as np 
import tensorflow as tf
from tf_agents.trajectories.trajectory import *
from tensorflow.python.framework import tensor_spec
from tf_agents.utils import common
import os 
import random

#@common.function()
class TFhistoryWrapper(TFEnvironmentBaseWrapper):
    """ 
    Wrapps an env and saves the last states of the env. It returns a concatenated step with history instead

    """
    def __init__(self,env,history_n):
        
        super(TFhistoryWrapper, self).__init__(env)
        self.history_n = history_n
        self.history = np.empty((0,0))
        self.current = np.empty((0,0))
        self.reset()
        
    #@common.function
    def time_step_spec(self):
        timestepspec = super(TFhistoryWrapper,self).time_step_spec()
        base = timestepspec.observation
        maxi = np.tile( base.maximum,self.history_n)
        mini = np.tile( base.minimum,self.history_n)
        new_obs = tensor_spec.BoundedTensorSpec(shape=maxi.shape,dtype=base.dtype,name=base.name,minimum=mini,maximum=maxi)
        return ts.TimeStep(timestepspec.step_type, timestepspec.reward, timestepspec.discount,observation = new_obs)

    #@common.function
    def observation_spec(self):
        base = super(TFhistoryWrapper,self).observation_spec()
        maxi = np.tile( base.maximum,self.history_n)
        mini = np.tile( base.minimum,self.history_n)
        new_obs = tensor_spec.BoundedTensorSpec(shape=maxi.shape,dtype=base.dtype,name=base.name,minimum=mini,maximum=maxi)
        return new_obs

    #@common.function
    def current_time_step(self):
        return self.current
        
    
    #@common.function
    def reset(self):
        self.history = np.empty((0,0))
        time_step = self._env.reset()
        obs = tf.cast(time_step.observation,dtype="float32")
        filler = tf.fill((self.history_n-self.history.shape[0]-1,obs.shape[0],obs.shape[1]),0.)
        filled_obs = tf.concat([tf.expand_dims(obs,0),filler],0)
        filled_obs = tf.reshape(filled_obs,[1,filled_obs.shape[0]*filled_obs.shape[2]])
        filled_step = ts.TimeStep(time_step.step_type, time_step.reward,
                              time_step.discount,observation = filled_obs)
        self.history = obs
        self.current = filled_step
        return filled_step
     
    @common.function
    def step (self,action):
        assert self.history.shape[0] != 0, "please reset the environment before using it"
        time_step = self._step(action)
        obs = tf.cast(time_step.observation,dtype="float32")
        filler = tf.fill((self.history_n-self.history.shape[0]-1,obs.shape[0],obs.shape[1]),0.)
        obs2 = tf.concat((self.history,obs),0)
        filled_obs = tf.concat((tf.expand_dims(obs2,1),filler),0)
        filled_obs = tf.reshape(filled_obs,[1,filled_obs.shape[0]*filled_obs.shape[2]])
        filled_step = ts.TimeStep(time_step.step_type, time_step.reward,
                              time_step.discount,observation = filled_obs)
        self.history = tf.concat([self.history,obs],0)
        if self.history.shape[0] > self.history_n-1:
            self.history = self.history[1:,:]
        self.current = filled_step
        return filled_step      

class PyhistoryWrapper(PyEnvironmentBaseWrapper):
    """ 
    Wrapps an env and saves the last states of the env. It returns a concatenated step with history instead
    """
    def __init__(self,env,history_n,atari):
        
        super(PyhistoryWrapper, self).__init__(env)
        self.history_n = history_n
        self.history = np.empty((0,0))
        self.isAtari = atari
        self.wasLast = False
        self.reset()
        
    def time_step_spec(self):
        timestepspec = self._env.time_step_spec()
        obs = timestepspec.observation
        new_obs = self.build_obs_spec(obs)
        return ts.TimeStep(timestepspec.step_type, timestepspec.reward, timestepspec.discount,observation = new_obs)

    def observation_spec(self):
        base = self._env.observation_spec()
        return self.build_obs_spec(base)

    def current_time_step(self):
        return self.current
        
    
    def build_obs_spec(self,obs):
        if not self.isAtari:
            maxi = np.tile( obs.maximum,self.history_n)
            mini = np.tile( obs.minimum,self.history_n)
        else:
            maxi = np.tile( obs.maximum,self.history_n*obs.shape[0])
            mini = np.tile( obs.minimum,self.history_n*obs.shape[0])
        new_obs = tensor_spec.BoundedTensorSpec(shape=maxi.shape,dtype=tf.float32,name=obs.name,minimum=mini,maximum=maxi)
        return new_obs
    
    def _reset(self):
        self.history = np.empty((0,0))
        time_step = self._env._reset()
        if not self.isAtari:
            obs = time_step.observation.astype("float32")
            filler = np.zeros((self.history_n-self.history.shape[0]-1,obs.shape[0]),dtype="float32")
        else:
            obs = time_step.observation.astype("float32")
            filler = np.zeros((self.history_n-self.history.shape[0]-1,obs.shape[0]),dtype="float32")

        filled_obs = np.concatenate((np.expand_dims(obs,0),filler),0)
        filled_obs = np.reshape(filled_obs,(filled_obs.shape[0]*filled_obs.shape[1]))
        filled_step = ts.TimeStep(time_step.step_type, time_step.reward,
                              time_step.discount,observation = filled_obs)
        self.history = np.expand_dims(obs,0)
        self.current = filled_step
        return filled_step
    
    def _step (self,action):
        if self.wasLast == False:

            ## Todo
            # Should delete the history when its the last step. after the step is returned. Need to check how it works. Right now old history is still incorporate
            assert self.history.shape[0] != 0, "please reset the environment before using it"
            time_step = self._env.step(action)
            if time_step.is_last() == True: 
                self.wasLast = True
            if not self.isAtari:
                obs = time_step.observation.astype("float32")
                filler = np.zeros((self.history_n-self.history.shape[0]-1,obs.shape[0]),dtype="float32")
            else:
                obs = time_step.observation.astype("float32")
                filler = np.zeros((self.history_n-self.history.shape[0]-1,obs.shape[0]),dtype="float32")
            obs2 =  np.concatenate((self.history,np.expand_dims(obs,0)),0)
            filled_obs = np.concatenate((obs2,filler),0)
            filled_obs = np.reshape(filled_obs,(filled_obs.shape[0]*filled_obs.shape[1]))
            filled_step = ts.TimeStep(time_step.step_type, time_step.reward,
                                  time_step.discount,observation = filled_obs)
            self.history = np.concatenate((self.history,np.expand_dims(obs,0)),0)
            if self.history.shape[0] > self.history_n-1:
                self.history = self.history[1:]
            self.current = filled_step
            return filled_step  
        else:
            self.wasLast = False
            return self._reset()

#Normalize broken for non control. But its okay since it sucks anyways...
class NormalizeWrapper(PyEnvironmentBaseWrapper):
    def __init__(self,env,customBounds = False,env_name="MissingName"):
        self.env_name = env_name
        super(NormalizeWrapper, self).__init__(env)
        
        if customBounds:
            self.lower,self.upper = self.get_approx_bounds()
        else:
            self.upper = env.observation_spec().maximum
            self.lower = env.observation_spec().minimum
            print(env.observation_spec())
 
    def time_step_spec(self):
        timestepspec = super(NormalizeWrapper,self).time_step_spec()
        new_obs = self.observation_spec()
        return ts.TimeStep(timestepspec.step_type, timestepspec.reward, timestepspec.discount,observation = new_obs)

    def observation_spec(self):
        obs = super(NormalizeWrapper,self).observation_spec()
        maxi = tf.ones([len(self.upper)])
        mini = tf.zeros([len(self .upper)])
        new_obs = tensor_spec.BoundedTensorSpec(shape=maxi.shape,dtype=obs.dtype,
                                                name=obs.name,minimum=mini,maximum=maxi)
        return new_obs
            
        
    def get_approx_bounds(self,n=10000):
        if os.path.exists("approx_" + self.env_name+ ".npy"):
            loading = np.load("approx_" + self.env_name+ ".npy")
            return loading[0],loading[1]
        else:
            # discrete not implemented
            obs_length = self._env.observation_spec().shape[0]
            print("No approximation found. New approximation running")
            maxi = np.full((obs_length,),-np.inf)
            mini = np.full((obs_length,),np.inf)
            maxiac = int(self._env.action_spec().maximum)
            miniac = int(self._env.action_spec().minimum)
            for x in range (n):
                action = random.randint(miniac,maxiac)
                a  = self._env.step(action).observation
                for x in range(len(a)):
                    if a[x]> maxi[x]:
                        maxi[x]  = a[x]
                    if a[x]< mini[x]:
                        mini[x]  = a[x]
            np.save("approx_" + self.env_name,np.array([mini,maxi]))
            print("New approx: " + str(mini) + "/" + str(maxi) )
            return mini,maxi
        
    def build_time_step(self,timestep):
        obs = timestep.observation
        return ts.TimeStep(timestep.step_type, timestep.reward, timestep.discount,
        self.normalize_time_step(obs))
        
    def normalize_time_step(self,obs):    
        out =   (obs - self.lower) / (self.upper -self.lower)
        return out.astype("float32")

    def _step(self,action):
        return self.build_time_step(self._env.step(action))
    
    def _reset(self):
        timestep = self._env.reset()
        return ts.TimeStep(timestep.step_type, timestep.reward, timestep.discount,
        self.normalize_time_step(timestep.observation))