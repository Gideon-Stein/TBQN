import tensorflow as tf
import imageio
import base64
import IPython
import keras
import matplotlib.pyplot as plt


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=2000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


class LinearCustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr,nSteps):
        super(LinearCustomSchedule, self).__init__()

        self.initial_lr = initial_lr
        self.decay = tf.cast(self.initial_lr, tf.float32)
        self.warmup_steps = nSteps
        self.decay = tf.cast(initial_lr/nSteps, tf.float32)
        print(self.decay)

    def __call__(self, step):
        return self.initial_lr - (step*self.decay)


class glorot_adapted(tf.keras.initializers.Initializer):
    def __init__(self,layerN, alpha=0.5):
        self.glorot = tf.keras.initializers.GlorotUniform()
        self.alpha = alpha
        self.layerN = layerN
    def __call__(self, shape, dtype=None):
        self.alpha = tf.cast(self.alpha,dtype=dtype)
        self.layerN = tf.cast(self.layerN,dtype=dtype)
        
        return self.glorot(shape,dtype) * (self.alpha / tf.math.sqrt(self.layerN))
    
    def get_config(self):  # To support serialization
        return {'alpha': self.alpha,'layerN': self.layerN}


def count_weights(net):
    count = 0
    for x in (net.trainable_weights):
        Lcount = 1
        for y in range( len(x.shape)):
            Lcount = x.shape[y] * Lcount
        count += Lcount
    print("Trainable weights: " +  str(count))


def collect_q (environment,agent,n):
    q_collection = []
    time_step = environment.current_time_step()
    for x in range(n):
        action_step = agent.policy.action(time_step)
        time_step = environment.step(action_step.action.numpy()[0])
        Q = agent._q_network(time_step.observation)
        q_collection.append(Q[0].numpy().tolist())
    return q_collection


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
    
def compute_avg_return_py(environment, policy, num_episodes=10):

    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)

            time_step = environment.step(action_step.action.numpy()[0])
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return


def embed_mp4(filename):
  """Embeds an mp4 file in the notebook."""
  video = open(filename,'rb').read()
  b64 = base64.b64encode(video)
  tag = '''
  <video width="640" height="480" controls>
    <source src="data:video/mp4;base64,{0}" type="video/mp4">
  Your browser does not support the video tag.
  </video>'''.format(b64.decode())
  return IPython.display.HTML(tag)


def create_policy_eval_video(policy,eval_env, filename, num_episodes=5, fps=60):
  filename = filename + ".mp4"
  with imageio.get_writer(filename, fps=fps) as video:
    for _ in range(num_episodes):
      time_step = eval_env.reset()
      video.append_data(eval_env.render())
      while not time_step.is_last():
        action_step = policy.action(time_step)
        time_step = eval_env.step(action_step.action.numpy()[0])
        video.append_data(eval_env.render())
    return filename

class gr_output(tf.keras.layers.Layer):
    def __init__(self, units):
        super(gr_output, self).__init__()
        self.W = tf.keras.layers.Dense(units,bias_initializer="glorot_uniform")

    def call(self, inp):
        x = inp[0]
        y = inp[1]
        out = self.W(x)
        out = tf.keras.activations.sigmoid(out)
        out = tf.keras.layers.Multiply()([out,y])
        out = out + x
        return out


class gru(tf.keras.layers.Layer):
    def __init__(self, units):
        super(gru, self).__init__()
        self.WR = tf.keras.layers.Dense(units)
        self.UR = tf.keras.layers.Dense(units)
        self.WZ = tf.keras.layers.Dense(units)
        self.UZ = tf.keras.layers.Dense(units,bias_initializer="glorot_uniform")
        self.WG = tf.keras.layers.Dense(units,bias_initializer="glorot_uniform")
        self.UG = tf.keras.layers.Dense(units)
        
    def call(self, inp):
        x = inp[0]
        y = inp[1]
        z = self.WZ(y) + self.UZ(x)
        r = self.WR(y) + self.UR(x)
        rx = tf.keras.layers.Multiply()([r,x])
        h = tf.math.tanh(self.WG(y) + self.UG(rx))
        g = tf.keras.layers.Multiply()([(1-z),x]) + tf.keras.layers.Multiply()([(z),h])
        return g


def plot_self_attention(attention, layer):
    fig = plt.figure(figsize=(16, 8))
  
    attention = tf.squeeze(attention, axis=0)
    for head in range(attention.shape[0]):
        ax = fig.add_subplot(2, 4, head+1)

        # plot the attention weights
        ax.matshow(attention[head][:, :], cmap='viridis')

        ax.set_xlabel('Head {}'.format(head+1))

    plt.tight_layout()
    plt.show()