# coding=utf-8
# Copyright 2018 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Various networks for Jax Dopamine agents."""

from dopamine.discrete_domains import atari_lib
from dopamine.discrete_domains import gym_lib
from flax import nn
import gin
import jax
import jax.numpy as jnp
import numpy as onp
from jax import random


gin.constant('jax_networks.LUNALANDER_OBSERVATION_DTYPE', jnp.float64)
gin.constant('jax_networks.MOUNTAINCAR_OBSERVATION_DTYPE', jnp.float64)

#---------------------------------------------------------------------------------------------------------------------

@gin.configurable
class NoisyNetwork(nn.Module):
  def apply(self, x, features, bias=True, kernel_init=None):
    def sample_noise(shape):
      #tf.random_normal
      noise = jax.random.normal(random.PRNGKey(0),shape)
      ##noise = jax.random.normal(shape)
      return noise
    def f(x):
      return jnp.multiply(jnp.sign(x), jnp.power(jnp.abs(x), 0.5))
    # Initializer of \mu and \sigma 
   
    def mu_init(key, shape):
        low = -1*1/jnp.power(x.shape[1], 0.5)
        high = 1*1/jnp.power(x.shape[1], 0.5)
        return onp.random.uniform(low,high,shape)

    def sigma_init(key, shape, dtype=jnp.float32): return jnp.ones(shape, dtype)*(0.1 / onp.sqrt(x.shape[1]))

    # Sample noise from gaussian
    p = sample_noise([x.shape[1], 1])
    q = sample_noise([1, features])
    f_p = f(p); f_q = f(q)
    w_epsilon = f_p*f_q; b_epsilon = jnp.squeeze(f_q)
    w_mu = self.param('kernel',(x.shape[1], features), mu_init)
    w_sigma = self.param('kernell',(x.shape[1], features),sigma_init)
    w = w_mu + jnp.multiply(w_sigma, w_epsilon)
    ret = jnp.matmul(x, w)

    if bias:
      # b = b_mu + b_sigma*b_epsilon
      b_mu = self.param('bias',(features,),mu_init)
      b_sigma = self.param('biass',(features,),sigma_init)
      b = b_mu + jnp.multiply(b_sigma, b_epsilon)
      return ret + b
    else:
      return ret

#---------------------------------------------< Cartpole >----------------------------------------------------------

@gin.configurable
class CartpoleDDQNNetwork(nn.Module):
  """Jax DQN network for Cartpole."""

  def apply(self, x, num_actions, noisy, dueling):
    print('CartpoleD-DQNNetwork')
    # We need to add a "batch dimension" as nn.Conv expects it, yet vmap will
    # have removed the true batch dimension.
    x = x[None, ...]
    x = x.astype(jnp.float32)
    x = x.reshape((x.shape[0], -1))  # flatten
    x -= gym_lib.CARTPOLE_MIN_VALS
    x /= gym_lib.CARTPOLE_MAX_VALS - gym_lib.CARTPOLE_MIN_VALS
    x = 2.0 * x - 1.0  # Rescale in range [-1, 1].
    
    if noisy:
        print('CartpoleDDQNNetwork-Noisy[Johan]')
        initializer = None
        bias = True
        def net(x, features, bias, kernel_init):
            return NoisyNetwork(x, features, bias, kernel_init)
    else:
        initializer = nn.initializers.xavier_uniform()
        bias = None
        def net(x, features, bias, kernel_init):
            return nn.Dense(x, features, kernel_init)      

    x = net(x, features=512, bias=bias, kernel_init=initializer)
    x = jax.nn.relu(x)
    x = net(x, features=512, bias=bias, kernel_init=initializer)  
    x = jax.nn.relu(x)

    if dueling:
        print('CartpoleDDQNNNetwork-Dueling[Johan]')
        adv = net(x, features=num_actions, bias=bias, kernel_init=initializer)
        val = net(x, features=1, bias=bias, kernel_init=initializer)
        q_values = val + (adv - (jnp.mean(adv, -1, keepdims=True)))

    else:
        q_values = net(x, features=num_actions, bias=bias, kernel_init=initializer)

    return atari_lib.DQNNetworkType(q_values)


@gin.configurable
class CartpoleRainbowFull(nn.Module):
  """Jax Rainbow network for Cartpole."""
  def apply(self, x, num_actions, num_atoms, support, noisy, dueling):
    # We need to add a "batch dimension" as nn.Conv expects it, yet vmap will
    # have removed the true batch dimension.
    x = x[None, ...]
    x = x.astype(jnp.float32)
    x = x.reshape((x.shape[0], -1))  # flatten
    x -= gym_lib.CARTPOLE_MIN_VALS
    x /= gym_lib.CARTPOLE_MAX_VALS - gym_lib.CARTPOLE_MIN_VALS
    x = 2.0 * x - 1.0  # Rescale in range [-1, 1].

    if noisy:
        print('CartpoleRainbowFull-Noisy')
        initializer = None
        bias = True
        def net(x, features, bias, kernel_init):
            return NoisyNetwork(x, features, bias, kernel_init)
    else:
        initializer = nn.initializers.xavier_uniform()
        bias = None
        def net(x, features, bias, kernel_init):
            return nn.Dense(x, features, kernel_init) 

    x = net(x, features=512, bias=bias, kernel_init=initializer)
    x = jax.nn.relu(x)
    x = net(x,features=512, bias=bias, kernel_init=initializer)
    x = jax.nn.relu(x)

    if dueling:
        print('CartpoleRainbowFull-Dueling')
        adv = net(x,features=num_actions * num_atoms, bias=bias, kernel_init=initializer)
        value = net(x, features=num_atoms, bias=bias, kernel_init=initializer)
        adv = adv.reshape((adv.shape[0], num_actions, num_atoms))
        value = value.reshape((value.shape[0], 1, num_atoms))
        logits = value + (adv - (jnp.mean(adv, -1, keepdims=True)))
        probabilities = nn.softmax(logits)
        q_values = jnp.sum(support * probabilities, axis=2)

    else:
        x = net(x, features=num_actions * num_atoms, bias=bias, kernel_init=initializer)
        logits = x.reshape((x.shape[0], num_actions, num_atoms))
        probabilities = nn.softmax(logits)
        q_values = jnp.sum(support * probabilities, axis=2)
    
    return atari_lib.RainbowNetworkType(q_values, logits, probabilities)

#---------------------------------------------< MountainCar >----------------------------------------------------------


@gin.configurable
class MountainCarDDQNNetwork(nn.Module):
  """Jax DQN network for Cartpole."""

  def apply(self, x, num_actions, noisy, dueling):
    print('MountainCarD-DQNNetwork')
    # We need to add a "batch dimension" as nn.Conv expects it, yet vmap will
    # have removed the true batch dimension.

    print('num_actions',num_actions)

    MOUNTAINCAR_MIN_VALS = onp.array([-1.2, -0.07])
    MOUNTAINCAR_MAX_VALS = onp.array([0.6, 0.07])


    x = x[None, ...]
    x = x.astype(jnp.float32)
    x = x.reshape((x.shape[0], -1))  # flatten
    x -= MOUNTAINCAR_MIN_VALS
    x /= MOUNTAINCAR_MAX_VALS - MOUNTAINCAR_MIN_VALS
    x = 2.0 * x - 1.0  # Rescale in range [-1, 1].
    
    if noisy:
        print('MountainCarDDQNNetwork-Noisy[Johan]')
        initializer = None
        bias = True
        def net(x, features, bias, kernel_init):
            return NoisyNetwork(x, features, bias, kernel_init)
    else:
        print('MountainCarDDQNNetwork-NO-Noisy')
        initializer = nn.initializers.xavier_uniform()
        bias = None
        def net(x, features, bias, kernel_init):
            return nn.Dense(x, features, kernel_init)      

    x = net(x, features=512, bias=bias, kernel_init=initializer)
    x = jax.nn.relu(x)
    x = net(x, features=512, bias=bias, kernel_init=initializer)  
    x = jax.nn.relu(x)

    if dueling:
        print('MountainCarDDQNNNetwork-Dueling[Johan]')
        adv = net(x, features=num_actions, bias=bias, kernel_init=initializer)
        val = net(x, features=1, bias=bias, kernel_init=initializer)
        q_values = val + (adv - (jnp.mean(adv, -1, keepdims=True)))

    else:
        q_values = net(x, features=num_actions, bias=bias, kernel_init=initializer)

    return atari_lib.DQNNetworkType(q_values)


@gin.configurable
class MountainCarRainbowFull(nn.Module):
  """Jax Rainbow network for Cartpole."""
  def apply(self, x, num_actions, num_atoms, support, noisy, dueling):
    # We need to add a "batch dimension" as nn.Conv expects it, yet vmap will
    # have removed the true batch dimension.
    x = x[None, ...]
    x = x.astype(jnp.float32)
    x = x.reshape((x.shape[0], -1))  # flatten
    #x -= gym_lib.CARTPOLE_MIN_VALS
    #x /= gym_lib.CARTPOLE_MAX_VALS - gym_lib.CARTPOLE_MIN_VALS
    #x = 2.0 * x - 1.0  # Rescale in range [-1, 1].

    if noisy:
        print('MountainCarRainbowFull-Noisy')
        initializer = None
        bias = True
        def net(x, features, bias, kernel_init):
            return NoisyNetwork(x, features, bias, kernel_init)
    else:
        initializer = nn.initializers.xavier_uniform()
        bias = None
        def net(x, features, bias, kernel_init):
            return nn.Dense(x, features, kernel_init) 

    x = net(x, features=512, bias=bias, kernel_init=initializer)
    x = jax.nn.relu(x)
    x = net(x,features=512, bias=bias, kernel_init=initializer)
    x = jax.nn.relu(x)

    if dueling:
        print('MountainCarRainbowFull-Dueling')
        adv = net(x,features=num_actions * num_atoms, bias=bias, kernel_init=initializer)
        value = net(x, features=num_atoms, bias=bias, kernel_init=initializer)
        adv = adv.reshape((adv.shape[0], num_actions, num_atoms))
        value = value.reshape((value.shape[0], 1, num_atoms))
        logits = value + (adv - (jnp.mean(adv, -1, keepdims=True)))
        probabilities = nn.softmax(logits)
        q_values = jnp.sum(support * probabilities, axis=2)

    else:
        x = net(x, features=num_actions * num_atoms, bias=bias, kernel_init=initializer)
        logits = x.reshape((x.shape[0], num_actions, num_atoms))
        probabilities = nn.softmax(logits)
        q_values = jnp.sum(support * probabilities, axis=2)
    
    return atari_lib.RainbowNetworkType(q_values, logits, probabilities)

#---------------------------------------------< LunarLander >----------------------------------------------------------

@gin.configurable
class LunarLanderDDQNNetwork(nn.Module):
  """Jax DQN network for Cartpole."""

  def apply(self, x, num_actions, noisy, dueling):
    # We need to add a "batch dimension" as nn.Conv expects it, yet vmap will
    # have removed the true batch dimension.
    x = x[None, ...]
    x = x.astype(jnp.float32)
    x = x.reshape((x.shape[0], -1))  # flatten
    #x -= gym_lib.ACROBOT_MIN_VALS
    #x /= gym_lib.ACROBOT_MAX_VALS - gym_lib.ACROBOT_MIN_VALS
    #x = 2.0 * x - 1.0  # Rescale in range [-1, 1].

    if noisy:
        print('LunarLanderDDQNNetwork-Noisy[Johan]')
        initializer = None
        bias = True
        def net(x, features, bias, kernel_init):
            return NoisyNetwork(x, features, bias, kernel_init)
    else:
        initializer = nn.initializers.xavier_uniform()
        bias = None
        def net(x, features, bias, kernel_init):
            return nn.Dense(x, features, kernel_init)

    x = net(x, features=512, bias=bias, kernel_init=initializer)
    x = jax.nn.relu(x)
    x = net(x, features=512, bias=bias, kernel_init=initializer)  
    x = jax.nn.relu(x)

    if dueling:
        print('LunarLanderDDQNNetwork-Dueling')
        adv = net(x, features=num_actions, bias=bias, kernel_init=initializer)
        val = net(x, features=1, bias=bias, kernel_init=initializer)
        q_values = val + (adv - (jnp.mean(adv, -1, keepdims=True)))

    else:
        q_values = net(x, features=num_actions, bias=bias, kernel_init=initializer)


    return atari_lib.DQNNetworkType(q_values)

@gin.configurable
class LunarLanderRainbowFull(nn.Module):
  """Jax Rainbow network for Cartpole."""
  def apply(self, x, num_actions, num_atoms, support, noisy, dueling):
    # We need to add a "batch dimension" as nn.Conv expects it, yet vmap will
    # have removed the true batch dimension.

    x = x[None, ...]
    x = x.astype(jnp.float32)
    x = x.reshape((x.shape[0], -1))  # flatten
    #x -= gym_lib.CARTPOLE_MIN_VALS
    #x /= gym_lib.CARTPOLE_MAX_VALS - gym_lib.CARTPOLE_MIN_VALS
    #x = 2.0 * x - 1.0  # Rescale in range [-1, 1].

    if noisy:
        print('LunarLander-Noisy[Johan]')
        initializer = None
        bias = True
        def net(x, features, bias, kernel_init):
            return NoisyNetwork(x, features, bias, kernel_init)
    else:
        initializer = nn.initializers.xavier_uniform()
        bias = None
        def net(x, features, bias, kernel_init):
            return nn.Dense(x, features, kernel_init)

    x = net(x, features=512, bias=bias, kernel_init=initializer)
    x = jax.nn.relu(x)
    x = net(x,features=512, bias=bias, kernel_init=initializer)
    x = jax.nn.relu(x)

    if dueling:
        print('LunarLanderRainbowFull-Dueling')
        adv = net(x,features=num_actions * num_atoms, bias=bias, kernel_init=initializer)
        value = net(x, features=num_atoms, bias=bias, kernel_init=initializer)
        adv = adv.reshape((adv.shape[0], num_actions, num_atoms))
        value = value.reshape((value.shape[0], 1, num_atoms))
        logits = value + (adv - (jnp.mean(adv, -1, keepdims=True)))
        probabilities = nn.softmax(logits)
        q_values = jnp.sum(support * probabilities, axis=2)

    else:
        x = net(x, features=num_actions * num_atoms, bias=bias, kernel_init=initializer)
        logits = x.reshape((x.shape[0], num_actions, num_atoms))
        probabilities = nn.softmax(logits)
        q_values = jnp.sum(support * probabilities, axis=2)
    
    return atari_lib.RainbowNetworkType(q_values, logits, probabilities)



#-----------------------------------------------< Acrabot >------------------------------------------------------------

@gin.configurable
class AcrabotDDQNNetwork(nn.Module):
  """Jax DQN network for Cartpole."""

  def apply(self,x, num_actions, noisy, dueling):
    # We need to add a "batch dimension" as nn.Conv expects it, yet vmap will
    # have removed the true batch dimension.
    x = x[None, ...]
    x = x.astype(jnp.float32)
    x = x.reshape((x.shape[0], -1))  # flatten
    x -= gym_lib.ACROBOT_MIN_VALS
    x /= gym_lib.ACROBOT_MAX_VALS - gym_lib.ACROBOT_MIN_VALS
    x = 2.0 * x - 1.0  # Rescale in range [-1, 1].

    if noisy:
        print('AcrabotDDQNNetwork-Noisy[Johan]')
        initializer = None
        bias = True
        def net(x, features, bias, kernel_init):
            return NoisyNetwork(x, features, bias, kernel_init)
    else:
        initializer = nn.initializers.xavier_uniform()
        bias = None
        def net(x, features, bias, kernel_init):
            return nn.Dense(x, features, kernel_init)      

    x = net(x, features=512, bias=bias, kernel_init=initializer)
    x = jax.nn.relu(x)
    x = net(x, features=512, bias=bias, kernel_init=initializer)  
    x = jax.nn.relu(x)

    if dueling:
        print('AcrabotDDQNNetwork-Dueling[Johan]')
        adv = net(x, features=num_actions, bias=bias, kernel_init=initializer)
        val = net(x, features=1, bias=bias, kernel_init=initializer)
        #q_values = val + (adv - (jnp.mean(adv, -1, keepdims=True)))
        q_values = val + (adv - (jnp.mean(adv, 1, keepdims=True)))
        print('q_values.shape',q_values.shape, q_values)

    else:
        q_values = net(x, features=num_actions, bias=bias, kernel_init=initializer)


    return atari_lib.DQNNetworkType(q_values)



@gin.configurable
class AcrabotRainbowFull(nn.Module):
  """Jax Rainbow network for Cartpole."""
  def apply(self, x, num_actions, num_atoms, support, noisy, dueling):
    # We need to add a "batch dimension" as nn.Conv expects it, yet vmap will
    # have removed the true batch dimension.
    x = x[None, ...]
    x = x.astype(jnp.float32)
    x = x.reshape((x.shape[0], -1))  # flatten
    x -= gym_lib.ACROBOT_MIN_VALS
    x /= gym_lib.ACROBOT_MAX_VALS - gym_lib.ACROBOT_MIN_VALS
    x = 2.0 * x - 1.0  # Rescale in range [-1, 1].

    if noisy:
        print(' AcrabotRainbowFull-Noisy[Johan]')
        initializer = None
        bias = True
        def net(x, features, bias, kernel_init):
            return NoisyNetwork(x, features, bias, kernel_init)
    else:
        initializer = nn.initializers.xavier_uniform()
        bias = None
        def net(x, features, bias, kernel_init):
            return nn.Dense(x, features, kernel_init)        

    x = net(x, features=512, bias=bias, kernel_init=initializer)
    x = jax.nn.relu(x)
    x = net(x,features=512, bias=bias, kernel_init=initializer)
    x = jax.nn.relu(x)

    if dueling:
        print('AcrabotRainbowFull-Dueling')
        adv = net(x,features=num_actions * num_atoms, bias=bias, kernel_init=initializer)
        value = net(x, features=num_atoms, bias=bias, kernel_init=initializer)
        adv = adv.reshape((adv.shape[0], num_actions, num_atoms))
        value = value.reshape((value.shape[0], 1, num_atoms))
        logits = value + (adv - (jnp.mean(adv, -1, keepdims=True)))
        probabilities = nn.softmax(logits)
        q_values = jnp.sum(support * probabilities, axis=2)

    else:
        x = net(x, features=num_actions * num_atoms, bias=bias, kernel_init=initializer)
        logits = x.reshape((x.shape[0], num_actions, num_atoms))
        probabilities = nn.softmax(logits)
        q_values = jnp.sum(support * probabilities, axis=2)
    
    return atari_lib.RainbowNetworkType(q_values, logits, probabilities)  


#-----------------------------------------------< Minatar Seaquest  >-----------------------------------------------------------

@gin.configurable
class SeaquestDDQNNetwork(nn.Module):
  """Jax DQN network for Cartpole."""

  def apply(self,x, num_actions, noisy, dueling):
    # We need to add a "batch dimension" as nn.Conv expects it, yet vmap will
    # have removed the true batch dimension.
    initializer_conv = nn.initializers.xavier_uniform()
    x = x.squeeze(3)
    x = x[None, ...]
    x = x.astype(jnp.float32)
    x = nn.Conv(x, features=16, kernel_size=(3, 3, 3), strides=(1, 1, 1),  kernel_init=initializer_conv)
    x = jax.nn.relu(x)
    x = x.reshape((x.shape[0], -1))  # flatten
    
    if noisy:
        print('SeaquestDDQNNetwork-Noisy[Johan]')
        initializer = None
        bias = True
        def net(x, features, bias, kernel_init):
            return NoisyNetwork(x, features, bias, kernel_init)
    else:
        #initializer = nn.initializers.xavier_uniform()
        initializer = initializer_conv
        bias = None
        def net(x, features, bias, kernel_init):
            return nn.Dense(x, features, kernel_init)      

    if dueling:
        print('SeaquestDDQNNetwork-Dueling[Johan]')
        adv = net(x, features=num_actions, bias=bias, kernel_init=initializer)
        val = net(x, features=1, bias=bias, kernel_init=initializer)
        q_values = val + (adv - (jnp.mean(adv, -1, keepdims=True)))

    else:
        q_values = net(x, features=num_actions, bias=bias, kernel_init=initializer)

    return atari_lib.DQNNetworkType(q_values)


@gin.configurable
class SeaquestRainbowFull(nn.Module):
  """Jax Rainbow network for Cartpole."""
  def apply(self, x, num_actions, num_atoms, support, noisy, dueling):
    # We need to add a "batch dimension" as nn.Conv expects it, yet vmap will
    # have removed the true batch dimension.
    initializer_conv = nn.initializers.xavier_uniform()
    x = x.squeeze(3)
    x = x[None, ...]
    x = x.astype(jnp.float32)
    x = nn.Conv(x, features=16, kernel_size=(3, 3, 3), strides=(1, 1, 1),  kernel_init=initializer_conv)
    x = jax.nn.relu(x)
    x = x.reshape((x.shape[0], -1))  # flatten.

    if noisy:
        print(' SeaquestRainbowFull-Noisy[Johan]')
        initializer = None
        bias = True
        def net(x, features, bias, kernel_init):
            return NoisyNetwork(x, features, bias, kernel_init)
    else:
        #initializer = nn.initializers.xavier_uniform()
        initializer = initializer_conv
        bias = None
        def net(x, features, bias, kernel_init):
            return nn.Dense(x, features, kernel_init)        

    if dueling:
        print('SeaquestRainbowFull-Dueling')
        adv = net(x,features=num_actions * num_atoms, bias=bias, kernel_init=initializer)
        value = net(x, features=num_atoms, bias=bias, kernel_init=initializer)
        adv = adv.reshape((adv.shape[0], num_actions, num_atoms))
        value = value.reshape((value.shape[0], 1, num_atoms))
        logits = value + (adv - (jnp.mean(adv, -1, keepdims=True)))
        probabilities = nn.softmax(logits)
        q_values = jnp.sum(support * probabilities, axis=2)

    else:
        x = net(x, features=num_actions * num_atoms, bias=bias, kernel_init=initializer)
        logits = x.reshape((x.shape[0], num_actions, num_atoms))
        probabilities = nn.softmax(logits)
        q_values = jnp.sum(support * probabilities, axis=2)
    
    return atari_lib.RainbowNetworkType(q_values, logits, probabilities)  

#-----------------------------------------------< Minatar Breakout  >-----------------------------------------------------------
@gin.configurable
class BreakoutDDQNNetwork(nn.Module):
  """Jax DQN network for Cartpole."""

  def apply(self,x, num_actions, noisy, dueling):
    # We need to add a "batch dimension" as nn.Conv expects it, yet vmap will
    # have removed the true batch dimension.
    initializer_conv = nn.initializers.xavier_uniform()
    #print('x.shape',x.shape)
    x = x.squeeze(3)
    #print('x.shape',x.shape)
    x = x[None, ...]
    x = x.astype(jnp.float32)
    #print('x.shape',x.shape)
    x = nn.Conv(x, features=16, kernel_size=(3, 3, 3), strides=(1, 1, 1),  kernel_init=initializer_conv)
    x = jax.nn.relu(x)
    x = x.reshape((x.shape[0], -1))  # flatten
    
    if noisy:
        print('BreakoutDDQNNetwork-Noisy[Johan]')
        initializer = None
        bias = True
        def net(x, features, bias, kernel_init):
            return NoisyNetwork(x, features, bias, kernel_init)
    else:
        #initializer = nn.initializers.xavier_uniform()
        initializer = initializer_conv
        bias = None
        def net(x, features, bias, kernel_init):
            return nn.Dense(x, features, kernel_init)      

    if dueling:
        print('BreakoutDDQNNetwork-Dueling[Johan]')
        adv = net(x, features=num_actions, bias=bias, kernel_init=initializer)
        val = net(x, features=1, bias=bias, kernel_init=initializer)
        q_values = val + (adv - (jnp.mean(adv, -1, keepdims=True)))

    else:
        q_values = net(x, features=num_actions, bias=bias, kernel_init=initializer)

    return atari_lib.DQNNetworkType(q_values)


@gin.configurable
class BreakoutRainbowFull(nn.Module):
  """Jax Rainbow network for Cartpole."""
  def apply(self, x, num_actions, num_atoms, support, noisy, dueling):
    # We need to add a "batch dimension" as nn.Conv expects it, yet vmap will
    # have removed the true batch dimension.
    initializer_conv = nn.initializers.xavier_uniform()
    x = x.squeeze(3)
    x = x[None, ...]
    x = x.astype(jnp.float32)
    x = nn.Conv(x, features=16, kernel_size=(3, 3, 3), strides=(1, 1, 1),  kernel_init=initializer_conv)
    x = jax.nn.relu(x)
    x = x.reshape((x.shape[0], -1))  # flatten.

    if noisy:
        print('BreakoutRainbowFull-Noisy[Johan]')
        initializer = None
        bias = True
        def net(x, features, bias, kernel_init):
            return NoisyNetwork(x, features, bias, kernel_init)
    else:
        #initializer = nn.initializers.xavier_uniform()
        initializer = initializer_conv
        bias = None
        def net(x, features, bias, kernel_init):
            return nn.Dense(x, features, kernel_init)        

    if dueling:
        print('BreakoutRainbowFull-Dueling')
        adv = net(x,features=num_actions * num_atoms, bias=bias, kernel_init=initializer)
        value = net(x, features=num_atoms, bias=bias, kernel_init=initializer)
        adv = adv.reshape((adv.shape[0], num_actions, num_atoms))
        value = value.reshape((value.shape[0], 1, num_atoms))
        logits = value + (adv - (jnp.mean(adv, -1, keepdims=True)))
        probabilities = nn.softmax(logits)
        q_values = jnp.sum(support * probabilities, axis=2)

    else:
        x = net(x, features=num_actions * num_atoms, bias=bias, kernel_init=initializer)
        logits = x.reshape((x.shape[0], num_actions, num_atoms))
        probabilities = nn.softmax(logits)
        q_values = jnp.sum(support * probabilities, axis=2)
    
    return atari_lib.RainbowNetworkType(q_values, logits, probabilities)  

#-----------------------------------------------< Minatar Asterix   >------------------------------------------------------------
@gin.configurable
class AsterixDDQNNetwork(nn.Module):
  """Jax DQN network for Cartpole."""

  def apply(self,x, num_actions, noisy, dueling):
    # We need to add a "batch dimension" as nn.Conv expects it, yet vmap will
    # have removed the true batch dimension.
    initializer_conv = nn.initializers.xavier_uniform()
    x = x.squeeze(3)
    x = x[None, ...]
    x = x.astype(jnp.float32)
    x = nn.Conv(x, features=16, kernel_size=(3, 3, 3), strides=(1, 1, 1),  kernel_init=initializer_conv)
    x = jax.nn.relu(x)
    x = x.reshape((x.shape[0], -1))  # flatten
    
    if noisy:
        print('AsterixDDQNNetwork-Noisy[Johan]')
        initializer = None
        bias = True
        def net(x, features, bias, kernel_init):
            return NoisyNetwork(x, features, bias, kernel_init)
    else:
        #initializer = nn.initializers.xavier_uniform()
        initializer = initializer_conv
        bias = None
        def net(x, features, bias, kernel_init):
            return nn.Dense(x, features, kernel_init)      

    if dueling:
        print('AsterixDDQNNetwork-Dueling[Johan]')
        adv = net(x, features=num_actions, bias=bias, kernel_init=initializer)
        val = net(x, features=1, bias=bias, kernel_init=initializer)
        q_values = val + (adv - (jnp.mean(adv, -1, keepdims=True)))

    else:
        q_values = net(x, features=num_actions, bias=bias, kernel_init=initializer)

    return atari_lib.DQNNetworkType(q_values)


@gin.configurable
class AsterixRainbowFull(nn.Module):
  """Jax Rainbow network for Cartpole."""
  def apply(self, x, num_actions, num_atoms, support, noisy, dueling):
    # We need to add a "batch dimension" as nn.Conv expects it, yet vmap will
    # have removed the true batch dimension.
    initializer_conv = nn.initializers.xavier_uniform()
    x = x.squeeze(3)
    x = x[None, ...]
    x = x.astype(jnp.float32)
    x = nn.Conv(x, features=16, kernel_size=(3, 3, 3), strides=(1, 1, 1),  kernel_init=initializer_conv)
    x = jax.nn.relu(x)
    x = x.reshape((x.shape[0], -1))  # flatten.

    if noisy:
        print('AsterixRainbowFull-Noisy[Johan]')
        initializer = None
        bias = True
        def net(x, features, bias, kernel_init):
            return NoisyNetwork(x, features, bias, kernel_init)
    else:
        #initializer = nn.initializers.xavier_uniform()
        initializer = initializer_conv
        bias = None
        def net(x, features, bias, kernel_init):
            return nn.Dense(x, features, kernel_init)        

    if dueling:
        print('AsterixRainbowFull-Dueling')
        adv = net(x,features=num_actions * num_atoms, bias=bias, kernel_init=initializer)
        value = net(x, features=num_atoms, bias=bias, kernel_init=initializer)
        adv = adv.reshape((adv.shape[0], num_actions, num_atoms))
        value = value.reshape((value.shape[0], 1, num_atoms))
        logits = value + (adv - (jnp.mean(adv, -1, keepdims=True)))
        probabilities = nn.softmax(logits)
        q_values = jnp.sum(support * probabilities, axis=2)

    else:
        x = net(x, features=num_actions * num_atoms, bias=bias, kernel_init=initializer)
        logits = x.reshape((x.shape[0], num_actions, num_atoms))
        probabilities = nn.softmax(logits)
        q_values = jnp.sum(support * probabilities, axis=2)
    
    return atari_lib.RainbowNetworkType(q_values, logits, probabilities)  

#-----------------------------------------------< Minatar Freeway   >-----------------------------------------------------------
@gin.configurable
class FreewayDDQNNetwork(nn.Module):
  """Jax DQN network for Cartpole."""

  def apply(self,x, num_actions, noisy, dueling):
    # We need to add a "batch dimension" as nn.Conv expects it, yet vmap will
    # have removed the true batch dimension.
    initializer_conv = nn.initializers.xavier_uniform()
    x = x.squeeze(3)
    x = x[None, ...]
    x = x.astype(jnp.float32)
    x = nn.Conv(x, features=16, kernel_size=(3, 3, 3), strides=(1, 1, 1),  kernel_init=initializer_conv)
    x = jax.nn.relu(x)
    x = x.reshape((x.shape[0], -1))  # flatten
    
    if noisy:
        print('FreewayDDQNNetwork-Noisy[Johan]')
        initializer = None
        bias = True
        def net(x, features, bias, kernel_init):
            return NoisyNetwork(x, features, bias, kernel_init)
    else:
        #initializer = nn.initializers.xavier_uniform()
        initializer = initializer_conv
        bias = None
        def net(x, features, bias, kernel_init):
            return nn.Dense(x, features, kernel_init)      

    if dueling:
        print('FreewayDDQNNetwork-Dueling[Johan]')
        adv = net(x, features=num_actions, bias=bias, kernel_init=initializer)
        val = net(x, features=1, bias=bias, kernel_init=initializer)
        q_values = val + (adv - (jnp.mean(adv, -1, keepdims=True)))

    else:
        q_values = net(x, features=num_actions, bias=bias, kernel_init=initializer)

    return atari_lib.DQNNetworkType(q_values)


@gin.configurable
class FreewayRainbowFull(nn.Module):
  """Jax Rainbow network for Cartpole."""
  def apply(self, x, num_actions, num_atoms, support, noisy, dueling):
    # We need to add a "batch dimension" as nn.Conv expects it, yet vmap will
    # have removed the true batch dimension.
    initializer_conv = nn.initializers.xavier_uniform()
    x = x.squeeze(3)
    x = x[None, ...]
    x = x.astype(jnp.float32)
    x = nn.Conv(x, features=16, kernel_size=(3, 3, 3), strides=(1, 1, 1),  kernel_init=initializer_conv)
    x = jax.nn.relu(x)
    x = x.reshape((x.shape[0], -1))  # flatten.

    if noisy:
        print('FreewayRainbowFull-Noisy[Johan]')
        initializer = None
        bias = True
        def net(x, features, bias, kernel_init):
            return NoisyNetwork(x, features, bias, kernel_init)
    else:
        #initializer = nn.initializers.xavier_uniform()
        initializer = initializer_conv
        bias = None
        def net(x, features, bias, kernel_init):
            return nn.Dense(x, features, kernel_init)        

    if dueling:
        print('FreewayRainbowFull-Dueling')
        adv = net(x,features=num_actions * num_atoms, bias=bias, kernel_init=initializer)
        value = net(x, features=num_atoms, bias=bias, kernel_init=initializer)
        adv = adv.reshape((adv.shape[0], num_actions, num_atoms))
        value = value.reshape((value.shape[0], 1, num_atoms))
        logits = value + (adv - (jnp.mean(adv, -1, keepdims=True)))
        probabilities = nn.softmax(logits)
        q_values = jnp.sum(support * probabilities, axis=2)

    else:
        x = net(x, features=num_actions * num_atoms, bias=bias, kernel_init=initializer)
        logits = x.reshape((x.shape[0], num_actions, num_atoms))
        probabilities = nn.softmax(logits)
        q_values = jnp.sum(support * probabilities, axis=2)
    
    return atari_lib.RainbowNetworkType(q_values, logits, probabilities)  

#-----------------------------------------------< Minatar Space Invaders  >-----------------------------------------------------
@gin.configurable
class InvadersDDQNNetwork(nn.Module):
  """Jax DQN network for Cartpole."""

  def apply(self,x, num_actions, noisy, dueling):
    # We need to add a "batch dimension" as nn.Conv expects it, yet vmap will
    # have removed the true batch dimension.
    initializer_conv = nn.initializers.xavier_uniform()
    x = x.squeeze(3)
    x = x[None, ...]
    x = x.astype(jnp.float32)
    x = nn.Conv(x, features=16, kernel_size=(3, 3, 3), strides=(1, 1, 1),  kernel_init=initializer_conv)
    x = jax.nn.relu(x)
    x = x.reshape((x.shape[0], -1))  # flatten
    
    if noisy:
        print('InvadersDDQNNetwork-Noisy[Johan]')
        initializer = None
        bias = True
        def net(x, features, bias, kernel_init):
            return NoisyNetwork(x, features, bias, kernel_init)
    else:
        #initializer = nn.initializers.xavier_uniform()
        initializer = initializer_conv
        bias = None
        def net(x, features, bias, kernel_init):
            return nn.Dense(x, features, kernel_init)      

    if dueling:
        print('InvadersDDQNNetwork-Dueling[Johan]')
        adv = net(x, features=num_actions, bias=bias, kernel_init=initializer)
        val = net(x, features=1, bias=bias, kernel_init=initializer)
        q_values = val + (adv - (jnp.mean(adv, -1, keepdims=True)))

    else:
        q_values = net(x, features=num_actions, bias=bias, kernel_init=initializer)

    return atari_lib.DQNNetworkType(q_values)


@gin.configurable
class InvadersRainbowFull(nn.Module):
  """Jax Rainbow network for Cartpole."""
  def apply(self, x, num_actions, num_atoms, support, noisy, dueling):
    # We need to add a "batch dimension" as nn.Conv expects it, yet vmap will
    # have removed the true batch dimension.
    initializer_conv = nn.initializers.xavier_uniform()
    x = x.squeeze(3)
    x = x[None, ...]
    x = x.astype(jnp.float32)
    x = nn.Conv(x, features=16, kernel_size=(3, 3, 3), strides=(1, 1, 1),  kernel_init=initializer_conv)
    x = jax.nn.relu(x)
    x = x.reshape((x.shape[0], -1))  # flatten.

    if noisy:
        print('InvadersRainbowFull-Noisy[Johan]')
        initializer = None
        bias = True
        def net(x, features, bias, kernel_init):
            return NoisyNetwork(x, features, bias, kernel_init)
    else:
        #initializer = nn.initializers.xavier_uniform()
        initializer = initializer_conv
        bias = None
        def net(x, features, bias, kernel_init):
            return nn.Dense(x, features, kernel_init)        

    if dueling:
        print('InvadersRainbowFull-Dueling')
        adv = net(x,features=num_actions * num_atoms, bias=bias, kernel_init=initializer)
        value = net(x, features=num_atoms, bias=bias, kernel_init=initializer)
        adv = adv.reshape((adv.shape[0], num_actions, num_atoms))
        value = value.reshape((value.shape[0], 1, num_atoms))
        logits = value + (adv - (jnp.mean(adv, -1, keepdims=True)))
        probabilities = nn.softmax(logits)
        q_values = jnp.sum(support * probabilities, axis=2)

    else:
        x = net(x, features=num_actions * num_atoms, bias=bias, kernel_init=initializer)
        logits = x.reshape((x.shape[0], num_actions, num_atoms))
        probabilities = nn.softmax(logits)
        q_values = jnp.sum(support * probabilities, axis=2)
    
    return atari_lib.RainbowNetworkType(q_values, logits, probabilities)
