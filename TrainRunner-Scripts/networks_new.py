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


@gin.configurable
class NoisyNetwork(nn.Module):
  def apply(self, x, features, bias=True):
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
class CartpoleDuelingDQNNetwork(nn.Module):
  """Jax DQN network for Cartpole."""
  
  def apply(self, x, num_actions):
    print('CartpoleDuelingDQNNetwork')
    initializer = nn.initializers.xavier_uniform()
    # We need to add a "batch dimension" as nn.Conv expects it, yet vmap will
    # have removed the true batch dimension.
    x = x[None, ...]
    x = x.astype(jnp.float32)
    x = x.reshape((x.shape[0], -1))  # flatten
    x -= gym_lib.CARTPOLE_MIN_VALS
    x /= gym_lib.CARTPOLE_MAX_VALS - gym_lib.CARTPOLE_MIN_VALS
    x = 2.0 * x - 1.0  # Rescale in range [-1, 1].
    x = nn.Dense(x, features=512, kernel_init=initializer)
    x = jax.nn.relu(x)
    x = nn.Dense(x, features=512, kernel_init=initializer)
    x = jax.nn.relu(x)

    adv = nn.Dense(x, features=num_actions, kernel_init=initializer)
    val = nn.Dense(x, features=1, kernel_init=initializer)

    #q_values = nn.Dense(x, features=num_actions, kernel_init=initializer)
    # https://jax.readthedocs.io/en/latest/_modules/jax/nn/functions.html (JAX Mean)

    #q_values = val + (adv - (jnp.mean(adv, 1, keepdims=True)))
    q_values = val + (adv - (jnp.mean(adv, -1, keepdims=True)))
    return atari_lib.DQNNetworkType(q_values)

@gin.configurable
class CartpoleNoisyDQNNetwork(nn.Module):
  
  def apply(self, x, num_actions):
    initializer = nn.initializers.xavier_uniform()
    # We need to add a "batch dimension" as nn.Conv expects it, yet vmap will
    # have removed the true batch dimension.
    x = x[None, ...]
    x = x.astype(jnp.float32)
    x = x.reshape((x.shape[0], -1))  # flatten
    x -= gym_lib.CARTPOLE_MIN_VALS
    x /= gym_lib.CARTPOLE_MAX_VALS - gym_lib.CARTPOLE_MIN_VALS
    x = 2.0 * x - 1.0  # Rescale in range [-1, 1].
    x = NoisyNetwork(x, features=512, bias=True)
    x = jax.nn.relu(x)
    x = NoisyNetwork(x,features=512, bias=True)
    x = jax.nn.relu(x)
    q_values = NoisyNetwork(x, features=num_actions, bias=True)
    return atari_lib.DQNNetworkType(q_values)


@gin.configurable
class CartpoleRainbowFull(nn.Module):
  """Jax Rainbow network for Cartpole."""
  print('Dueling-Rainbow')
  def apply(self, x, num_actions, num_atoms, support):
    initializer = nn.initializers.xavier_uniform()
    # We need to add a "batch dimension" as nn.Conv expects it, yet vmap will
    # have removed the true batch dimension.
    x = x[None, ...]
    x = x.astype(jnp.float32)
    x = x.reshape((x.shape[0], -1))  # flatten
    x -= gym_lib.CARTPOLE_MIN_VALS
    x /= gym_lib.CARTPOLE_MAX_VALS - gym_lib.CARTPOLE_MIN_VALS
    x = 2.0 * x - 1.0  # Rescale in range [-1, 1].
    #x = nn.Dense(x, features=512, kernel_init=initializer)
    x = NoisyNetwork(x, features=512, bias=True)
    x = jax.nn.relu(x)
    #x = nn.Dense(x, features=512, kernel_init=initializer)
    x = NoisyNetwork(x,features=512, bias=True)
    x = jax.nn.relu(x)

    #adv = nn.Dense(x, features=num_actions * num_atoms, kernel_init=initializer)
    adv = NoisyNetwork(x,features=num_actions * num_atoms, bias=True)
    #value = nn.Dense(x, features= num_atoms, kernel_init=initializer)
    value = NoisyNetwork(x, features=num_atoms, bias=True)

    adv = adv.reshape((adv.shape[0], num_actions, num_atoms))
    value = value.reshape((value.shape[0], 1, num_atoms))

    q_atoms = value + (adv - (jnp.mean(adv, -1, keepdims=True)))

    probabilities = nn.softmax(q_atoms)
    q_values = jnp.sum(support * probabilities, axis=2)

    return atari_lib.RainbowNetworkType(q_values, q_atoms, probabilities)

#---------------------------------------------< LunarLander >----------------------------------------------------------

@gin.configurable
class LunarLanderDQNNetwork(nn.Module):
  """Jax DQN network for Cartpole."""

  def apply(self, x, num_actions):
    initializer = nn.initializers.xavier_uniform()
    # We need to add a "batch dimension" as nn.Conv expects it, yet vmap will
    # have removed the true batch dimension.
    x = x[None, ...]
    x = x.astype(jnp.float32)
    x = x.reshape((x.shape[0], -1))  # flatten
    #x -= gym_lib.ACROBOT_MIN_VALS
    #x /= gym_lib.ACROBOT_MAX_VALS - gym_lib.ACROBOT_MIN_VALS
    #x = 2.0 * x - 1.0  # Rescale in range [-1, 1].
    x = nn.Dense(x, features=512, kernel_init=initializer)
    x = jax.nn.relu(x)
    x = nn.Dense(x, features=512, kernel_init=initializer)
    x = jax.nn.relu(x)
    q_values = nn.Dense(x, features=num_actions, kernel_init=initializer)
    return atari_lib.DQNNetworkType(q_values)


@gin.configurable
class LunarLanderDuelingDQNNetwork(nn.Module):
  """Jax DQN network for Cartpole."""
  
  def apply(self, x, num_actions):
    initializer = nn.initializers.xavier_uniform()
    # We need to add a "batch dimension" as nn.Conv expects it, yet vmap will
    # have removed the true batch dimension.
    x = x[None, ...]
    x = x.astype(jnp.float32)
    x = x.reshape((x.shape[0], -1))  # flatten
    #x -= gym_lib.CARTPOLE_MIN_VALS
    #x /= gym_lib.CARTPOLE_MAX_VALS - gym_lib.CARTPOLE_MIN_VALS
    #x = 2.0 * x - 1.0  # Rescale in range [-1, 1].
    x = nn.Dense(x, features=512, kernel_init=initializer)
    x = jax.nn.relu(x)
    x = nn.Dense(x, features=512, kernel_init=initializer)
    x = jax.nn.relu(x)
    print('x',x.shape,len(x))

    adv = nn.Dense(x, features=num_actions, kernel_init=initializer)
    val = nn.Dense(x, features=1, kernel_init=initializer)

    q_values = val + (adv - (jnp.mean(adv, -1, keepdims=True)))
    return atari_lib.DQNNetworkType(q_values)


@gin.configurable
class LunarLanderNoisyDQNNetwork(nn.Module):
  
  def apply(self, x, num_actions):
    initializer = nn.initializers.xavier_uniform()
    # We need to add a "batch dimension" as nn.Conv expects it, yet vmap will
    # have removed the true batch dimension.
    x = x[None, ...]
    x = x.astype(jnp.float32)
    x = x.reshape((x.shape[0], -1))  # flatten
    #x -= gym_lib.CARTPOLE_MIN_VALS
    #x /= gym_lib.CARTPOLE_MAX_VALS - gym_lib.CARTPOLE_MIN_VALS
    #x = 2.0 * x - 1.0  # Rescale in range [-1, 1].
    x = NoisyNetwork(x, features=512, bias=True)
    x = jax.nn.relu(x)
    x = NoisyNetwork(x,features=512, bias=True)
    x = jax.nn.relu(x)
    q_values = NoisyNetwork(x, features=num_actions, bias=True)
    return atari_lib.DQNNetworkType(q_values)

@gin.configurable
class LunarLanderRainbowFull(nn.Module):
  """Jax Rainbow network for Cartpole."""
  print('Dueling-Rainbow')
  def apply(self, x, num_actions, num_atoms, support):
    initializer = nn.initializers.xavier_uniform()
    # We need to add a "batch dimension" as nn.Conv expects it, yet vmap will
    # have removed the true batch dimension.
    x = x[None, ...]
    x = x.astype(jnp.float32)
    x = x.reshape((x.shape[0], -1))  # flatten
    #x -= gym_lib.CARTPOLE_MIN_VALS
    #x /= gym_lib.CARTPOLE_MAX_VALS - gym_lib.CARTPOLE_MIN_VALS
    #x = 2.0 * x - 1.0  # Rescale in range [-1, 1].

    #x = nn.Dense(x, features=512, kernel_init=initializer)
    x = NoisyNetwork(x, features=512, bias=True)
    x = jax.nn.relu(x)
    #x = nn.Dense(x, features=512, kernel_init=initializer)
    x = NoisyNetwork(x,features=512, bias=True)
    x = jax.nn.relu(x)

    #adv = nn.Dense(x, features=num_actions * num_atoms, kernel_init=initializer)
    adv = NoisyNetwork(x,features=num_actions * num_atoms, bias=True)
    #value = nn.Dense(x, features= num_atoms, kernel_init=initializer)
    value = NoisyNetwork(x, features=num_atoms, bias=True)
    
    adv = adv.reshape((adv.shape[0], num_actions, num_atoms))
    value = value.reshape((value.shape[0], 1, num_atoms))

    q_atoms = value + (adv - (jnp.mean(adv, -1, keepdims=True)))

    probabilities = nn.softmax(q_atoms)
    q_values = jnp.sum(support * probabilities, axis=2)

    return atari_lib.RainbowNetworkType(q_values, q_atoms, probabilities)

#------------------------------------------------------------------------------------------------------------

@gin.configurable
class AcrabotDuelingDQNNetwork(nn.Module):
  """Jax DQN network for Cartpole."""
  
  def apply(self, x, num_actions):
    print('CartpoleDuelingDQNNetwork')
    initializer = nn.initializers.xavier_uniform()
    # We need to add a "batch dimension" as nn.Conv expects it, yet vmap will
    # have removed the true batch dimension.
    x = x[None, ...]
    x = x.astype(jnp.float32)
    x = x.reshape((x.shape[0], -1))  # flatten
    x -= gym_lib.ACROBOT_MIN_VALS
    x /= gym_lib.ACROBOT_MAX_VALS - gym_lib.ACROBOT_MIN_VALS
    x = 2.0 * x - 1.0  # Rescale in range [-1, 1].
    x = nn.Dense(x, features=512, kernel_init=initializer)
    x = jax.nn.relu(x)
    x = nn.Dense(x, features=512, kernel_init=initializer)
    x = jax.nn.relu(x)

    adv = nn.Dense(x, features=num_actions, kernel_init=initializer)
    val = nn.Dense(x, features=1, kernel_init=initializer)

    #q_values = nn.Dense(x, features=num_actions, kernel_init=initializer)
    # https://jax.readthedocs.io/en/latest/_modules/jax/nn/functions.html (JAX Mean)

    #q_values = val + (adv - (jnp.mean(adv, 1, keepdims=True)))
    q_values = val + (adv - (jnp.mean(adv, -1, keepdims=True)))
    return atari_lib.DQNNetworkType(q_values)
    



@gin.configurable
class AcrabotRainbowFull(nn.Module):
  """Jax Rainbow network for Cartpole."""
  print('Dueling-Rainbow')
  def apply(self, x, num_actions, num_atoms, support):
    initializer = nn.initializers.xavier_uniform()
    # We need to add a "batch dimension" as nn.Conv expects it, yet vmap will
    # have removed the true batch dimension.
    x = x[None, ...]
    x = x.astype(jnp.float32)
    x = x.reshape((x.shape[0], -1))  # flatten
    x -= gym_lib.ACROBOT_MIN_VALS
    x /= gym_lib.ACROBOT_MAX_VALS - gym_lib.ACROBOT_MIN_VALS
    x = 2.0 * x - 1.0  # Rescale in range [-1, 1].
    #x = nn.Dense(x, features=512, kernel_init=initializer)
    x = NoisyNetwork(x, features=512, bias=True)
    x = jax.nn.relu(x)
    #x = nn.Dense(x, features=512, kernel_init=initializer)
    x = NoisyNetwork(x,features=512, bias=True)
    x = jax.nn.relu(x)

    #adv = nn.Dense(x, features=num_actions * num_atoms, kernel_init=initializer)
    adv = NoisyNetwork(x,features=num_actions * num_atoms, bias=True)
    #value = nn.Dense(x, features= num_atoms, kernel_init=initializer)
    value = NoisyNetwork(x, features=num_atoms, bias=True)

    adv = adv.reshape((adv.shape[0], num_actions, num_atoms))
    value = value.reshape((value.shape[0], 1, num_atoms))

    q_atoms = value + (adv - (jnp.mean(adv, -1, keepdims=True)))

    probabilities = nn.softmax(q_atoms)
    q_values = jnp.sum(support * probabilities, axis=2)

    return atari_lib.RainbowNetworkType(q_values, q_atoms, probabilities)