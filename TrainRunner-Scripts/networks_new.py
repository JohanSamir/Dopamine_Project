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


#gin.constant('jax_networks.CARTPOLE_OBSERVATION_DTYPE', jnp.float64)
#gin.constant('jax_networks.ACROBOT_OBSERVATION_DTYPE', jnp.float64)


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

    #q_values = nn.Dense(x, features=num_actions, kernel_init=initializer)
    # https://jax.readthedocs.io/en/latest/_modules/jax/nn/functions.html (JAX Mean)

    #q_values = val + (adv - (jnp.mean(adv, 1, keepdims=True)))
    q_values = val + (adv - (jnp.mean(adv, -1, keepdims=True)))
    return atari_lib.DQNNetworkType(q_values)


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
class CartpoleRainbowDuelingNetwork(nn.Module):
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
    x = nn.Dense(x, features=512, kernel_init=initializer)
    x = jax.nn.relu(x)
    x = nn.Dense(x, features=512, kernel_init=initializer)
    x = jax.nn.relu(x)

    adv = nn.Dense(x, features=num_actions * num_atoms, kernel_init=initializer)
    value = nn.Dense(x, features= num_atoms, kernel_init=initializer)
    
    adv = adv.reshape((adv.shape[0], num_actions, num_atoms))
    value = value.reshape((value.shape[0], 1, num_atoms))

    q_atoms = value + (adv - (jnp.mean(adv, -1, keepdims=True)))

    #probabilities = nn.softmax(logits)
    probabilities = nn.softmax(q_atoms)
    q_values = jnp.sum(support * probabilities, axis=2)

    return atari_lib.RainbowNetworkType(q_values, q_atoms, probabilities)
