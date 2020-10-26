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
"""Compact implementation of a simplified Rainbow agent in Jax.

Specifically, we implement the following components from Rainbow:

  * n-step updates;
  * prioritized replay; and
  * distributional RL.

These three components were found to significantly impact the performance of
the Atari game-playing agent.

Furthermore, our implementation does away with some minor hyperparameter
choices. Specifically, we

  * keep the beta exponent fixed at beta=0.5, rather than increase it linearly;
  * remove the alpha parameter, which was set to alpha=0.5 throughout the paper.

Details in "Rainbow: Combining Improvements in Deep Reinforcement Learning" by
Hessel et al. (2018).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
from absl import logging
import functools
from dopamine.jax import networks
from dopamine.jax.agents.dqn import dqn_agent
from dopamine.replay_memory import circular_replay_buffer, prioritized_replay_buffer
from flax import nn
from flax import optim
import gin
import jax
import jax.numpy as jnp
import numpy as onp
import tensorflow as tf


def huber_loss(targets, predictions, delta=1.0):
  x = jnp.abs(targets - predictions)
  return jnp.where(x <= delta,
                   0.5 * x**2,
                   0.5 * delta**2 + delta * (x - delta))

def mse_loss(targets, predictions):
  #(q_value - (expected_q_value.data)).pow(2).mean()
  return jnp.mean(jnp.power((targets - (predictions)),2))


@functools.partial(jax.jit, static_argnums=(7,8,9))
def train(target_network, optimizer, states, actions, next_states, rewards,
          terminals, cumulative_gamma,double_dqn, mse_inf):
  """Run the training step."""
  def loss_fn(model, target, mse_inf):
    q_values = jax.vmap(model, in_axes=(0))(states).q_values
    q_values = jnp.squeeze(q_values)
    replay_chosen_q = jax.vmap(lambda x, y: x[y])(q_values, actions)

    if mse_inf:
      print('mse_loss')
      loss = jnp.mean(jax.vmap(mse_loss)(target, replay_chosen_q))
    else:
      loss = jnp.mean(jax.vmap(huber_loss)(target, replay_chosen_q))
    return loss


  grad_fn = jax.value_and_grad(loss_fn)

  if double_dqn:
    print('Target_DDQN')
    target = target_DDQN(optimizer, target_network, next_states, rewards,  terminals, cumulative_gamma)
  else:
    target = target_q(target_network, next_states, rewards,  terminals, cumulative_gamma) 

  #optimizer.target ('Online')
  # target ('Target')
  loss, grad = grad_fn(optimizer.target, target, mse_inf)
  optimizer = optimizer.apply_gradient(grad)
  return optimizer, loss

def target_DDQN(model, target_network, next_states, rewards, terminals, cumulative_gamma):
  """Compute the target Q-value."""
  next_q_values = jax.vmap(model.target, in_axes=(0))(next_states).q_values
  next_q_values = jnp.squeeze(next_q_values)
  replay_next_qt_max = jnp.argmax(next_q_values, axis=1)
  next_q_state_values = jax.vmap(target_network, in_axes=(0))(next_states).q_values

  q_values = jnp.squeeze(next_q_state_values)
  replay_chosen_q = jax.vmap(lambda t, u: t[u])(q_values, replay_next_qt_max)
 
  return jax.lax.stop_gradient(rewards + cumulative_gamma * replay_chosen_q *
                               (1. - terminals))

def target_q(target_network, next_states, rewards, terminals, cumulative_gamma):
  """Compute the target Q-value."""
  q_vals = jax.vmap(target_network, in_axes=(0))(next_states).q_values
  q_vals = jnp.squeeze(q_vals)
  replay_next_qt_max = jnp.max(q_vals, 1)

  # Calculate the Bellman target value.
  #   Q_t = R_t + \gamma^N * Q'_t+1
  # where,
  #   Q'_t+1 = \argmax_a Q(S_t+1, a)
  #          (or) 0 if S_t is a terminal state,
  # and
  #   N is the update horizon (by default, N=1).
  return jax.lax.stop_gradient(rewards + cumulative_gamma * replay_next_qt_max *
                               (1. - terminals))

#@gin.configurable
#@functools.partial(jax.jit, static_argnums=())
def identity_epsilon(unused_decay_period, unused_step, unused_warmup_steps,
                     epsilon):
  return epsilon

@gin.configurable
class JaxDQNAgentNew(dqn_agent.JaxDQNAgent):
  """A compact implementation of a simplified Rainbow agent."""

  def __init__(self,
               num_actions,
               observation_shape=dqn_agent.NATURE_DQN_OBSERVATION_SHAPE,
               observation_dtype=dqn_agent.NATURE_DQN_DTYPE,
               stack_size=dqn_agent.NATURE_DQN_STACK_SIZE,
               network=networks.NatureDQNNetwork,

               minatar = True,
               env = "CartPole", 
               normalize_obs = True,
               hidden_layer=2, 
               neurons=512,
               prioritized=False,
               noisy = False,
               dueling = False,
               double_dqn=False,
               mse_inf=False,

               gamma=0.99,
               update_horizon=1,
               min_replay_history=20000,
               update_period=4,
               target_update_period=8000,
               epsilon_fn=dqn_agent.linearly_decaying_epsilon,
               epsilon_train=0.01,
               epsilon_eval=0.001,
               epsilon_decay_period=250000,
               eval_mode=False,
               max_tf_checkpoints_to_keep=4,
               optimizer='adam',
               summary_writer=None,
               summary_writing_frequency=500,
               allow_partial_reload=False):
    """Initializes the agent and constructs the necessary components.

    Args:
      num_actions: int, number of actions the agent can take at any state.
      observation_shape: tuple of ints or an int. If single int, the observation
        is assumed to be a 2D square.
      observation_dtype: DType, specifies the type of the observations. Note
        that if your inputs are continuous, you should set this to jnp.float32.
      stack_size: int, number of frames to use in state stack.
      network: flax.nn Module that is initialized by shape in _create_network
        below. See dopamine.jax.networks.RainbowNetwork as an example.
      num_atoms: int, the number of buckets of the value function distribution.
      vmax: float, the value distribution support is [-vmax, vmax].
      gamma: float, discount factor with the usual RL meaning.
      update_horizon: int, horizon at which updates are performed, the 'n' in
        n-step update.
      min_replay_history: int, number of transitions that should be experienced
        before the agent begins training its value function.
      update_period: int, period between DQN updates.
      target_update_period: int, update period for the target network.
      epsilon_fn: function expecting 4 parameters:
        (decay_period, step, warmup_steps, epsilon). This function should return
        the epsilon value used for exploration during training.
      epsilon_train: float, the value to which the agent's epsilon is eventually
        decayed during training.
      epsilon_eval: float, epsilon used when evaluating the agent.
      epsilon_decay_period: int, length of the epsilon decay schedule.
      replay_scheme: str, 'prioritized' or 'uniform', the sampling scheme of the
        replay memory.
      optimizer: str, name of optimizer to use.
      summary_writer: SummaryWriter object for outputting training statistics.
        Summary writing disabled if set to None.
      summary_writing_frequency: int, frequency with which summaries will be
        written. Lower values will result in slower training.
      allow_partial_reload: bool, whether we allow reloading a partial agent
        (for instance, only the network parameters).
    """
    # We need this because some tools convert round floats into ints.
    self._minatar = minatar
    self._env = env 
    self._normalize_obs = normalize_obs
    self._hidden_layer = hidden_layer
    self._neurons=neurons 
    self._prioritized=prioritized
    self._noisy = noisy
    self._dueling = dueling
    self._double_dqn = double_dqn
    self._mse_inf = mse_inf


    self._rng = jax.random.PRNGKey(0)
    state_shape = observation_shape + (stack_size,)
    self.state = onp.zeros(state_shape)
    # _replay = self._build_replay_buffer_prioritized() if prioritized == True else self._build_replay_buffer())
    self._replay = self._build_replay_buffer_prioritized() if self._prioritized == True else self._build_replay_buffer()
    self._optimizer_name = optimizer
    self._build_networks_and_optimizer()

    super(JaxDQNAgentNew, self).__init__(
        num_actions=num_actions,
        observation_shape=observation_shape,
        observation_dtype=observation_dtype,
        stack_size=stack_size,
        network=network.partial(nnum_actions=num_actions,
                                minatar=self._minatar,
                                env=self._env,
                                normalize_obs=self._normalize_obs,
                                hidden_layer=self._hidden_layer, 
                                neurons=self._neurons,
                                noisy=self._noisy,
                                dueling=self._dueling),
        gamma=gamma,
        update_horizon=update_horizon,
        min_replay_history=min_replay_history,
        update_period=update_period,
        target_update_period=target_update_period,
        #epsilon_fn=epsilon_fn,
        epsilon_fn = identity_epsilon if self._noisy == True else epsilon_fn,
        epsilon_train=epsilon_train,
        #epsilon_train = 0 if self._noisy == True else epsilon_train,
        epsilon_eval=epsilon_eval,
        epsilon_decay_period=epsilon_decay_period,
        optimizer=optimizer,
        summary_writer=summary_writer,
        summary_writing_frequency=summary_writing_frequency,
        allow_partial_reload=allow_partial_reload)

  def _build_replay_buffer(self):
    #print('self.observation_shape:', self.gamma)
    #print('replay')
    """Creates the replay buffer used by the agent."""
    return circular_replay_buffer.OutOfGraphReplayBuffer(
        observation_shape=self.observation_shape,
        stack_size=self.stack_size,
        update_horizon=self.update_horizon,
        gamma=self.gamma,
        observation_dtype=self.observation_dtype)

  def _build_replay_buffer_prioritized(self):
    print('prioritized')
    """Creates the prioritized replay buffer used by the agent."""
    return prioritized_replay_buffer.OutOfGraphPrioritizedReplayBuffer(
        observation_shape=self.observation_shape,
        stack_size=self.stack_size,
        update_horizon=self.update_horizon,
        gamma=self.gamma,
        observation_dtype=self.observation_dtype)
    
  def _train_step(self):
    """Runs a single training step.

    Runs training if both:
      (1) A minimum number of frames have been added to the replay buffer.
      (2) `training_steps` is a multiple of `update_period`.

    Also, syncs weights from online_network to target_network if training steps
    is a multiple of target update period.
    """
    # Run a train op at the rate of self.update_period if enough training steps
    # have been run. This matches the Nature DQN behaviour.
    if self._replay.add_count > self.min_replay_history:
      if self.training_steps % self.update_period == 0:
        self._sample_from_replay_buffer()

        self.optimizer, loss = train(self.target_network,
                                     self.optimizer,
                                     self.replay_elements['state'],
                                     self.replay_elements['action'],
                                     self.replay_elements['next_state'],
                                     self.replay_elements['reward'],
                                     self.replay_elements['terminal'],
                                     self.cumulative_gamma,
                                     self._double_dqn,
                                     self._mse_inf)
        if self.prioritized == 'prioritized':
          # The original prioritized experience replay uses a linear exponent
          # schedule 0.4 -> 1.0. Comparing the schedule to a fixed exponent of
          # 0.5 on 5 games (Asterix, Pong, Q*Bert, Seaquest, Space Invaders)
          # suggested a fixed exponent actually performs better, except on Pong.
          probs = self.replay_elements['sampling_probabilities']
          loss_weights = 1.0 / jnp.sqrt(probs + 1e-10)
          loss_weights /= jnp.max(loss_weights)

          # Rainbow and prioritized replay are parametrized by an exponent
          # alpha, but in both cases it is set to 0.5 - for simplicity's sake we
          # leave it as is here, using the more direct sqrt(). Taking the square
          # root "makes sense", as we are dealing with a squared loss.  Add a
          # small nonzero value to the loss to avoid 0 priority items. While
          # technically this may be okay, setting all items to 0 priority will
          # cause troubles, and also result in 1.0 / 0.0 = NaN correction terms.
          self._replay.set_priority(self.replay_elements['indices'],
                                    jnp.sqrt(loss + 1e-10))

        if (self.summary_writer is not None and
            self.training_steps > 0 and
            self.training_steps % self.summary_writing_frequency == 0):
          summary = tf.compat.v1.Summary(value=[
              tf.compat.v1.Summary.Value(tag='HuberLoss', simple_value=loss)])
          self.summary_writer.add_summary(summary, self.training_steps)
      if self.training_steps % self.target_update_period == 0:
        self._sync_weights()

    self.training_steps += 1

  def _store_transition(self, last_observation, action, reward, is_terminal):
    """Stores an experienced transition.

    Pedantically speaking, this does not actually store an entire transition
    since the next state is recorded on the following time step.

    Args:
      last_observation: numpy array, last observation.
      action: int, the action taken.
      reward: float, the reward.
      is_terminal: bool, indicating if the current state is a terminal state.
    """
    if self.prioritized==True:
      priority = self._replay.sum_tree.max_recorded_priority
      self._replay.add(last_observation, action, reward, is_terminal, priority)
    else:
      self._replay.add(last_observation, action, reward, is_terminal)