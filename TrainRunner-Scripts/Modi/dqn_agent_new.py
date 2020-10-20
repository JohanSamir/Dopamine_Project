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
"""Compact implementation of a DQN agent in JAx."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import math

from absl import logging
#from dopamine.jax import networks
from dopamine.replay_memory import circular_replay_buffer, prioritized_replay_buffer
from flax import nn
from flax import optim
import gin
import jax
import jax.numpy as jnp
import numpy as onp
import tensorflow as tf

from dopamine.jax.agents.dqn import dqn_agent
from dopamine.jax import networks

NATURE_DQN_OBSERVATION_SHAPE = dqn_agent.NATURE_DQN_OBSERVATION_SHAPE
NATURE_DQN_DTYPE = jnp.uint8
NATURE_DQN_STACK_SIZE = dqn_agent.NATURE_DQN_STACK_SIZE

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


@functools.partial(jax.jit, static_argnums=(3, 4, 5, 6, 7, 9, 10))
def select_action(network, state, rng, num_actions, eval_mode,
                  epsilon_eval, epsilon_train, epsilon_decay_period,
                  training_steps, min_replay_history, epsilon_fn):
  """Select an action from the set of available actions.

  Chooses an action randomly with probability self._calculate_epsilon(), and
  otherwise acts greedily according to the current Q-value estimates.

  Args:
    network: Jax Module to use for inference.
    state: input state to use for inference.
    rng: Jax random number generator.
    num_actions: int, number of actions (static_argnum).
    eval_mode: bool, whether we are in eval mode (static_argnum).
    epsilon_eval: float, epsilon value to use in eval mode (static_argnum).
    epsilon_train: float, epsilon value to use in train mode (static_argnum).
    epsilon_decay_period: float, decay period for epsilon value for certain
      epsilon functions, such as linearly_decaying_epsilon, (static_argnum).
    training_steps: int, number of training steps so far.
    min_replay_history: int, minimum number of steps in replay buffer
      (static_argnum).
    epsilon_fn: function used to calculate epsilon value (static_argnum).

  Returns:
    rng: Jax random number generator.
    action: int, the selected action.
  """
  epsilon = jnp.where(eval_mode,
                      epsilon_eval,
                      epsilon_fn(epsilon_decay_period,
                                 training_steps,
                                 min_replay_history,
                                 epsilon_train))

  rng, rng1, rng2 = jax.random.split(rng, num=3)
  p = jax.random.uniform(rng1)
  return rng, jnp.where(p <= epsilon,
                        jax.random.randint(rng2, (), 0, num_actions),
                        jnp.argmax(network(state).q_values))


@gin.configurable
class JaxDQNAgentNew(dqn_agent.JaxDQNAgent):
  """A JAX implementation of the DQN agent."""

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
               epsilon_fn=None,
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

    Note: We are using the Adam optimizer by default for JaxDQN, which differs
          from the original NatureDQN and the dopamine TensorFlow version. In
          the experiments we have ran, we have found that using Adam yields
          improved training performance.

    Args:
      num_actions: int, number of actions the agent can take at any state.
      observation_shape: tuple of ints describing the observation shape.
      observation_dtype: jnp.dtype, specifies the type of the observations.
      stack_size: int, number of frames to use in state stack.
      network: Jax network to use for training.
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
      eval_mode: bool, True for evaluation and False for training.
      max_tf_checkpoints_to_keep: int, the number of TensorFlow checkpoints to
        keep.
      optimizer: str, name of optimizer to use.
      summary_writer: SummaryWriter object for outputting training statistics.
      summary_writing_frequency: int, frequency with which summaries will be
        written. Lower values will result in slower training.
      allow_partial_reload: bool, whether we allow reloading a partial agent
        (for instance, only the network parameters).
    """
    assert isinstance(observation_shape, tuple)
    logging.info('Creating %s agent with the following parameters:',
                 self.__class__.__name__)
    
    logging.info('\t observation_shape: %f', observation_shape)
    logging.info('\t observation_dtype: %f', observation_dtype)
    logging.info('\t stack_size: %f', stack_size)
    logging.info('\t gamma: %f', gamma)
    logging.info('\t update_horizon: %f', update_horizon)
    logging.info('\t min_replay_history: %d', min_replay_history)
    logging.info('\t update_period: %d', update_period)
    logging.info('\t target_update_period: %d', target_update_period)
    logging.info('\t epsilon_train: %f', epsilon_train)
    logging.info('\t epsilon_eval: %f', epsilon_eval)
    logging.info('\t epsilon_decay_period: %d', epsilon_decay_period)
    logging.info('\t optimizer: %s', optimizer)
    logging.info('\t max_tf_checkpoints_to_keep: %d',
                 max_tf_checkpoints_to_keep)

    
    super(JaxRainbowAgentNew, self).__init__(

	    self.num_actions = num_actions
	    self.minatar = minatar
	    self.env = env 
	    self.normalize_obs = normalize_obs
	    self.hidden_layer= hidden_layer
	    self.neurons=neurons 
	    self.noisy = noisy
	    self.dueling = dueling
	    self.observation_shape = tuple(observation_shape)
	    self.observation_dtype = observation_dtype
	    self.stack_size = stack_size
	    self.network = network.partial(num_actions=num_actions, minatar = minatar, env = env,normalize_obs = normalize_obs,
	                                  hidden_layer= hidden_layer, neurons=neurons, noisy=noisy,dueling=dueling)
	    self.double_dqn = double_dqn
	    self.mse_inf = mse_inf
	    self.gamma = gamma
	    self.update_horizon = update_horizon
	    self.cumulative_gamma = math.pow(gamma, update_horizon)
	    self.min_replay_history = min_replay_history
	    self.target_update_period = target_update_period
	    #self.epsilon_fn = epsilon_fn
	    self.epsilon_fn = dqn_agent.identity_epsilon if noisy == True else epsilon_fn
	    self.epsilon_train = epsilon_train
	    #self.epsilon_train = 0 if noisy == True else epsilon_train
	    self.epsilon_eval = epsilon_eval
	    self.epsilon_decay_period = epsilon_decay_period
	    self.update_period = update_period
	    self.eval_mode = eval_mode
	    self.training_steps = 0
	    self.summary_writer = summary_writer
	    self.summary_writing_frequency = summary_writing_frequency
	    self.allow_partial_reload = allow_partial_reload

	    self._rng = jax.random.PRNGKey(0)
	    print('observation_shape:',observation_shape,type(observation_shape))
	    state_shape = self.observation_shape + (stack_size,)
	    print('state_shape:',state_shape,type(state_shape))
	    self.state = onp.zeros(state_shape)
	    #self._replay = self._build_replay_buffer()
	    self.prioritized=prioritized
	    self._replay = self._build_replay_buffer_prioritized() if prioritized == True else self._build_replay_buffer()
	    self._optimizer_name = optimizer
	    self._build_networks_and_optimizer()

	    # Variables to be initialized by the agent once it interacts with the
	    # environment.
	    self._observation = None
	    self._last_observation = None)

	    #print('self.stack_size:',self.stack_size ,'stack_size:',stack_size )
	    #print('self.observation_shape',self.observation_shape)
	    #print('self.observation_dtype',self.observation_dtype)
	    #print('network',self.network)


  def _create_network(self, name):
    """Builds the convolutional network used to compute the agent's Q-values.

    Args:
      name: str, this name is passed to the Jax Module.

    Returns:
      network: Jax Model, the network instantiated by Jax.
    """
    _, initial_params = self.network.init(self._rng, name=name,
                                          x=self.state,
                                          num_actions=self.num_actions)
    return nn.Model(self.network, initial_params)


  def _build_replay_buffer(self):
    print('replay')
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


  def begin_episode(self, observation):
    """Returns the agent's first action for this episode.

    Args:
      observation: numpy array, the environment's initial observation.

    Returns:
      int, the selected action.
    """
    self._reset_state()
    self._record_observation(observation)

    if not self.eval_mode:
      self._train_step()

    self._rng, self.action = select_action(self.online_network,
                                           self.state,
                                           self._rng,
                                           self.num_actions,
                                           self.eval_mode,
                                           self.epsilon_eval,
                                           self.epsilon_train,
                                           self.epsilon_decay_period,
                                           self.training_steps,
                                           self.min_replay_history,
                                           self.epsilon_fn)
    self.action = onp.asarray(self.action)
    return self.action

 
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
                                     self.double_dqn,
                                     self.mse_inf)
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
