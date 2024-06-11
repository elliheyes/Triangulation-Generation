"""
agent.py
--------
"""

import random
import numpy as np
from collections import deque
from tensorflow import keras as tfk


class Agent(object):
    def __init__(self, model: tfk.Sequential, target_model: tfk.Sequential, buffer_size: int):
        """
        @param model:           Specifies the main model used for training.
        @param target_model:    Specifies the target model which is updated after some number of training steps. Ensures stability.
        @param buffer_size:     Size of the replay buffer.
        """
        self._memory = deque(maxlen=buffer_size)
        self._gamma = 0.95
        self._epsilon = 1.0
        self._epsilon_min = 0.01
        self._epsilon_decay = 0.995
        self._policy_lr = 0.7

        self._model = model
        self._target_model = target_model

        self._num_actions = model.output_shape[-1]
        self.update_target_model()

    def store(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, terminated: bool):
        """
        Stores (s_t, a_t, r_t, s_t+1, terminated) into memory.
        """
        self._memory.append([state, action, reward, next_state, terminated])

    @property
    def current_memory_size(self):
        """
        @return Size of the part of the buffer that has been filled.
        """
        return len(self._memory)

    def update_target_model(self):
        """
        Copies the weights of the main model into target model.
        """
        self._target_model.set_weights(self._model.get_weights())

    def act(self, state: np.ndarray) -> int:
        """
        @return integer corresponding to an action.
        """
        if np.random.rand() <= self._epsilon:
            return random.randrange(self._num_actions)

        state = np.expand_dims(state.flatten(), axis=0)
        Q_a = self._model(state)
        return np.argmax(Q_a)

    def replay(self, batch_size):
        """
        Replays the buffer and trains the model.
        @param batch_size: Specifies the number of elements to be sampled from memory during replay.
        @return Training history.
        """
        minibatch = random.sample(self._memory, batch_size)

        states_batch = []
        next_states_batch = []
        for state, _, _, next_state, _ in minibatch:
            states_batch.append(state)
            next_states_batch.append(next_state)
        states_batch = np.asarray(states_batch)
        next_states_batch = np.asarray(next_states_batch)

        Q_s = self._model.predict(states_batch, verbose = 0)
        Q_s_next = self._target_model.predict(next_states_batch, verbose = 0)

        X = []
        Y = []

        for idx, (state, action, reward, next_state, terminated) in enumerate(minibatch):
            maxQ_next = reward if terminated else (
                reward + self._gamma * np.max(Q_s_next[idx]))

            Q_s[idx][action] = (1.0 - self._policy_lr)*Q_s[idx][action] + self._policy_lr * maxQ_next

            X.append(state)
            Y.append(Q_s[idx])


        history = self._model.fit(np.asarray(X), np.asarray(Y), batch_size = batch_size, shuffle = True, verbose = 0)
        if self._epsilon > self._epsilon_min:
            self._epsilon *= self._epsilon_decay

        return history
