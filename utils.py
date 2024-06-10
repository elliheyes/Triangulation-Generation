"""
utils.py
--------
"""
from tqdm import trange

from copy import deepcopy

import numpy as np
from agent import Agent
from environment import Environment


def train_agent(env: Environment, agent: Agent, num_epochs = 1024, max_steps = 32, batch_size = 128, min_memory_size = 1024, verbosity = 0) -> np.ndarray:
    """
    @return Array of (epochs, loss_vals). Shape: (Num_epochs, 2).
    """
    steps_to_update_target_model = 0
    epochs = trange(num_epochs) if verbosity > 0 else range(num_epochs)
    loss_vals = []

    for epoch in epochs:
        state = env.random_state()
        terminated = False
        total_loss = 0.0
        count = 0

        for _ in range(max_steps):
            steps_to_update_target_model += 1
            action = agent.act(state)
            new_state, (fitness, done) = env.act(state, action)

            reward = 0
            if done:
                reward = 10
                terminated = True
            else:
                reward = -1. + (fitness - env.fitness(state)[0])

            agent.store(state, action, reward, new_state, terminated)

            if terminated or (steps_to_update_target_model % 4 == 0):
                if agent.current_memory_size > min_memory_size:
                    history = agent.replay(batch_size)
                    total_loss += history.history['loss'][-1]
                    count += 1

            # Move
            state = deepcopy(new_state)

            if terminated:
                if steps_to_update_target_model >= 100:
                    agent.update_target_model()
                    steps_to_update_target_model = 0

            if terminated:
                break
        if verbosity > 0:
            epochs.set_description(f"Epoch {epoch}, loss = {total_loss / (count+1e-10):.02f}")
        if agent.current_memory_size > min_memory_size:
            loss_vals.append([epoch, total_loss / (count+1e-10)])
    return np.asarray(loss_vals)


def walk_agent(state, env: Environment, agent: Agent, max_steps = 32):
    path = [state]
    for _ in range(max_steps):
        action = agent.act(state)
        state, (_, done) = env.act(state, action)
        path.append(state)
        if done:
            break
    return path
