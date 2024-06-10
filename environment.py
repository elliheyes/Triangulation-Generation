"""
environment.py
--------------
"""

import abc
from copy import deepcopy
from typing import Tuple, List

import numpy as np
from cytools import Polytope
from notebooks.integer_rref import i4mat_rref


class Environment(object):
    @abc.abstractmethod
    def R(self, state) -> Tuple[float, bool]:
        """Computes fitness function.

        @param state: Specifies current state.

        @return Tuple (reward, terminated).
        """

    @abc.abstractmethod
    def act(self, state, action) -> Tuple[List, float]:
        """
        @param state: Current state.
        @param action: Action using which to act on the state.

        @return Tuple (next state, fitness of the next state).
        """

    @property
    @abc.abstractmethod
    def num_actions(self):
        """
        @return Number of actions.
        """


class MultiEnvironment(Environment):
    def __init__(self, environments: List[Environment]):
        self._environments = environments

    @staticmethod
    def _combine_r_vals(r_val_1, r_val_2):
        return r_val_1[0] + r_val_2[0], r_val_1[1] & r_val_2[1]

    def R(self, state) -> Tuple[float, bool]:
        """
        @params state: List of states.
        """
        r_val = (0.0, True)

        for i, environment in enumerate(self._environments):
            r_val_curr = environment.R(state[i])
            r_val = self._combine_r_vals(
                r_val_curr, r_val)
            # r_val += r_val_curr
            # terminated &= terminated_curr

        return r_val

    def act(self, state, action):
        new_states = []
        r_val = (0.0, True)
        for s_curr, a_curr, env_curr in zip(state, action, self._environments):
            new_state, r_val_curr = env_curr.act(s_curr, a_curr)

            new_states.append(new_state)
            r_val = self._combine_r_vals(
                r_val_curr, r_val)
        return new_states, r_val

    def add(self, other: Environment):
        """
        Adds a new environment.
        """
        self._environments.append(other)

    def __add__(self, other: Environment):
        self.add(other)


class SubpolytopeEnvironment(Environment):
    def __init__(self, polytope: Polytope, fibration_dim: int):
        self._p = polytope
        self._points = polytope.points()[1:]
        self._d = fibration_dim
        self._num_actions = fibration_dim

    def random_state(self):
        return np.random.choice(self._points.shape[0], size=(self._d))

    @staticmethod
    def reduce_polytope(vertices):
        vertices_copy = np.array(vertices, copy=True)
        W = np.asarray(i4mat_rref(vertices.shape[0], vertices.shape[1], vertices_copy)[0]).astype(np.float64)
        local_vertices = np.round(vertices@np.linalg.pinv(W))
        idx = np.argwhere(np.all(local_vertices[..., :] == 0, axis=0))

        return np.delete(local_vertices, idx, axis=1)

    def intersect(self, state):
        vertices_basis = []
        for pt_id in state:
            vertices_basis.append(self._points[pt_id])

        vertices_basis = np.asarray(vertices_basis)
        vertices = []
        for pt in self._points:
            if np.linalg.matrix_rank(
                np.append(vertices_basis, [pt], axis=0)) == self._d:
                vertices.append(pt)
        return np.asarray(vertices)

    def R(self, state):
        vertices = self.intersect(state)

        reward = 0.0
        reward -= (np.linalg.matrix_rank(vertices) - self._d)**2

        if len(vertices.shape) < 2:
            reward -= 10.
        else:
            vertices_reduced = np.asarray(self.reduce_polytope(vertices), np.int32)
            if vertices_reduced.size != 0:
                p_reduced = Polytope(vertices_reduced)
                reward += (1.0 if p_reduced.is_reflexive() else -1.0)
                reward -= (p_reduced.dimension() - self._d)**2
            else:
                reward -= 10
            # try:
            #     p_reduced = Polytope(np.asarray(self.reduce_polytope(vertices), np.int32))
            #     reward += (1.0 if p_reduced.is_reflexive() else -1.0)
            #     reward -= (p_reduced.dimension() - self._d)**2
            # except:
            #     reward -= 1

        return reward, reward > 0

    def act(self, state, action):
        new_state = deepcopy(state)
        new_state[action] = ((new_state[action] + 1) % (self._points.shape[0]))

        return new_state, self.R(new_state)

    @property
    def num_actions(self):
        return self._d

    @property
    def all_actions(self):
        return [i for i in range(self._d)]


if __name__ == "__main__":
    p = Polytope([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1],
        [-1,-1,0,0],
        [-1,-1,-1,-1]])
    subpoly = SubpolytopeEnvironment(p, 2)
    state = subpoly.random_state()
    print(subpoly.R(state))

    multi = MultiEnvironment([subpoly, subpoly])
    print(multi.R([state, state]))