"""
environment.py
--------------
"""

import abc
from copy import deepcopy
from functools import reduce
from typing import Tuple, List

from Elli import utils

import numpy as np
from cytools import Polytope
from notebooks.integer_rref import i4mat_rref


class Environment(object):
    @abc.abstractmethod
    def random_state():
        """
        Generates a random state.
        """

    @abc.abstractmethod
    def fitness(self, state) -> Tuple[float, bool]:
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
        self._state_shapes = None

    def _restore_shape(self, flat_state):
        if self._state_shapes is None:
            self._state_shapes = [*map(lambda x: np.asarray(x.random_state()).shape, self._environments)]
        num_envs = len(self._environments)

        states = []
        shift_counter = 0
        for i in range(num_envs):
            block_size = np.prod(self._state_shapes[i])
            states.append(flat_state[shift_counter:shift_counter+block_size].reshape(self._state_shapes[i]))
            shift_counter += block_size
        return states

    def random_state(self):
        states = [*map(lambda x: x.random_state(), self._environments)]
        if self._state_shapes is None:
            self._state_shapes = [*map(lambda x: np.asarray(x).shape, states)]
        return np.concatenate([*map(lambda x: np.asarray(x).reshape(-1), states)])

    @staticmethod
    def _combine_r_vals(r_val_1, r_val_2):
        return r_val_1[0] + r_val_2[0], r_val_1[1] & r_val_2[1]

    def fitness(self, state) -> Tuple[float, bool]:
        """
        @params state: List of states.
        """
        state = self._restore_shape(state)
        r_val = (0.0, True)

        for i, environment in enumerate(self._environments):
            r_val_curr = environment.fitness(state[i])
            r_val = self._combine_r_vals(
                r_val_curr, r_val)

        # TODO compatibility.
        # NOTE: In some cases doing one search depends if the other one is valid.
        # TODO: make it into a graph with (u->v)∈E if v depends on u. Pass path to root.

        return r_val

    def act(self, state, action):
        state = self._restore_shape(state)

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

    def __iadd__(self, other: Environment):
        self.add(other)
        return self

    @property
    def num_actions(self):
        return reduce(lambda x,y: x*y, map(lambda x: x.num_actions, self._environments))


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

    def fitness(self, state):
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

        return new_state, self.fitness(new_state)

    @property
    def num_actions(self):
        return self._d

    @property
    def all_actions(self):
        return [i for i in range(self._d)]


class TriangulationEnvironment(Environment):
    def __init__(self, polytope: Environment):
        self._p = polytope
        self._two_face_Ts = utils.get_two_face_triangs(polytope)
        self._max_num_triangs = max(len(x) for x in self._two_face_Ts)
        self._action_list = utils.get_T_actions(self._two_face_Ts)

    def get_triangulation(self, state):
        return utils.combine_triangulation(self._p, self._two_face_Ts, state)

    def random_state(self):
        return utils.random_T_state(self._two_face_Ts, self._max_num_triangs)

    def fitness(self, state) -> Tuple[float, bool]:
        f_val = utils.T_fitness(self._p, self._two_face_Ts, state)
        return f_val, f_val == 1

    def act(self, state, action) -> Tuple[List, float]:
        new_state = utils.T_act(state, self._action_list[action])
        return new_state, self.fitness(new_state)

    @property
    def num_actions(self):
        return self._max_num_triangs



class FibrationEnvironment(MultiEnvironment):
    def __init__(self, polytope: Environment, fibration_dim: int):
        super().__init__(environments = [
            TriangulationEnvironment(polytope),
            SubpolytopeEnvironment(polytope, fibration_dim)
        ])
        self._p = polytope

    def fitness(self, state):
        fitness, done = super().fitness(state)

        # TODO compatibility
        # if done:
            # t_src = self._t_env.get_triangulation(state[0])
        return fitness, done

if __name__ == "__main__":
    from cytools import fetch_polytopes
    all_polys = fetch_polytopes(h11=15, lattice="N", limit=100, as_list=True)
    p = all_polys[15]

    t = p.triangulate()
    for t_1 in t.neighbor_triangulations():
        print(len(t_1.neighbor_triangulations()[0].neighbor_triangulations()))

    # t_env = TriangulationEnvironment(p)
    # state = t_env.random_state()
    # print(state)
    # print(t_env.fitness(state))
    # print(t_env.act(state, 1))
