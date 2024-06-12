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
from cytools.triangulation import Triangulation
from notebooks.integer_rref import i4mat_rref


class Environment(object):
    @abc.abstractmethod
    def random_state(self):
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
    def act(self, state, action: int) -> Tuple[List, float]:
        """
        @param state: Current state.
        @param action: Index of the action using which to act on the state.

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

    def _flatten_state(self, state):
        return np.concatenate([*map(lambda x: np.asarray(x).reshape(-1), state)])

    def random_state(self):
        states = [*map(lambda x: x.random_state(), self._environments)]
        if self._state_shapes is None:
            self._state_shapes = [*map(lambda x: np.asarray(x).shape, states)]
        return self._flatten_state(states)

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
        return r_val

    def act(self, state, action):
        state = self._restore_shape(state)
        # Map action over the list.
        action_nums = [*map(lambda x: x.num_actions, self._environments)]

        # idx = x1 + x2 L1 + x3 L1 L2 + x4 L1 L2 L3

        # x1 = idx % L1
        # idx = (idx - x1) / L1
        # # idx = x2 + x3 L2 + ...
        # x2 = idx % L2
        # idx = (idx - x2) / L2
        # ...

        actions = []
        action_idx = action
        for i in range(len(action_nums)):
            actions.append(action_idx % action_nums[i])
            action_idx = (action_idx - actions[-1])//action_nums[i]

        new_states = []
        r_val = (0.0, True)
        for s_curr, a_curr, env_curr in zip(state, actions, self._environments):
            new_state, r_val_curr = env_curr.act(s_curr, a_curr)

            new_states.append(new_state)
            r_val = self._combine_r_vals(
                r_val_curr, r_val)
        return self._flatten_state(new_states), r_val

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
        for pt_id in np.asarray(state, np.int32):
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


class HTriangulationEnvironment(Environment):
    def __init__(self, polytope: Polytope, step_size: float = 0.5):
        self._p = polytope
        self._num_actions = polytope.points().shape[0]
        self._step_size = step_size

    def get_triangulation(self, state):
        return self._p.triangulate(heights = state, check_heights = False)

    def random_state(self):
        return np.random.random(self._num_actions)

    def fitness(self, state):
        triang = self.get_triangulation(state)

        reward = -1.0
        if triang.is_fine():
            reward += 1.5
        if triang.is_star():
            reward += 1.5
        # # Always true
        # if triang.is_regular():
        #     reward += 1.0

        return reward, reward == 2.0

    def act(self, state, action):
        shift = np.zeros(self._num_actions)
        sgn = 1.0
        if action >= self._num_actions:
            sgn = -1.0
        shift[action % self._num_actions] = sgn*self._step_size

        new_state = state + shift
        return new_state, self.fitness(new_state)

    @property
    def num_actions(self):
        return self._num_actions*2


# TODO: Move into the class
def get_indices(p, subp):
    return [np.where(np.all(v == p, axis=1))[0][0] for v in subp]

def boundary(simplices):
    dsimplices = []
    for s in simplices:
        for i in range(1, len(s)):
            dsimplices.append(
                np.append(s[:i], s[(i+1):]))

    dsimplices = np.asarray(dsimplices)
    return dsimplices

def restrict(dsimplices, subp_vertices):
    sub_simplices = np.where(np.all(np.isin(dsimplices, subp_vertices), axis=1))[0]
    sub_simplices = dsimplices[sub_simplices]

    local_sub_simplices = []
    for s in sub_simplices:
        local_s = []
        for v in s:
            local_s.append(np.where(subp_vertices == v)[0][0])
        local_sub_simplices.append(local_s)
    return np.unique(local_sub_simplices, axis=0)

def project(vertices):
    vertices_copy = np.array(vertices, copy=True)
    W = np.asarray(i4mat_rref(vertices.shape[0], vertices.shape[1], vertices_copy)[0]).astype(np.float64)
    local_vertices = np.round(vertices@np.linalg.pinv(W))
    idx = np.argwhere(np.all(local_vertices[..., :] == 0, axis=0))

    return np.delete(local_vertices, idx, axis=1)

def compose(x, *args):
    out = x
    for f in args:
        out = f(out)
    return out


class FibrationEnvironment(MultiEnvironment):
    def __init__(self, polytope: Environment, fibration_dim: int):
        super().__init__(environments = [
            HTriangulationEnvironment(polytope),
            SubpolytopeEnvironment(polytope, fibration_dim)
        ])
        self._p = polytope
        self._d = fibration_dim

    def get_structure(self, state) -> Tuple[Triangulation, Polytope]:
        """
        @return A tuple consisting of:
            1. Triangulation corresponding to the state.
            2. A subpolytope corresponding to the state.
        """
        state = self._restore_shape(state)
        return self._environments[0].get_triangulation(state[0]),\
              Polytope(self._environments[1].intersect(state[1]))

    def _compatibility_fitness(self, state):
        done = True
        state = self._restore_shape(state)
        r_compatibility = 0.0
        if done:
            triang = self._environments[0].get_triangulation(state[0])
            vertices = self._environments[1].intersect(state[1])

            subsimplices = restrict(
                dsimplices = compose(triang.simplices(), *([boundary]*(self._p.dimension() - self._d))), #boundary(t.simplices()),
                subp_vertices = get_indices(
                    self._p.points(), Polytope(vertices).points()))

            triang_pts = project(Polytope(vertices).points())

            subpoly = Polytope(np.asarray(triang_pts, np.int32))
            triang_pts_idx = subpoly.points_to_indices(triang_pts)

            t_sub = Triangulation(
                poly = subpoly,
                pts = triang_pts_idx,
                simplices = subsimplices,
                check_input_simplices=False)

            # Verify fine condition
            t_valid = t_sub.is_valid()
            if t_valid:
                for cond in [t_sub.is_fine(), t_sub.is_regular(), t_sub.is_star()]:
                    r_compatibility += (1 if cond else -1)
                    t_valid = t_valid and cond
            done &= t_valid

            return r_compatibility, done

    def fitness(self, state):
        r_val = super().fitness(state)
        if r_val[-1]:
            r_val = self._combine_r_vals(
                r_val,
                self._compatibility_fitness(state))
        return r_val

    def act(self, state, action):
        new_state, r_val = super().act(state, action)
        if r_val[-1]:
            r_val = self._combine_r_vals(
                r_val,
                self._compatibility_fitness(new_state))
        return new_state, r_val


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
