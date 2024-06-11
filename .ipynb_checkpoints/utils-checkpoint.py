"""
utils.py
--------
"""

import ppl
import random
import numpy as np
from copy import deepcopy
from flint import fmpz_mat
from cytools import Polytope
from cytools.cone import Cone
from cytools.utils import gcd_list
from integer_rref import i4mat_rref


def get_two_face_triangs(poly):
    """ Input: cytools polyotpe
        Output: 2d list of all two-face triangulations """
    two_face_triangs = []
    for face in poly.faces(2):
        F = Polytope(face.points())
        
        # get all fine regular triangulations of the two-face
        Ts = F.all_triangulations(only_regular=True, only_star=False, only_fine=True, include_points_interior_to_facets=True, as_list=True)
        
        # get list of simplices for two-face triangulations
        Ts_simps = []
        for T in Ts:
            new_simp = [sorted([poly.points_to_labels(F.points()[i]) for i in simp]) for simp in T.simps()]
            Ts_simps.append(new_simp)
        
        two_face_triangs.append(Ts_simps)
    
    return two_face_triangs


def random_T_state(two_face_triangs, max_num_triangs):
    """ Input: - 2d list of all two-face triangulations
               - maximum number of two-face triangulations
        Output: random 2d bitlist state                       """
    bitlist = [[0 for j in range(max_num_triangs)] for i in range(len(two_face_triangs))]
    
    # choose random triangulation for each two-face
    for i in range(len(two_face_triangs)):
        j = random.choice(range(len(two_face_triangs[i])))
        bitlist[i][j] = 1

    return bitlist


def bits_to_simps(bitlist, two_face_triangs):
    """ Input: - bitlist describing triangulation of two-faces 
               - list of all two-face triangulation simplices
        Output: the list of two-face triangulation simplices    """
    two_simps = []
    for i in range(len(bitlist)):
        for j in range(len(bitlist[i])):
            if bitlist[i][j] == 1:
                two_simps.append(two_face_triangs[i][j])
    return two_simps


def flatten_T_state(state):
    """ Input: 2d bitlist describing triangulation of two-faces 
        Output: 1d flattened bitlist state                   """
    return np.array(state).flatten()


def restore_T_state(flat_state, max_num_triangs):
    """ Input: 1d flattened bitlist state describing triangulation of two-faces
        Output: 2d bitlist state                                                """
    return np.array(flat_state).reshape((int(len(flat_state)/max_num_triangs),max_num_triangs))


def random_SP_state(poly, fib_dim):
    """ Input: - cytools polytope
               - fibration dimension 
        Output: random subpolytope state   """
    
    return np.random.choice(poly.points()[1:].shape[0], size=(fib_dim))


def get_T_actions(two_face_triangs):
    """ Input: 2d list of all two-face triangulations
        Output: the list of two possible triangulation actions  """
    # [i,j]: choose the j-th triangulation of the i-th 2-face
    action_list = []
    for i in range(len(two_face_triangs)):
        for j in range(len(two_face_triangs[i])):
            if len(two_face_triangs[i]) > 1:
                action_list.append([i,j])
    return action_list


def T_act(state, action):
    """ Input: - triangulation state
               - action 
        Output: a new state obtained by acting on the input state with the action """
    new_state = deepcopy(state)
    new_state[action[0]] = [0 for i in range(len(state[action[0]]))]
    new_state[action[0]][action[1]] = 1
    return new_state


def SP_act(state, action, num_points):
    """
    Input: state-action pair
    Output: new state obtained by acting on the old state with the action 
    """
    new_state = deepcopy(state)
    new_state[action] = ((new_state[action] + 1) % num_points)
    return new_state

# two_face_secondary_cone function adapted from secondary cone function in CYtools - https://cy.tools/docs/documentation/triangulation#secondary_cone
def two_face_secondary_cone(poly, two_face, simps):
    """ Input: - cytools polytpoe
               - two-face 
               - two-face triangulation simplices
        Output: secondary cone of facet embedded in total polytope height space """
    face = Polytope(two_face.points())
    
    # define the extended point lists of the two-face and the ambient polytope
    two_pts_ext = [list(pt)+[1,] for pt in face.points(optimal=True)]
    pts_ext = [list(pt)+[1,] for pt in poly.points(optimal=True)]
    
    # get the two-face triangulation simplices in terms of two-face indices
    two_simps = [[face.points_to_labels(poly.points()[simp[i]]) for i in range(3)] for simp in simps]
    two_simps = [set(s) for s in two_simps]
    
    full_v = np.zeros(len(pts_ext), dtype=int)
    m = np.zeros((2+1, 2+2), dtype=int)
    null_vecs = set()
    for i in range(len(two_simps)):
        for j in range(i+1,len(two_simps)):
            # define the simplices
            two_s1 = two_simps[i]
            two_s2 = two_simps[j]

            # eunsre that the simplices have a large enough intersection
            two_comm_pts = two_s1 & two_s2
            if len(two_comm_pts) != 2:
                continue
 
            two_diff_pts = list(two_s1 ^ two_s2)
            two_comm_pts = list(two_comm_pts)
            for k,pt in enumerate(two_diff_pts):    m[:,k] = two_pts_ext[pt]
            for k,pt in enumerate(two_comm_pts):    m[:,k+2] = two_pts_ext[pt]
        
            # calculate nullspace/hyperplane inequality
            v = fmpz_mat(m.tolist()).nullspace()[0]
            v = np.array(v.transpose().tolist()[0], dtype=int)

            # ensure the sign is correct
            if v[0] < 0:
                v *= -1

            # reduce the vector
            g = gcd_list(v)
            if g != 1:
                v //= g

            # construct the full vector (including all points)
            for k,pt in enumerate(two_diff_pts):    full_v[poly.points_to_labels(face.points()[pt])] = v[k]
            for k,pt in enumerate(two_comm_pts):    full_v[poly.points_to_labels(face.points()[pt])] = v[k+2]
            
            null_vecs.add(tuple(full_v))
            
            for k,pt in enumerate(two_diff_pts):    full_v[poly.points_to_labels(face.points()[pt])] = 0
            for k,pt in enumerate(two_diff_pts):    full_v[poly.points_to_labels(face.points()[pt])] = 0
    
    null_vecs = list(null_vecs)
    if len(null_vecs):
        hyps = null_vecs
    else:  
        hyps = np.zeros((0,len(pts_ext)), dtype=int)
    sec_cone = Cone(hyperplanes=hyps,check=False)
    
    return sec_cone


def reduce_polytope(vertices):
    """ Input: vertex matrix
        Output: reduced vertex matrix """
    vertices_copy = np.array(vertices, copy=True)
        
    # find integer reduced echelon form of vertex matrix
    W = np.asarray(i4mat_rref(vertices.shape[0], vertices.shape[1], vertices_copy)[0]).astype(np.float64)
        
    # transform vertices
    local_vertices = np.round(vertices@np.linalg.pinv(W))
        
    # find and delete null entries
    idx = np.argwhere(np.all(local_vertices[..., :] == 0, axis=0))

    return np.delete(local_vertices, idx, axis=1)


def intersect(points, inds, dim):
    """ Input: list of points, list of point indices
        Output: ... """
    vertices_basis = []
    for pt_id in inds:
        vertices_basis.append(points[pt_id])
    vertices_basis = np.asarray(vertices_basis)
    
    vertices = []
    for pt in points:
        if np.linalg.matrix_rank(np.append(vertices_basis, [pt], axis=0)) == dim:
            vertices.append(pt)
    
    return np.asarray(vertices)


def T_fitness(poly, two_face_triangs, state): 
    """ Input: - cytools polytope
               - list of two-face triangulation simplicies
               - triangulation state
        Output: FRST fitness value between 0 and 1            """
    two_face_simps = bits_to_simps(state, two_face_triangs)
        
    # compute the secondary cones of all the non-simplicial two-face triangulations
    Cs = []
    for i in range(len(two_face_simps)):
        if len(two_face_simps[i]) > 1:
            Cs.append(two_face_secondary_cone(poly, poly.faces(2)[i], two_face_simps[i]))
    
    # compute the intersection of the secondary cones 
    inters = Cs[0]
    for i in range(1,len(Cs)):
        inters = inters.intersection(Cs[i])
    
    # check the regularity condition
    cs = ppl.Constraint_System()
    vrs = [ppl.Variable(i) for i in range(inters._ambient_dim)]
    for h in inters._hyperplanes:
        cs.insert(sum(h[i]*vrs[i] for i in range(inters._ambient_dim))>= 0)
    cone = ppl.C_Polyhedron(cs)
     
    fitness = 1 - abs(cone.affine_dimension()-inters.ambient_dimension())/inters.ambient_dimension()
    
    return fitness


def SP_fitness(points, state, fib_dim):
    vertices = intersect(points, state, fib_dim)
    if len(vertices.shape) < 2:
        fitness = 0
    else:
        vertices_reduced = np.asarray(reduce_polytope(vertices), np.int32)
        if vertices_reduced.size != 0:
            p_reduced = Polytope(vertices_reduced)
            refl = (1 if p_reduced.is_reflexive() else 0)
            dim = abs(p_reduced.dimension()-fib_dim)/max(p_reduced.dimension(),fib_dim)
            fitness = 0.5*refl + 0.5*(1-dim)
        else:
            fitness = 0
    
    return fitness
