import ppl
import math
import random
import cytools
import numpy as np
from copy import deepcopy
from flint import fmpz_mat
from itertools import product
from cytools import Polytope
from cytools.cone import Cone
from cytools.utils import gcd_list
from cytools.triangulation import Triangulation
from itertools import combinations
from scipy.spatial import ConvexHull


def get_two_face_triangs(poly):
    """ Input: cytools polyotpe
        Output: 2d list of all two-face triangulations """

    # loop through each two-face 
    two_face_triangs = []
    for face in poly.faces(2):
        
        # define face as polytope object
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
    
    # initialise bitlist
    bitlist = [[0 for j in range(max_num_triangs)] for i in range(len(two_face_triangs))]
    
    # choose random triangulation for each two-face
    for i in range(len(two_face_triangs)):
        j = random.choice(range(len(two_face_triangs[i])))
        bitlist[i][j] = 1

    return bitlist


def random_T_V_state(two_face_triangs, max_num_triangs, h11):
    """ Input: - 2d list of all two-face triangulations
               - maximum number of two-face triangulations 
               - h11 value of CY
        Output: random 3d bitlist state encoding the triangulation and line bundle sum   """
    
    # determine the dimensions of the state bitlist
    dim1 = max(len(two_face_triangs),5)
    dim2 = max(max_num_triangs,h11)
    
    # randomly generate triangulation bitlist
    T = [[0 for j in range(dim2)] for i in range(dim1)]
    for i in range(len(two_face_triangs)):
        j = random.choice(range(len(two_face_triangs[i])))
        T[i][j] = 1
    
    # randomly generate line bundle sum
    V = [[0 for j in range(dim2)] for i in range(dim1)]
    for i in range(5):
        for j in range(h11):
            V[i][j] = random.randint(-4,4)
            
    state = [T,V]
    return state


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


def get_T_actions(two_face_triangs):
    """ Input: 2d list of all two-face triangulations
        Output: the list of two possible triangulation actions  """
    # [i,j]: choose the j-th triangulation of the i-th 2-face
    action_list = [1]
    for i in range(len(two_face_triangs)):
        for j in range(len(two_face_triangs[i])):
            if len(two_face_triangs[i]) > 1:
                action_list.append([i,j])
    return action_list


def get_T_V_actions(two_face_triangs, h11):
    """ Input: - 2d list of all two-face triangulations
               - h11 value of CY
        Output: - the list of all possible actions on the 
                  triangulation and line bundle sum      
                - the number of triangulation and bundle actions     """
    # [i,j]: choose the j-th triangulation of the i-th 2-face
    T_actions = []
    for i in range(len(two_face_triangs)):
        for j in range(len(two_face_triangs[i])):
            if len(two_face_triangs[i]) > 1:
                T_actions.append([0,i,j,0])
    
    # [i,j,k]: change the k^i_j by k
    V_actions = []
    for i in range(5):
        for j in range(h11):
            for k in [-1,1]:
                V_actions.append([1,i,j,k])
    
    # combine identity, triangulation and bundle actions
    action_list = [1] + T_actions + V_actions
    return action_list, [len(T_actions),len(V_actions)]


def T_act(state, action):
    """ Input: - triangulation state
               - action 
        Output: a new state obtained by acting on the input state with the action """
    new_state = deepcopy(state)
    if action != 1:
        new_state[action[0]] = [0 for i in range(len(state[action[0]]))]
        new_state[action[0]][action[1]] = 1
    return new_state


def T_V_act(state, action):
    """ Input: - triangulation-bundle state 
               - action 
        Output: a new state obtained by acting on the input state with the action """
    new_state = deepcopy(state)
    if action != 1:
        if action[0] == 0:
            new_state[0][action[1]] = [0 for i in range(len(state[0][action[1]]))]
            new_state[0][action[1]][action[2]] = 1
        else:
            if new_state[1][action[1]][action[2]]+action[3]>-5 and new_state[1][action[1]][action[2]]+action[3]<5:
                new_state[1][action[1]][action[2]] += action[3]
    return new_state


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


def T_fitness(poly, two_face_triangs, state): 
    """ Input: - cytools polytope
               - list of two-face triangulation simplicies
               - triangulation state
        Output: FRST fitness value between 0 and 1            """
    
    # covert bitlist to list of two-face simplices
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


def V_fitness(V, h11, c2, dijk): 
    """ Input: - state describing line bundle sum
               - h11 value of CY
               - second Chern class of CY
               - triple intersection numbers of CY
        Output: the reward value between 0 and 1 for the state """
    
    # check E8 embedding condition 
    embeddings = [0 for i in range(h11)]
    for i in range(h11):
        if V[0][i]+V[1][i]+V[2][i]+V[3][i]+V[4][i] == 0:
            embeddings[i] = 1
        
    # check anomaly cancellation condition
    anomalies = [0 for i in range(h11)]
    for i in range(h11):
        c2iV = 0
        for a in range(5):
            for j in range(h11):
                for k in range(h11):
                    c2iV += -0.5*dijk[i][j][k]*V[a][j]*V[a][k]
        if c2iV <= c2[i]:
            anomalies[i] = 1
                
    # check the poly-stability condition 
    Ms = []
    for a in range(5):
        M = [[0 for i in range(h11)] for j in range(h11)]
        for i in range(h11):
            for j in range(h11):
                for k in range(h11):
                    M[j][k] += dijk[i][j][k]*V[a][i]
        Ms.append(M)
    vs = [[i,j,k,l,m] for i,j,k,l,m in product(range(-2,3), repeat=5)]
    stabilities = [0 for a in range(len(vs))]
    for i in range(len(vs)):
        M = [[vs[i][0]*Ms[0][j][k]+vs[i][1]*Ms[1][j][k]+vs[i][2]*Ms[2][j][k]+vs[i][3]*Ms[3][j][k]+vs[i][4]*Ms[4][j][k] for j in range(h11)] for k in range(h11)]
        if (max(np.array(M).flatten()) > 0 and min(np.array(M).flatten()) < 0):
            stabilities[i] = 1
        
    fitness = (1/3)*sum(stabilities)/len(vs) + (1/3)*sum(anomalies)/h11 + (1/3)*sum(embeddings)/h11
    
    return fitness


def combine_triangulation(poly, two_face_triangs, state):
    """ Input: a polytope, the list of 2-face triangulations and the state
        Output: the combined triangulation                                  """
    
    # get the list facets
    facets = []
    for i in range(len(poly.facets())):
        facets.append(poly.points_to_labels(poly.facets()[i].points()))

    # get the list of two-faces 
    two_faces = []
    for i in range(len(poly.faces(2))):
        two_faces.append(poly.points_to_labels(poly.faces(2)[i].points()))
        
    # get the list of two-face simplices
    two_face_simps = [item for sublist in bits_to_simps(state, two_face_triangs) for item in sublist]

    # get list of edges from two-face simplices
    edges = []
    for i in range(len(two_face_simps)):
        com = list(combinations(two_face_simps[i], 2))
        for j in range(3):
            if not list(com[j]) in edges:
                edges.append(list(com[j]))
    
    # get the list of possible simplices
    possible_simps = [list(item) for item in list(combinations(poly.points(as_indices=True)[1:],4))]

    # check the list of allowed simplices
    all_simps = []
    for simp in possible_simps:
        possible_simp_edges = [list(item) for item in list(combinations(simp,2))]

        # identify missing edges
        missing_edges = []
        for edge in possible_simp_edges:
            if not edge in edges:
                missing_edges.append(edge)

        # if all edges exist then add to simplices list
        if missing_edges == []:
            all_simps.append(simp)

        # else check if the missing edges don't exist on the two faces and if so add simplices list
        else:
            test1 = True
            for edge in missing_edges:
                test2 = True
                for two_face in two_faces:
                    if edge[0] in two_face and edge[1] in two_face: 
                        test2 = False
                        break
                if not test2:
                    test1 = False
                    break
            if test1:
                all_simps.append(simp)

    # get the list of possible simplices for each facet
    facet_simps = [[] for facet in facets]
    for simp in all_simps:
        for facet in facets:
            if simp[0] in facet and simp[1] in facet and simp[2] in facet and simp[3] in facet:
                facet_simps[facets.index(facet)].append(simp)

    # get the combination of simplices list that give a valid triangulation of each facet
    possible_triang_simps = [[] for facet in facets]
    for i in range(len(facets)):
        # if facet is simplicial then add only simplex to list
        if len(facet_simps[i]) == 1:
            possible_triang_simps[i].append([[0]+facet_simps[i][0]])

        # else find a combination of simplices that gives a valid triangulation of the facet
        else:    
            # define the facet as a polytope
            F = Polytope(poly.points(facets[i]),labels=poly.points(facets[i],as_indices=True))
            pts = F.points(optimal=True)
            pts_ext = np.array([list(pt)+[1,] for pt in pts])

            # get the list of simplices for the facet with the approriate indices labelling 
            all_F_simps = [[F.points_to_indices(F.points(j))[0] for j in simp] for simp in facet_simps[i]]

            # try different combinations of simplices until a valid triangulation is found
            for j in range(2,len(all_F_simps)+1):
                inds_list = [list(item) for item in list(combinations(list(range(len(all_F_simps))),j))]

                for inds in inds_list:
                    F_simps = [all_F_simps[ind] for ind in inds]
                    valid = True

                    # 1. check the volumes of all the simplices
                    v = 0
                    for s in F_simps:
                        tmp_v = abs(int(round(np.linalg.det([pts_ext[i] for i in s]))))
                        if tmp_v == 0:
                            valid = False
                            break
                        v += tmp_v
                    if not valid:
                        continue

                    # 2. check if the volumes add up to the volume of the facet
                    facet_vol = int(round(ConvexHull(pts).volume*math.factorial(F._dim)))
                    if v != facet_vol:
                        valid = False
                        continue

                    # 3. check if the simplices have full-dimensional intersections
                    for k,s1 in enumerate(F_simps):
                        for s2 in F_simps[k+1:]:
                            inters = Cone(pts_ext[s1]).intersection(Cone(pts_ext[s2]))
                            if inters.is_solid():
                                valid = False
                                break
                        if not valid:
                            break
                
                    if valid:
                        possible_triang_simps[i].append([[0]+facet_simps[i][ind] for ind in inds])
                        
                        
    found = False
    count = 0
    while (not found) and (count<100):
        count += 1
        triang_simps = []
        for i in range(len(possible_triang_simps)):
            triang_simps = triang_simps + random.choice(possible_triang_simps[i])
        T = Triangulation(poly=poly, pts=poly.points(as_indices=True), simplices=triang_simps, check_input_simplices=False) 
        if T.is_valid():
            T = Triangulation(poly=poly, pts=poly.points(as_indices=True), simplices=triang_simps)
            found = True
            return T
    
    if not found:
        print("No valid triangulation found!")
        return None
