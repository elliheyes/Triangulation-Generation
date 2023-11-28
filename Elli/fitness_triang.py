# author: elli heyes (elli.heyes@city.ac.uk)
# date last edited: 28/11/2023
import math
import numpy as np
from copy import deepcopy
from flint import fmpz_mat
from cytools.cone import Cone
from cytools.utils import gcd_list
from scipy.spatial import ConvexHull

def secondary_cone(pts_ext, simps, dim):
    simps = [set(s) for s in simps]
    m = np.zeros((dim+1, dim+2), dtype=int)
    null_vecs = set()
    for i,s1 in enumerate(simps):
        for s2 in simps[i+1:]:
            # eunsre that the simplices have a large enough intersection
            comm_pts = s1 & s2
            if len(comm_pts) != dim:
                continue
 
            diff_pts = list(s1 ^ s2)
            comm_pts = list(comm_pts)
            for j,pt in enumerate(diff_pts):    m[:,j] = pts_ext[pt]
            for j,pt in enumerate(comm_pts):    m[:,j+2] = pts_ext[pt]

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
            full_v = np.zeros(len(pts_ext), dtype=int)

            for i,pt in enumerate(diff_pts):    full_v[pt] = v[i]
            for i,pt in enumerate(comm_pts):    full_v[pt] = v[i+2]

            full_v = tuple(full_v)
            if full_v not in null_vecs:
                null_vecs.add(full_v)

    sec_cone = Cone(hyperplanes=list(null_vecs),check=False)
    return sec_cone

def fitness(poly, simps):
    
    # initialise the terms 
    term1 = term2 = term3 = term4 = term5 = 0
    
    # define the triangulation points
    triang_pts = poly.points_not_interior_to_facets()
    tmp_triang_pts = {tuple(pt) for pt in np.array(triang_pts, dtype=int)}
   
    # append 1 to each point
    pts = deepcopy(triang_pts)
    pts_ext = np.array([list(pt)+[1,] for pt in pts])
        
    # check the volumes of all the simplices
    v = 0
    for s in simps:
        tmp_v = abs(int(round(np.linalg.det([pts_ext[i] for i in s]))))                
        if tmp_v == 0:
            term1 -= 10
        v += tmp_v
           
    # check if the volumes of the simplices add up to the volume of the polytope
    dim = np.linalg.matrix_rank([pt+(1,) for pt in tmp_triang_pts])-1
    poly_vol = int(round(ConvexHull(pts).volume*math.factorial(dim)))
    if v != poly_vol:
        term2 = -10
        
    # check if simplices have full-dimensional intersections
    for i,s1 in enumerate(simps):
        for s2 in simps[i+1:]:
            inters = Cone(pts_ext[s1]).intersection(Cone(pts_ext[s2]))
            if inters.is_solid():
                term3 -= 10
    
    # if valid triangulation then check final conditions 
    if term1 + term2 + term3 == 0:
        # check the fineness condition 
        if (len(set.union(*[set(s) for s in simps])) != len(triang_pts)):
            term4 = -5    
    
        # check if the triangulation is regular
        C = secondary_cone(pts_ext, simps, dim)
        if not C.is_solid():
            term5 = -5
 
    return term1 + term2 + term3 + term4 + term5


