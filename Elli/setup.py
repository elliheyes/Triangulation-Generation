from itertools import combinations
from sage.geometry.lattice_polytope import LatticePolytope
from sage.geometry.polyhedron.constructor import Polyhedron
from sage.geometry.triangulation.point_configuration import PointConfiguration
from sage.geometry.cone import Cone

def sim_faces(points):
    """"Find the simplicial faces."""
    polytope = LatticePolytope(points)
    
    sim_faces_points = []
    for i in range(len(polytope.faces(codim=1))):
        if len(polytope.faces(codim=1)[i].boundary_points()) == polytope.dim():
            points = []
            for j in range(len(polytope.faces(codim=1)[i].boundary_points())):
                points.append(polytope.faces(codim=1)[i].points()[j])
            sim_faces_points.append(points)
    
    return sim_faces_points

def non_sim_faces(points):
    """"Find the non-simplicial faces."""
    polytope = LatticePolytope(points)
    
    non_sim_faces_inds = []
    non_sim_faces_points = []
    non_sim_faces_Hrep = []
    for i in range(len(polytope.faces(codim=1))):
        if len(polytope.faces(codim=1)[i].boundary_points()) > polytope.dim():
            non_sim_faces_inds.append(i)
            
            points = []
            for j in range(len(polytope.faces(codim=1)[i].boundary_points())):
                points.append(polytope.faces(codim=1)[i].points()[j])
            non_sim_faces_points.append(points)
            
            polyhedron = Polyhedron(points)
            non_sim_faces_Hrep.append(polyhedron.Hrepresentation())
    
    return non_sim_faces_inds, non_sim_faces_points, non_sim_faces_Hrep

def adj_faces(non_sim_faces_inds, non_sim_faces_points, non_sim_faces_Hrep, POLYDIM):
    """"Find adjacent faces in a list of faces."""
    face_pairs = list(combinations(range(len(non_sim_faces_inds)), 2))
    
    adj_faces_inds = []
    adj_faces_points = []
    for i in range(len(face_pairs)):
        for j in range(len(non_sim_faces_Hrep[face_pairs[i][0]])-1):
            for k in range(len(non_sim_faces_Hrep[face_pairs[i][1]])-1):
                pos_equal = 1
                for l in range(POLYDIM+1):
                    if non_sim_faces_Hrep[face_pairs[i][0]][1+j][l] != non_sim_faces_Hrep[face_pairs[i][1]][1+k][l]:
                        pos_equal = 0
                neg_equal = 1
                for l in range(POLYDIM+1):
                    if non_sim_faces_Hrep[face_pairs[i][0]][1+j][l] != -non_sim_faces_Hrep[face_pairs[i][1]][1+k][l]:
                        neg_equal = 0
                if pos_equal or neg_equal:
                    adj_faces_inds.append(face_pairs[i])
                    common_points = []
                    for point in non_sim_faces_points[face_pairs[i][0]]:
                        eval_sum = non_sim_faces_Hrep[face_pairs[i][0]][1+j][0]
                        for l in range(POLYDIM):
                            eval_sum += non_sim_faces_Hrep[face_pairs[i][0]][1+j][1+l]*point[l]
                        if eval_sum == 0:
                            common_points.append(point)
                    adj_faces_points.append(common_points)
                    
    return adj_faces_inds, adj_faces_points

def non_sim_sub_sim(non_sim_faces_points, POLYDIM):
    """"Find all the allowed codim=1 sub simplices of the non-simplicial faces."""
    non_sim_faces_sub_sim_points = []
    non_sim_faces_sub_sim_Hrep = []
    num_non_sim_faces_sub_sim = []
    for i in range(len(non_sim_faces_points)):
        all_points = list(combinations(non_sim_faces_points[i], POLYDIM))
        allowed_points = []
        allowed_Hrep = []
        for j in range(len(all_points)):
            polytope = LatticePolytope(all_points[j])
            polyhedron = Polyhedron(all_points[j])
            if polytope.dim() == POLYDIM-1 and len(polytope.interior_points()) == 0:
                allowed_points.append(all_points[j])
                allowed_Hrep.append(polyhedron.Hrepresentation())
        non_sim_faces_sub_sim_points.append(allowed_points)
        non_sim_faces_sub_sim_Hrep.append(allowed_Hrep)
        num_non_sim_faces_sub_sim.append(len(allowed_Hrep))
        
    return non_sim_faces_sub_sim_points, non_sim_faces_sub_sim_Hrep, num_non_sim_faces_sub_sim

def gale_trans(poly, POLYDIM):
    """"Find the list of points and the corresponding list of gale transform points."""
    # define the list of boundary points of the polytope plus the origin
    points = [[0 for i in range(POLYDIM)]]
    for i in range(len(poly.boundary_points())):
        point = []
        for j in range(POLYDIM):
            point.append(poly.boundary_points()[i][j])
        points.append(point)
    
    # define the gale transform of the point configuration
    pc = PointConfiguration(points)
    GT = pc.Gale_transform().transpose()
    
    # format the gale transform into a list
    gale = []
    for i in range(len(points)):
        point = []
        for j in range(len(GT[0])):
            point.append(GT[i][j])
        gale.append(point)
    
    return points, gale

def sim_gale_cones(poly, non_sim_faces_inds, points, gale, POLYDIM):
    """"Find the list of gale transform cones of the simplicial faces."""
    sim_face_gale_cones = []
    for i in range(len(poly.faces(codim=1))):
        if not i in non_sim_faces_inds:
            face_point_inds = [0]
            for j in range(POLYDIM):
                point = []
                for k in range(POLYDIM):
                    point.append(poly.faces(codim=1)[i].points()[j][k])
                face_point_inds.append(points.index(point))
            face_gale_points = []
            for j in range(len(gale)):
                if not j in face_point_inds:
                    face_gale_points.append(gale[j])
            c = Cone(face_gale_points)
            sim_face_gale_cones.append(c)
    return sim_face_gale_cones

def non_sim_sub_sim_gale_cones(non_sim_faces_sub_sim_points, points, gale, POLYDIM):
    """"Find the list of gale transform cones of the sub simplices of the
    non-simplicial faces."""
    
    non_sim_faces_sub_sim_gale_cones = []
    for i in range(len(non_sim_faces_sub_sim_points)):
        non_sim_face_sub_sim_gale_cones = []
        for j in range(len(non_sim_faces_sub_sim_points[i])):
            S_point_inds = [0]
            for k in range(POLYDIM):
                point = []
                for l in range(POLYDIM):
                    point.append(non_sim_faces_sub_sim_points[i][j][k][l])
                S_point_inds.append(points.index(point))
            S_gale_points = []
            for k in range(len(gale)):
                if not k in S_point_inds:
                    S_gale_points.append(gale[k])
            c = Cone(S_gale_points)
            non_sim_face_sub_sim_gale_cones.append(c)
        non_sim_faces_sub_sim_gale_cones.append(non_sim_face_sub_sim_gale_cones)
    
    return non_sim_faces_sub_sim_gale_cones
    
