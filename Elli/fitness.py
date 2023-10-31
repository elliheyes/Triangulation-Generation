from itertools import combinations

def fitness_func(non_sim_faces_points, non_sim_faces_sub_sim_points, non_sim_faces_sub_sim_Hrep, 
                 adj_faces_inds, adj_faces_points, sim_face_gale_cones, non_sim_faces_sub_sim_gale_cones, 
                 POLYDIM, bits):
    """"Compute the fitness of a triangulation."""
    score = 0
    
    union_weight = 1
    gap_weight = 1
    overlap_weight = 1
    consistency_weight = 1
    regularity_weight = 1
    
    # add a penalty if the union of the points sub simplices does not equal the face 
    union_penalty = 0
    for i in range(len(bits)):
        for point in non_sim_faces_points[i]:
            exist = 0
            for j in range(len(bits[i])):
                if bits[i][j] == 1 and point in non_sim_faces_sub_sim_points[i][j]:
                    exist = 1
            if not exist:
                union_penalty += 1
    score -= union_weight * union_penalty
    
    # add a penalty if there are gaps between the sub simplices 
    gap_penalty = 0
    for i in range(len(bits)):
        for j in range(len(bits[i])):
            if bits[i][j] == 1:
                adj = 0
                for k in range(len(bits[i])):
                    if bits[i][k] == 1 and k != j:
                        for l in range(POLYDIM):
                            for m in range(POLYDIM):
                                test = 1
                                for n in range(POLYDIM+1):
                                    if non_sim_faces_sub_sim_Hrep[i][j][1+l][n] != -non_sim_faces_sub_sim_Hrep[i][k][1+m][n]:
                                        test = 0
                                if test:
                                    adj = 1
                if not adj:
                    gap_penalty += 1
    score -= gap_weight * gap_penalty
                        
    # add a penalty for overlap of sub simplices
    overlap_penalty = 0
    for i in range(len(bits)):
        # get a list of the indices of the included sub simplices
        sub_sim_inds = [j for j in range(len(bits[i])) if bits[i][j]==1]
        
        # find the list of pairs of included sub simplices
        sub_sim_pairs = list(combinations(sub_sim_inds, 2))
        
        for j in range(len(sub_sim_pairs)):
            # find the overlap of points between the pair of included sub simplices
            overlap_points = []
            for point in non_sim_faces_sub_sim_points[i][sub_sim_pairs[j][0]]:
                if point in non_sim_faces_sub_sim_points[i][sub_sim_pairs[j][1]]:
                    overlap_points.append(point)
            
            # if the number of overlap points is more than 1 check that the pair sub simplices 
            # share a bounding hyperplane and if so check that the overlap points lie on this plane
            if len(overlap_points) > 1:
                overlap_is_ok = 0
                for k in range(len(non_sim_faces_sub_sim_Hrep[i][sub_sim_pairs[j][0]])-1):
                    for l in range(len(non_sim_faces_sub_sim_Hrep[i][sub_sim_pairs[j][1]])-1):
                        pos_equal = 1
                        for m in range(POLYDIM+1):
                            if non_sim_faces_sub_sim_Hrep[i][sub_sim_pairs[j][0]][1+k][m] != non_sim_faces_sub_sim_Hrep[i][sub_sim_pairs[j][1]][1+l][m]:
                                pos_equal = 0
                        neg_equal = 1
                        for m in range(POLYDIM):
                            if non_sim_faces_sub_sim_Hrep[i][sub_sim_pairs[j][0]][1+k][m] != -non_sim_faces_sub_sim_Hrep[i][sub_sim_pairs[j][1]][1+l][m]:
                                neg_equal = 0
                        if pos_equal or neg_equal:
                            H = non_sim_faces_sub_sim_Hrep[i][sub_sim_pairs[j][0]][1+k]
                            is_ok = 1
                            for point in overlap_points:
                                eval_sum = H[0]
                                for m in range(POLYDIM):
                                    eval_sum += H[1+m] * point[m]
                                if eval_sum != 0:
                                    is_ok = 0
                            if is_ok:
                                overlap_is_ok = 1
                if not overlap_is_ok:
                    overlap_penalty += 1
    score -= overlap_weight * overlap_penalty
                
    # add a penalty if the triangulations are not consistent between bounding non-simplicial faces
    consistency_penalty = 0
    for i in range(len(adj_faces_inds)):
        face1 = []
        for j in range(len(bits[adj_faces_inds[i][0]])):
            if bits[adj_faces_inds[i][0]][j] == 1:
                point_list = []
                for point in non_sim_faces_sub_sim_points[adj_faces_inds[i][0]][j]:
                    if point in adj_faces_points[i]:
                        point_list.append(point)
                if len(point_list) == POLYDIM-1:
                    face1.append(sorted(point_list))
        face2 = []
        for j in range(len(bits[adj_faces_inds[i][1]])):
            if bits[adj_faces_inds[i][1]][j] == 1:
                point_list = []
                for point in non_sim_faces_sub_sim_points[adj_faces_inds[i][1]][j]:
                    if point in adj_faces_points[i]:
                        point_list.append(point)
                if len(point_list) == POLYDIM-1:
                    face2.append(sorted(point_list))
        if sorted(face1) != sorted(face2):
            consistency_penalty += 1
    score -= consistency_weight * consistency_penalty
    
    # add a penalty for the regularity condition
    
    # find the list of gale cones for the triangulation
    S_cones = sim_face_gale_cones
    for i in range(len(bits)):
        for j in range(len(bits[i])):
            if bits[i][j] == 1:
                S_cones.append(non_sim_faces_sub_sim_gale_cones[i][j])
    
    # find the intersection of the gale cones
    intersection_cone = S_cones[0]
    for i in range(1,len(S_cones)):
        intersection_cone = intersection_cone.intersection(S_cones[i])
    
    # check if the intersection cone is empty
    regularity_penalty = 0
    if intersection_cone.is_empty():
        regularity_penalty = 1
    else:
        regularity_penalty = 0
    
    score -= regularity_weight * regularity_penalty
    
    return score


