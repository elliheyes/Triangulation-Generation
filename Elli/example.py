from setup import sim_faces, non_sim_faces, adj_faces, non_sim_sub_sim, gale_trans, sim_gale_cones, non_sim_sub_sim_gale_cones
from GA import random_pop, evol_pop
from sage.geometry.lattice_polytope import LatticePolytope

# define the polytope
POLYDIM = 3
vertices = [[1,0,0],[0,1,0],[0,0,1],[-1,-1,-1],[-1,-1,2]]
poly = LatticePolytope(vertices)

# find the simplicial faces
sim_faces_points = sim_faces(vertices)

# find the non-simplicial faces
non_sim_faces_inds, non_sim_faces_points, non_sim_faces_Hrep = non_sim_faces(vertices)
                                                               
# find all of the adjacent non_simplicial faces
adj_faces_inds, adj_faces_points = adj_faces(non_sim_faces_inds, non_sim_faces_points, non_sim_faces_Hrep, POLYDIM)
            
# find all the allowed codim=1 sub simplices of the non-simplicial faces
non_sim_faces_sub_sim_points, non_sim_faces_sub_sim_Hrep, num_non_sim_faces_sub_sim = non_sim_sub_sim(non_sim_faces_points, POLYDIM)

# compute the Gale transform of the point configuration
points, gale = gale_trans(poly, POLYDIM)

# compute the gale transform cones of the simplicial faces 
sim_faces_gale_cones = sim_gale_cones(poly, non_sim_faces_inds, points, gale, POLYDIM)

# compute the gale transform cones of the non-simplicial faces sub simplices
non_sim_faces_sub_sim_gale_cones = non_sim_sub_sim_gale_cones(non_sim_faces_sub_sim_points, points, gale, POLYDIM)
   
# define parameters
pop_size = 200
num_gen = 10
num_run = 10
mut_rate = 0.01

# generate random initial population  
initial_pop = random_pop(non_sim_faces_points, non_sim_faces_sub_sim_points, 
               non_sim_faces_sub_sim_Hrep, adj_faces_inds, adj_faces_points, 
               sim_faces_gale_cones, non_sim_faces_sub_sim_gale_cones, num_non_sim_faces_sub_sim, 
               POLYDIM, pop_size)

# evolve random population over generations and extract terminal states
terminal_states = evol_pop(initial_pop, num_gen, mut_rate)


