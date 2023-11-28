# author: elli heyes (elli.heyes@city.ac.uk)
# date last edited: 28/11/2023
import math
import copy
import random
import numpy as np
from itertools import repeat
from multiprocessing import Pool
from fitness_triang import fitness


class state:
    """"Triangulation state object:
         .poly is the ambient polytope
         .simps is the list of simplices 
         .fitness is the fitness of the triangulation
         .terminal is 1 if the triangulation is FRST and 0 otherwise"""
    def __init__(self, poly, simps):
        self.poly = poly
        self.simps = simps
        self.fitness = 0.
        self.terminal = 0
     
    def compute_fitness(self):
        self.fitness = fitness(self.poly, self.simps)
        if self.fitness == 0:
            self.terminal = 1
        else:
            self.terminal = 0


class population:
    """"Population object:
        .state_list is the list of triangulation states
        .num_sim_faces is the number of simplicial faces
        .non_sim_face_points is the list of points in the non-simplicial faces
        .max_simps is the maximum number of simplices
        .poly_dim is the dimension of the polytope
        .pop_size is the population size
        .max_fitness is the maximum fitness score
        .av_fitness is the average fitness score
        .num_term is the number of terminal states"""
    def __init__(self, state_list, num_sim_faces, non_sim_face_points):
        self.state_list = state_list
        self.num_sim_faces = num_sim_faces
        self.non_sim_face_points = non_sim_face_points
        self.max_simps = len(state_list[0].simps)
        self.poly_dim = state_list[0].poly.dim()
        self.pop_size = len(state_list)
        self.max_fitness = 0.
        self.av_fitness = 0.
        self.num_term = 0
        
    def compute_max_fitness(self):
        max_fit = self.state_list[0].fitness
        for i in range(1,self.pop_size):
            if self.state_list[i].fitness > max_fit:
                max_fit = self.state_list[i].fitness
        self.max_fitness = max_fit 
    
    def compute_av_fitness(self):
        sum_fit = 0
        for i in range(self.pop_size):
            sum_fit += self.state_list[i].fitness
        self.av_fitness = sum_fit / self.pop_size
        
    def compute_num_term(self):
        total_term = 0
        for i in range(self.pop_size):
            if self.state_list[i].terminal:
                total_term += 1
        self.num_term = total_term
        
    def mutate_pop(self, mut_rate):
        # determine the number of mutations to perform
        num_mut = self.pop_size * mut_rate
    
        # run over mutations
        for i in range(int(num_mut)):
            # determine a random state in the population
            state_pos = random.choice(range(self.pop_size))
            
            # determine a random simplex in the state
            sim_pos = random.choice(range(self.num_sim_faces,self.max_simps))
            non_sim_face_index = math.floor((sim_pos-self.num_sim_faces)/math.ceil(len(non_sim_face_points[i])/poly.dim()))
            
            # determine a random point in the simplex
            point_pos = random.choice(range(1,self.poly_dim))
        
            # mutate by randomly choosing a point in the non-simplicial face
            self.state_list[state_pos].simps[sim_pos][point_pos] = random.choice(self.non_sim_face_points[non_sim_face_index])
            self.state_list[state_pos].simps[sim_pos] = sorted(self.state_list[state_pos].simps[sim_pos])

            # update fitness
            self.state_list[state_pos].compute_fitness()
        
        # update average and maximum fitness and number of terminal states
        self.compute_max_fitness()
        self.compute_av_fitness()
        self.compute_num_term()
        

def crossover(state1, state2, num_sim_faces):
    """"Cross two states."""
    # determine a random simplex in the triangulation
    sim_pos = random.choice(range(num_sim_faces,len(state1.simps)))
            
    # determine a random point in the simplex
    point_pos = random.choice(range(1,len(state1.simps[sim_pos])))
    
    # swap the relevant parts of the bit lists
    new_simps1 = copy.deepcopy(state1.simps)
    new_simps2 = copy.deepcopy(state2.simps)
    for i in range(sim_pos,len(new_simps1)):
        if i == sim_pos:
            start = point_pos
        else:
            start = 0
        for j in range(start,len(new_simps1[i])):
            new_simps1[i][j] = state2.simps[i][j]
            new_simps2[i][j] = state1.simps[i][j]
    
    # define the new states
    new_state1 = state(poly=state1.poly, simps=new_simps1)
    new_state2 = state(poly=state2.poly, simps=new_simps2)
    
    # update the fitness
    new_state1.compute_fitness()
    new_state2.compute_fitness()

    return new_state1, new_state2
         
    
def random_state(poly, sim_face_simps, non_sim_face_points):
    """"Generate a random state."""
    # start with the simplicial face simplices
    simps = sim_face_simps
    
    # add random simplices for the non_simplicial faces
    for i in range(len(non_sim_face_points)):
        # set the number of simplices in the triangulation of the face to be equal to the number of points divided by the poly dimension
        num_simp = math.ceil(len(non_sim_face_points[i])/poly.dim())
        for j in range(num_simp):
            # randomly select points from the face
            simp = sorted([0]+random.sample(non_sim_face_points[i],poly.dim()))
            simps.append(simp)
        
    S = state(poly=poly, simps=simps)
    S.compute_fitness()
    
    return S

        
def random_pop(poly, pop_size):
    """"Generate a random population."""
    # get the simplicial and non-simplicial faces
    faces = poly.facets()
    sim_face_simps = []
    non_sim_face_points = []
    for i in range(len(faces)):
        num_points = len(faces[i].boundary_points())
        if num_points == poly.dim():
            simp = [0]
            for j in range(num_points):
                simp.append(list(poly.points_not_interior_to_facets(as_indices=True)).index(poly.points_to_indices(faces[i].boundary_points()[j])))
            sim_face_simps.append(simp)
        else:
            points = []
            for j in range(num_points):
                points.append(list(poly.points_not_interior_to_facets(as_indices=True)).index(poly.points_to_indices(faces[i].boundary_points()[j])))
            non_sim_face_points.append(points)
    num_sim_faces = len(sim_face_simps)
    
    state_list = []
    for i in range(pop_size):
        print(i)
        state_list.append(random_state(poly, sim_face_simps, non_sim_face_points))
    
    pop = population(state_list, num_sim_faces, non_sim_face_points)
    pop.compute_max_fitness()
    pop.compute_av_fitness()
    pop.compute_num_term()
    
    return pop


def sort_pop(pop):
    """"Sort a population based on the fitness scores."""
    fitness_scores = [pop.state_list[i].fitness for i in range(pop.size)]
    sorted_indices = np.argsort(np.array(fitness_scores))
    sorted_state_list = [pop.state_list[i] for i in sorted_indices]
    return population(sorted_state_list, pop.num_sim_faces, pop.non_sim_face_points)


def select_and_cross(pop, p):
    """"Select two pairs of individuals from the population and cross them."""
    indices = random.choices(range(pop.pop_size), p, k=2)
    state1, state2 = crossover(pop.state_list[indices[0]], pop.state_list[indices[1]], pop.num_sim_faces)
    return [state1, state2]
    

def next_pop(pop, mut_rate=0.01):
    """"Update a population by performing selection, crossover and mutation."""
    df = pop.max_fitness - pop.av_fitness
    if df <= 0:
        p = [1/pop.pop_size for i in range(pop.pop_size)]
    else:
        p = [((3-1)*(pop.state_list[i].fitness-pop.av_fitness)+df)/df/pop.pop_size for i in range(pop.pop_size)]

    state_list = []
    for i in range(int(pop.pop_size/2)):
        state_list = state_list + select_and_cross(pop, p)
    new_pop = population(state_list, pop.num_sim_faces, pop.non_sim_face_points)
    
    new_pop.mutate_pop(mut_rate=mut_rate)

    return new_pop
            
    
def term_states(pop):
    """"Select terminal states from a population."""
    states = []
    poly = pop.state_list[0].poly      
    triang_pts = poly.points_not_interior_to_facets() 
    for i in range(pop.pop_size):
        if pop.state_list[i].terminal == 1:   
            simps = []
            for simp in pop.state_list[i].simps:
                if not simp in simps:
                    simps.append(simp)
            T = Triangulation(triang_pts, poly=poly, simplices=simps, check_input_simplices=False)
            states.append(T)
    return states


def remove_redundancy(term_states):
    """"Remove redundancy in a list of terminal states."""
    reduced_states = []
    for i in range(len(term_states)):
        equiv = 0
        for j in range(len(reduced_states)):
            if term_states[i].is_equivalent(reduced_states[j]):
                equiv = 1
                break
        if not equiv:
            reduced_states.append(term_states[i])
    return reduced_states
    

def evol_pop(pop, num_gen, mut_rate, monitor=True):
    """"Evolve a population over generations and extract terminal states."""    

    term_states_list = remove_redundancy(term_states(pop))

    if monitor:
        print("Total # Terminal States    Average Fitness    Maximum Fitness")
    for i in range(num_gen):
        pop = next_pop(pop, mut_rate=mut_rate)
        
        term_states_list = term_states_list + term_states(pop)
        
        if monitor:
            print("    "+str(len(term_states_list))+"                        "+str(round(pop.av_fitness,2))+
                  "                  "+str(round(pop.max_fitness,2)))
    
    term_states_list = remove_redundancy(term_states_list)
    print("Total # of reduced terminal states: "+str(len(term_states_list)))

    return term_states_list
