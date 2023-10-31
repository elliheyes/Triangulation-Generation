import copy
import random
import numpy as np
from itertools import repeat
from multiprocessing import Pool
from fitness import fitness_func


class triang:
    """"Triangulation class:
         .non_sim_faces_points is the list of points in the non-simplicial faces
         .non_sim_faces_sub_sim_points is the list of points in the sub simplices 
         .non_sim_faces_sub_sim_Hrep is the list of hyperplanes of the sub simplices
         .adj_faces_inds is the list of adjacent non-simplicial faces
         .adj_faces_points is the list of points in the adjacent non-simplicial faces
         .bits is the list of bitlists for the non-simplicial faces
         .fitness is the fitness of the triangulation
         .terminal is 1 if the triangulation is FRST and 0 otherwise"""
    def __init__(self, non_sim_faces_points, non_sim_faces_sub_sim_points, 
                 non_sim_faces_sub_sim_Hrep, adj_faces_inds, adj_faces_points, 
                 sim_face_gale_cones, non_sim_faces_sub_sim_gale_cones, POLYDIM, bits):
        self.non_sim_faces_points = non_sim_faces_points
        self.non_sim_faces_sub_sim_points = non_sim_faces_sub_sim_points
        self.non_sim_faces_sub_sim_Hrep = non_sim_faces_sub_sim_Hrep
        self.adj_faces_inds = adj_faces_inds
        self.adj_faces_points = adj_faces_points
        self.bits = bits
        self.sim_face_gale_cones = sim_face_gale_cones
        self.non_sim_faces_sub_sim_gale_cones = non_sim_faces_sub_sim_gale_cones
        self.polydim = POLYDIM
        self.fitness = 0.
        self.terminal = 0
    
    # compute the fitness 
    def compute_fitness(self):
        self.fitness = fitness_func(self.non_sim_faces_points, 
                                    self.non_sim_faces_sub_sim_points, 
                                    self.non_sim_faces_sub_sim_Hrep, 
                                    self.adj_faces_inds, 
                                    self.adj_faces_points,
                                    self.sim_face_gale_cones,
                                    self.non_sim_faces_sub_sim_gale_cones,
                                    self.polydim,
                                    self.bits)
        if self.fitness == 0:
            self.terminal = 1
        else:
            self.terminal = 0


class population:
    """"Population class:
        .state_list is the list of triangulation states
        .pop_size is the population size
        .num_bits is the total number of bits in the population
        .max_fitness is the maximum fitness score
        .av_fitness is the average fitness score
        .num_term is the number of terminal states"""
    def __init__(self, non_sim_faces_points, non_sim_faces_sub_sim_points, 
                 non_sim_faces_sub_sim_Hrep, adj_faces_inds, adj_faces_points, 
                 sim_face_gale_cones, non_sim_faces_sub_sim_gale_cones, state_list):
        self.non_sim_faces_points = non_sim_faces_points
        self.non_sim_faces_sub_sim_points = non_sim_faces_sub_sim_points
        self.non_sim_faces_sub_sim_Hrep = non_sim_faces_sub_sim_Hrep
        self.adj_faces_inds = adj_faces_inds
        self.adj_faces_points = adj_faces_points
        self.sim_face_gale_cones = sim_face_gale_cones
        self.non_sim_faces_sub_sim_gale_cones = non_sim_faces_sub_sim_gale_cones
        self.polydim = state_list[0].polydim
        self.state_list = state_list
        self.pop_size = len(state_list)
        self.num_bits = len(np.array(state_list[0].bits).flatten()) * len(state_list)
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
        num_mut = (self.num_bits * mut_rate).round()
    
        # run over mutations
        for i in range(num_mut):
            # determine a random state in the population
            state_pos = random.choice(range(self.pop_size))
            
            # determine a random face in the state
            face_pos = random.choice(range(len(self.state_list[state_pos].bits)))
            
            # determine a random sub simplex in the face
            sub_sim_pos = random.choice(range(len(self.state_list[state_pos].bits[face_pos])))
        
            # flip bit
            self.state_list[state_pos].bits[face_pos][sub_sim_pos] = (self.state_list[state_pos].bits[face_pos][sub_sim_pos]+1)%2
            
            # update fitness
            self.state_list[state_pos].compute_fitness()
        
        # update average and maximum fitness and number of terminal states
        self.compute_max_fitness()
        self.compute_av_fitness()
        self.compute_num_term()
        

def cross_bits(bits1, bits2):
    """"Cross two bit lists."""
    # choose a random non-simplicial face 
    face_pos = random.choice(range(len(bits1)))
    
    # choose a random sub simplex in the chosen face
    sub_sim_pos = random.choice(range(len(bits1[face_pos])))
    
    # swap the relevant parts of the bit lists
    new_bits1 = copy.deepcopy(bits1)
    new_bits2 = copy.deepcopy(bits2)
    for i in range(face_pos,len(bits1)):
        if i == face_pos:
            start = sub_sim_pos
        end = len(bits1[i])
        for j in range(start,end):
            new_bits1[i][j] = bits2[i][j]
            new_bits2[i][j] = bits1[i][j]

    return new_bits1, new_bits2
            

def random_triang(non_sim_faces_points, 
                  non_sim_faces_sub_sim_points, 
                  non_sim_faces_sub_sim_Hrep, 
                  adj_faces_inds, 
                  adj_faces_points, 
                  sim_face_gale_cones,
                  non_sim_faces_sub_sim_gale_cones,
                  POLYDIM,
                  num_non_sim_faces_sub_sim):
    """"Generate a random triangulation state."""
    bits = []
    for i in range(len(num_non_sim_faces_sub_sim)):
        face_bits = []
        for j in range(num_non_sim_faces_sub_sim[i]):
            face_bits.append(random.choice(range(2)))
        bits.append(face_bits)
    T = triang(non_sim_faces_points, 
               non_sim_faces_sub_sim_points, 
               non_sim_faces_sub_sim_Hrep, 
               adj_faces_inds, 
               adj_faces_points, 
               sim_face_gale_cones,
               non_sim_faces_sub_sim_gale_cones,
               POLYDIM,
               bits)
    T.compute_fitness()
    return T
          
# TODO: add equivalence check as well as equality
def equiv_triang(triang1, triang2):
    """"Determine whether two triangualtions are equivalent."""
    bits_flat1 = np.array(triang1.bits).flatten()
    bits_flat2 = np.array(triang2.bits).flatten()
    
    if (bits_flat1 == bits_flat2).all():
        return 1
    else:
        return 0
    
def random_pop(non_sim_faces_points, non_sim_faces_sub_sim_points, 
               non_sim_faces_sub_sim_Hrep, adj_faces_inds, adj_faces_points, 
               sim_face_gale_cones, non_sim_faces_sub_sim_gale_cones, 
               num_non_sim_faces_sub_sim, POLYDIM, pop_size):
    """"Generate a random population."""
    state_list = []
    for i in range(pop_size):
        state_list.append(random_triang(non_sim_faces_points, 
                                        non_sim_faces_sub_sim_points, 
                                        non_sim_faces_sub_sim_Hrep, 
                                        adj_faces_inds, 
                                        adj_faces_points, 
                                        sim_face_gale_cones,
                                        non_sim_faces_sub_sim_gale_cones,
                                        POLYDIM,
                                        num_non_sim_faces_sub_sim))
    
    pop = population(non_sim_faces_points, non_sim_faces_sub_sim_points, 
                 non_sim_faces_sub_sim_Hrep, adj_faces_inds, adj_faces_points, 
                 sim_face_gale_cones, non_sim_faces_sub_sim_gale_cones, state_list)
    
    pop.compute_max_fitness()
    pop.compute_av_fitness()
    pop.compute_num_term()
    
    return pop


def sort_pop(pop):
    """"Sort a population based on the fitness scores."""
    fitnesses = [pop.state_list[i].fitness for i in range(pop.size)]
    sorted_indices = np.argsort(np.array(fitnesses))
    sorted_state_list = [pop.state_list[i] for i in sorted_indices]
    return population(pop.non_sim_faces_points, pop.non_sim_faces_sub_sim_points, 
                 pop.non_sim_faces_sub_sim_Hrep, pop.adj_faces_inds, pop.adj_faces_points, 
                 pop.sim_face_gale_cones, pop.non_sim_faces_sub_sim_gale_cones, sorted_state_list)

def select_and_breed(pop, p):
    """"Select two pairs of individuals from the population and breed them."""
    # select a pair of fit individuals from the population
    indices = random.choices(range(pop.pop_size), p, k=2)
    
    # cross the bitlists
    bits1, bits2 = cross_bits(pop.state_list[indices[0]].bits, pop.state_list[indices[1]].bits)
        
    # define the new triangulations
    state1 = triang(pop.non_sim_faces_points, pop.non_sim_faces_sub_sim_points, 
                 pop.non_sim_faces_sub_sim_Hrep, pop.adj_faces_inds, pop.adj_faces_points, 
                 pop.sim_face_gale_cones, pop.non_sim_faces_sub_sim_gale_cones, pop.polydim, bits1)
    state2 = triang(pop.non_sim_faces_points, pop.non_sim_faces_sub_sim_points, 
                 pop.non_sim_faces_sub_sim_Hrep, pop.adj_faces_inds, pop.adj_faces_points, 
                 pop.sim_face_gale_cones, pop.non_sim_faces_sub_sim_gale_cones, pop.polydim, bits2)
    
    # compute the fitnesses
    state1.compute_fitness()
    state2.compute_fitness()
    
    return state1, state2
    

def next_pop(pop, mut_rate=0.01):
    """"Update a population by performing selection, crossover and mutation."""
    # determine the roulette selection probabilities
    df = pop.max_fitness - pop.av_fitness
    if df <= 0:
        p = [1/pop.pop_size for i in range(pop.pop_size)]
    else:
        p = [((3-1)*(pop.state_list[i].fitness-pop.av_fitness)+df)/df/pop.pop_size for i in range(pop.pop_size)]

    # choose individuals for breeding and perform crossover
    with Pool(processes=6) as pool:
        states = pool.starmap(select_and_breed, repeat((pop,p),int(pop.pop_size/2)))
    states = list(np.array(states).flatten())
    
    # define the new population
    new_pop = population(pop.non_sim_faces_points, pop.non_sim_faces_sub_sim_points, 
                 pop.non_sim_faces_sub_sim_Hrep, pop.adj_faces_inds, pop.adj_faces_points, 
                 pop.sim_face_gale_cones, pop.non_sim_faces_sub_sim_gale_cones, states)
    
    # mutate the new population
    new_pop.mutate_pop(mut_rate=mut_rate)

    return new_pop
            
    
def term_states(pop):
    """"Select terminal states from a population."""
    states = []
    for i in range(pop.pop_size):
        if pop.state_list[i].terminal == 1:
            states.append(pop.state_list[i])
    return states

# redundancy whilst equiv_triang is empty
def remove_redundancy(term_states):
    """"Remove redundancy in a list of terminal states."""
    reduced_states = []
    for i in range(len(term_states)):
        equiv = 0
        for j in range(len(reduced_states)):
            if equiv_triang(term_states[i],reduced_states[j]):
                equiv = 1
                break
        if not equiv:
            reduced_states.append(term_states[i])
    return reduced_states
    

def evol_pop(pop, num_gen, mut_rate, monitor=True):
    """"Evolve a population over generations and extract terminal states."""    
    # define initisal terminal states list
    term_states_list = remove_redundancy(term_states(pop))

    # loop over generations
    if monitor:
        print("Total # Terminal States    Average Fitness    Maximum Fitness")
    for i in range(num_gen):
        pop = next_pop(pop, mut_rate=mut_rate)
        
        term_states_list = term_states_list + remove_redundancy(term_states(pop))
        term_states_list = remove_redundancy(term_states_list)
        
        if monitor:
            print("    "+str(len(term_states_list))+"                        "+str(pop.av_fitness)+
                  "                    "+str(pop.max_fitness))

    return term_states_list


def search(non_sim_faces_points, non_sim_faces_sub_sim_points, 
           non_sim_faces_sub_sim_Hrep, adj_faces_inds, adj_faces_points,
           sim_face_gale_cones, non_sim_faces_sub_sim_gale_cones, 
           POLYDIM, num_non_sim_faces_sub_sim, num_run, pop_size, num_gen, mut_rate):
    """"Evolve several random populations over generations and extract terminal states."""    
    
    # loop over runs
    terminal_states = []
    print("Total # Terminal States")
    for i in range(num_run):
        # generatate random population
        initial_pop = random_pop(non_sim_faces_points, 
                                 non_sim_faces_sub_sim_points, 
                                 non_sim_faces_sub_sim_Hrep, 
                                 adj_faces_inds, adj_faces_points, 
                                 sim_face_gale_cones, non_sim_faces_sub_sim_gale_cones,
                                 POLYDIM, num_non_sim_faces_sub_sim, pop_size)
        
        # evolve population over num_gen generations and extract terminal states
        terminal_states = terminal_states + evol_pop(initial_pop, num_gen, mut_rate, monitor=False)
        
        # remove the redunancy in the list of terminal states
        terminal_states = remove_redundancy(terminal_states)
        
        print("     "+str(len(terminal_states)))

    return terminal_states


