import random
import numpy as np
from fitness import fitness_func


class triang:
    """"Triangulation class:
         .points is the list of points of the polytope
         .heights is the list of heights for each point
         .fitness is the fitness of the triangulation
         .terminal is 1 if the triangulation is FRST and 0 otherwise"""
    def __init__(self, points, heights):
        self.points = points
        self.heights = heights
        self.fitness = 0.
        self.terminal = 0
    
    # compute the fitness 
    def compute_fitness(self):
        self.fitness = fitness_func(self.points,self.heights)
        if self.fitness == 0:
            self.terminal = 1


class population:
    """"Population class:
        .state_list is the list of triangulation states
        .size is the population size
        .max_fitness is the maximum fitness score
        .av_fitness is the average fitness score
        .num_term is the number of terminal states"""
    def __init__(self,state_list):
        self.state_list = state_list
        self.size = len(state_list)
        self.max_fitness = 0.
        self.av_fitness = 0.
        self.num_term = 0
        
    def compute_max_fitness(self):
        max_fit = self.state_list[0].fitness
        for i in range(1,self.size):
            if self.state_list[i].fitness > max_fit:
                max_fit = self.state_list[i].fitness
        self.max_fitness = max_fit 
    
    def compute_av_fitness(self):
        sum_fit = 0
        for i in range(self.size):
            sum_fit += self.state_list[i].fitness
        self.av_fitness = sum_fit / self.size
        
    def compute_num_term(self):
        total_term = 0
        for i in range(self.size):
            if self.state_list[i].terminal:
                total_term += 1
        self.num_term = total_term
        
    def mutate_pop(self, mutrate):
        # define the length of the height lists
        height_list_len = len(self.state_list[0].heights)
    
        # determine the total number of heights in the population 
        num_heights = height_list_len * self.size
    
        # determine the number of mutations to perform
        num_mut = round(num_heights * mutrate)
    
        # run over mutations
        for i in range(num_mut):
            # determine a random position in the entire population
            pos = random.choice(range(num_heights))
        
            # determine a random permutation 
            pert = np.random.normal(0,0.05)
        
            # determine the individual where the mutation occurs
            ipos = int(pos/height_list_len)
        
            # determine the position within the individual where the mutation occurs
            hpos = pos % height_list_len
        
            # mutate height
            self.state_list[ipos].heights[hpos] + pert
            
            # update fitness
            self.state_list[ipos].compute_fitness()
        
        # update average and maximum fitness and number of terminal states
        self.compute_max_fitness()
        self.compute_av_fitness()
        self.compute_num_term()
        

def random_heights(points):
    """"Generate a random set of heights by perturbing the Delaunay heights."""
    return [np.dot(p,p) + np.random.normal(0,0.05) for p in points]


def cross_heights(heights1, heights2, numcuts=1):
    """"Cross two height lists."""
    # generate an array of cut positions
    cuts = list(np.sort(random.sample(range(len(heights1)),numcuts)))
    cuts.append(len(heights1))
    
    # swap the relevant parts of the height lists
    new_heights_1 = heights1.copy()
    new_heights_2 = heights2.copy()
    for i in range(0,numcuts,2):
        start = cuts[i]
        end = cuts[i+1]
        for j in range(start,end):
            new_heights_1[j] = heights2[j]
            new_heights_2[j] = heights1[j]
    
    return new_heights_1, new_heights_2
            

def random_triang(points):
    """"Generate a random triangulation state."""
    heights = random_heights(points) 
    T = triang(points, heights)
    T.compute_fitness()
    return T
           
 
# TO DO
def equiv_triang(triang1, triang2):
    """"Determine whether two triangualtions are equivalent."""
    equiv=0
    return equiv

    
def random_pop(points, popsize):
    """"Generate a random population."""
    state_list = []
    for i in range(popsize):
        state_list.append(random_triang(points))
    return population(state_list)


def sort_pop(pop):
    """"Sort a population based on the fitness scores."""
    fitnesses = [pop.state_list[i].fitness for i in range(pop.size)]
    sorted_indices = np.argsort(np.array(fitnesses))
    sorted_state_list = [pop.state_list[i] for i in sorted_indices]
    return population(sorted_state_list)
    

def next_pop(pop, numcuts=1, mutrate=0.01):
    """"Update a population by performing selection, crossover and mutation."""
    # determine the roulette selection probabilities
    df = pop.max_fitness - pop.av_fitness
    if df <= 0:
        p = [1/pop.size for i in range(pop.size)]
    else:
        p = [((3-1)*(pop.state_list[i].fitness-pop.av_fitness)+df)/df/pop.size for i in range(pop.size)]

    # choose individuals for breeding and perform crossover
    states = []
    for i in range(0,pop.size,2):
        # select the positions of two individuals
        indices = random.choices(range(pop.size),p,k=2)
        
        # cross the two selected individuals 
        heights1, heights2 = cross_heights(pop.state_list[indices[0]].heights, 
                                           pop.state_list[indices[1]].heights, numcuts=numcuts)
        
        # define the new triangulations
        state1 = triang(pop.state_list[0].points, heights1)
        state2 = triang(pop.state_list[0].points, heights2)
        state1.compute_fitness()
        state2.compute_fitness()
        states.append(state1)
        states.append(state2)
    
    # define the new population
    new_pop = population(states)
    
    # mutate the new population
    new_pop.mutate_pop(mutrate=mutrate)
    
    return new_pop
            
    
def term_states(pop):
    """"Select terminal states from a population."""
    states = []
    for i in range(pop.size):
        if pop.state_list[i].terminal == 1:
            states.append(pop.state_list[i])
    return states

# redundancy whilst equiv_triang is empty
def removeredundancy(term_states):
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
    

def evol_pop(initial_pop, numgen, numcuts, mutrate):
    """"Evolve a population over generations and extract terminal states."""
    # load initial population 
    evol = [initial_pop]
    
    # define terminal states list
    term_states_list = [term_states(initial_pop)]

    # loop over generations
    print("# Terminal States   Total # Terminal States",)
    pop = initial_pop
    for i in range(numgen):
        new_pop = next_pop(pop, numcuts=numcuts, mutrate=mutrate)
        evol.append(new_pop)
        
        term_states_list = term_states_list + term_states(new_pop)
        removeredundancy(term_states_list)
        
        print(str(new_pop.num_term)+"                    "+str(len(term_states_list)))
        
        pop = new_pop

    return term_states_list


