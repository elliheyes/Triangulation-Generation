from GA import random_pop, evol_pop

points = [[1,0,0],[0,1,0],[0,0,1],[-1,-1,-1]]

initial_pop = random_pop(points, 100)

term_states = evol_pop(initial_pop, 100, numcuts=1, mutrate=0.01)