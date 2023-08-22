from cytools import Polytope
from cytools.triangulation import Triangulation

# TO DO
def fitness_func(points, heights):
    """"Compute the fitness of a triangulation given the points and a set of
    heights."""
    score = 0

    poly = Polytope(points)
    triang_pts = [tuple(pt) for pt in poly.points()]
    t = Triangulation(triang_pts, poly=poly, heights = heights, make_star = True)
    if not t.is_fine():
        score += 1

    if not t.is_star():
        score += 1

    if not t.is_regular():
        score += 1

    # TODO Additional conditions

    return score
