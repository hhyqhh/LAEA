from problem.LZG import LZG01
from pymoo.optimize import minimize

from algorithm.lsea import LSEA


if __name__=='__main__':

    problem = LZG01(n_var=5)

    algorithm = LSEA(pop_size=50)


    res = minimize(problem,
                   algorithm,
                   ('n_evals', 300),
                   verbose=True)
