
import scipy.optimize as opt
import math as m
import typing as tp
import numpy as np
import matplotlib.pyplot as plt
import json


def plot_fcn(fcn: tp.Callable[[float], float], lims: tp.Tuple[float, float], n_pts: int = 300) -> None:

    x = np.linspace(lims[0], lims[1], n_pts)
    fx = [fcn(x_) for x_ in x]
    plt.plot(x, fx)
    plt.show()

def to_bracket(lim: tp.Tuple[float, float]) -> tp.Tuple[float, float, float]:
    return (lim[0], lim[0]+(lim[1]-lim[0])/2, lim[1])   

def minimiize_scalar_brent_tests(results: dict):

    def f1(x):
        return m.exp(x) - 4*x

    lim1 = (0.0, 3.0)
    desc1 = "Test minimisation of exp(x) - 4x on [0,3]"


    def f2(x): 
        return 1e-8 * (x**2)
    
    lim2 = (-10000, 10000)
    desc2 = "Test minimisation of 1e-8x^2 on [-10000,10000]"


    def f3(x): 
        return x**2 + 0.1 * np.sin(50*x)
    
    lim3 = (-2.0, 2.0)
    desc3 = "Test minimisation of x^2 + 0.1sin(50x) on [-2,2]"


    def f4(x): 
        return abs(x - 2.0) + 1.0
    
    lim4 = (0.0, 4.0)
    desc4 = "Test minimisation of abs(x - 2) + 1 on [0,4]"    

    def f5(x): 
        return (x**2 - 4) ** 2
    
    lim5 = (-3, 5) 
    desc5 = "Test minimisation of (x^2 - 4)^2 on [-3,5]"    

    funs = [f1, f2, f3, f4, f5]
    lims = [lim1, lim2, lim3, lim4, lim5]
    descs = [desc1, desc2, desc3, desc4, desc5]

    for i in range(len(funs)):
        # Run minimization and store results
        out = opt.minimize_scalar(funs[i], method = "Brent", tol = 1e-8, bracket=to_bracket(lims[i]), options={"disp": True})

        name = 'minimise_scalar_brent_test{}'.format(i+1)

        results[name] = dict()
        results[name]["description"] = descs[i]
        results[name]["values"] = dict()
        results[name]["values"]["bracket"] = to_bracket(lims[i])
        results[name]["values"]["xmin"] = out.x
        results[name]["values"]["fmin"] = out.fun
        results[name]["values"]["niter"] = out.nit
        results[name]["values"]["nfeval"] = out.nfev
#..................................................................................................

def minimiize_scalar_bounded_tests(results: dict):

    def f1(x):
        return m.exp(x) - 4*x

    lim1 = (0.0, 3.0)
    desc1 = "Test minimisation of exp(x) - 4x on [0,3]"


    def f2(x): 
        return 1e-8 * (x**2)
    
    lim2 = (-10000, 10000)
    desc2 = "Test minimisation of 1e-8x^2 on [-10000,10000]"


    def f3(x): 
        return x**2 + 0.1 * np.sin(50*x)
    
    lim3 = (-2.0, 2.0)
    desc3 = "Test minimisation of x^2 + 0.1sin(50x) on [-2,2]"


    def f4(x): 
        return abs(x - 2.0) + 1.0
    
    lim4 = (0.0, 4.0)
    desc4 = "Test minimisation of abs(x - 2) + 1 on [0,4]"    

    def f5(x): 
        return (x**2 - 4) ** 2
    
    lim5 = (-3, 5) 
    desc5 = "Test minimisation of (x^2 - 4)^2 on [-3,5]"    

    def f6(x):
        return m.exp(x) - 4*x

    lim6 = (2.0, 3.0)
    desc6 = "Test minimisation of exp(x) - 4x on [2,3]"
    

    def f7(x): 
        return 1e-8 * (x**2)
    
    lim7 = (-10000, 0)
    desc7 = "Test minimisation of 1e-8x^2 on [-10000,0]"


    def f8(x): 
        return x**2 + 0.1 * np.sin(50*x)
    
    lim8 = (0.0, 2.0)
    desc8 = "Test minimisation of x^2 + 0.1sin(50x) on [0,2]"
    

    def f9(x): 
        return abs(x - 2.0) + 1.0
    
    lim9 = (2.0, 4.0)
    desc9 = "Test minimisation of abs(x - 2) + 1 on [0,4]"    

    def f10(x): 
        return (x**2 - 4) ** 2
    
    lim10 = (-1, 0) 
    desc10 = "Test minimisation of (x^2 - 4)^2 on [-1,0]"        

    funs = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10]
    lims = [lim1, lim2, lim3, lim4, lim5, lim6, lim7, lim8, lim9, lim10]
    descs = [desc1, desc2, desc3, desc4, desc5, desc6, desc7, desc8, desc9, desc10]

    for i in range(len(funs)):
        # Run minimization and store results
        out = opt.minimize_scalar(funs[i], method = "Bounded", bounds=lims[i], tol = 1e-8)

        name = 'minimise_scalar_bounded_test{}'.format(i+1)

        results[name] = dict()
        results[name]["description"] = descs[i]
        results[name]["values"] = dict()
        results[name]["values"]["bounds"] = lims[i]
        results[name]["values"]["xmin"] = out.x
        results[name]["values"]["fmin"] = out.fun
        results[name]["values"]["niter"] = out.nit
        results[name]["values"]["nfeval"] = out.nfev
#..................................................................................................
        
def debug_test():

    def f1(x):
        return m.exp(x) - 4*x

    lim1 = (0.0, 3.0)
    desc1 = "Test minimisation of exp(x) - 4x on [0,3]"
    out = opt.minimize_scalar(f1, method = "Bounded", tol = 1e-8, bounds=lim1, options={"disp": True})

        
def main():

    # debug_test()

    # results_brent = dict()
    # minimiize_scalar_brent_tests(results_brent)
    # with open('minimise-scalar-brent.json', 'w') as f:
    #     json.dump(results_brent, f, indent=4)


    results_bounded = dict()
    minimiize_scalar_bounded_tests(results_bounded)
    with open('minimise-scalar-bounded.json', 'w') as f:
        json.dump(results_bounded, f, indent=4)
   
    
    


if __name__ == '__main__':
    main()