import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt


def draw_function(f):

    # Generate a grid over the range [-10, 10] x [-10, 10]
    x_vals = np.linspace(-10, 10, 500)
    y_vals = np.linspace(-10, 10, 500)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Evaluate the function over the grid
    Z = np.array([[f([x, y]) for x, y in zip(row_x, row_y)] for row_x, row_y in zip(X, Y)])

    # Plot the 2D scalar map
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(X, Y, Z, levels=50, cmap='viridis')
    plt.colorbar(contour, label='Function Value')
    plt.title('2D Scalar Map of Function f')
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def quartic():
    xmin = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    def f(x):
        out = 0.0
        for i in range(len(x)):
            out += (x[i] - xmin[i])**4
        return out

    x0 = np.array([10.0, -10.0, 0.0 , 57.0, -23.0])
    out = opt.minimize(f, x0, method='CG', tol=1e-10,  options={'disp': True, 'return_all': True})
    print(out)



def general_quadratic():

    # Define the function
    def f(x):
        # Define the matrix and vector
        a = np.array([[2.0, 0.1], [0.3, 10.0]])  # Equivalent to SMatrix<2, 2>
        b = np.array([1.0, 2.0])                 # Equivalent to SVector<2>
        x = np.array(x)
        
        # Compute the value of the function
        return np.dot(x, np.dot(a, x)) + np.dot(b, x)

    # draw_function(f)
    out = opt.minimize(f, [1, 1], method='CG', )
    print(out)


def rosenbrock():

    def f(xarr): 
        x = xarr[0]
        y = xarr[1]
        a = 1.0
        b = 100.0
        return (a- x)**2 + b * (y - x**2)**2


    # draw_function(f)
    out = opt.minimize(f, [0, 3], method='CG', options={'disp': True, 'return_all': True})
    print(out)



def main():
    # quartic()
    rosenbrock()


if __name__ == "__main__":
    main()