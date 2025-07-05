
import sympy as sp


def quadratic_1d():
    x, r1, r2 = sp.symbols('x r1 r2')
    gx = (x - r1) * (x - r2)

def cubic_1d():
    x, r1, r2, r3 = sp.symbols('x r1 r2 r3')
    gx = (x - r1) * (x - r2) * (x - r3)

    dgx = sp.diff(gx, x)
    dgx_expanded = sp.expand(dgx)
    dgx_collected = sp.collect(dgx_expanded, x)
    sp.pprint(dgx_collected) 
    critical_points = sp.solve(sp.Eq(dgx, 0), x)
    dgx_substituted = [critical_points[i].subs({r1: 1, r2: 2, r3: 3}) for i in range(len(critical_points))]
    # sp.pprint(dgx_substituted)
    sp.pprint(sp.simplify(dgx_substituted[0]))


def quadratic_form1():

    # Define the symbols
    x, y, z = sp.symbols('x y z')
    # Define a vector of variables
    variables = sp.Matrix([x, y, z])

    # Define a symmetric matrix A (for a proper quadratic form)
    # You can initialize with specific values or keep it symbolic
    a11, a12, a13 = sp.symbols('a11 a12 a13')
    a21, a22, a23 = sp.symbols('a21 a22 a23')
    a31, a32, a33 = sp.symbols('a31 a32 a33')

    A_matrix = sp.Matrix([[a11, a12, a13],
                          [a21, a22, a23],
                          [a31, a32, a33]])

    # Alternatively, you can use the MatrixSymbol directly
    # A_matrix = sp.Matrix(A)

    # Calculate the quadratic form x^T A x
    quadratic_form = variables.transpose() * A_matrix * variables

    # Expand the result to get the full expression
    expanded_form = sp.expand(quadratic_form[0, 0])
    print("\nExpanded form:")
    sp.pprint(expanded_form)

    # Compute the gradient with respect to x, y, z
    grad = sp.Matrix([sp.diff(expanded_form, var) for var in (x, y, z)])
    print("\nGradient:")
    sp.pprint(grad)


def quadratic_form2():


    # Define the symbols
    x, y, z = sp.symbols('x y z')
    c1, c2, c3 = sp.symbols('c1 c2 c3')

    # Define a vector of variables
    variables = sp.Matrix([x, y, z])
    constants = sp.Matrix([c1, c2, c3])

    # Define a symmetric matrix A (for a proper quadratic form)
    # You can initialize with specific values or keep it symbolic
    a11, a12, a13 = sp.symbols('a11 a12 a13')
    a21, a22, a23 = sp.symbols('a21 a22 a23')
    a31, a32, a33 = sp.symbols('a31 a32 a33')

    A_matrix = sp.Matrix([[a11, a12, a13],
                          [a21, a22, a23],
                          [a31, a32, a33]])

    # Alternatively, you can use the MatrixSymbol directly
    # A_matrix = sp.Matrix(A)

    # Calculate the quadratic form x^T A x
    quadratic_form = variables.transpose() * A_matrix * variables

    # Expand the result to get the full expression
    expanded_form = sp.expand(quadratic_form[0, 0])
    print("\nExpanded form:")
    sp.pprint(expanded_form)


def quadratic_form3():

    # Define the symbols
    x, y, z = sp.symbols('x y z')
    # Define a vector of variables
    variables = sp.Matrix([x, y, z])

    # Define a symmetric matrix A with specific values
    A_matrix = sp.Matrix([[5, 1, 2],
                          [1, 5, 3],
                          [2, 3, 5]])

    # Calculate the quadratic form x^T A x
    quadratic_form = variables.transpose() * A_matrix * variables

    # Expand the result to get the full expression
    expanded_form = sp.expand(quadratic_form[0, 0])
    print("\nExpanded form:")
    sp.pprint(expanded_form)

    # Compute the gradient with respect to x, y, z
    grad = sp.Matrix([sp.diff(expanded_form, var) for var in (x, y, z)])
    print("\nGradient:")
    sp.pprint(grad)

    # Evaluate the value and gradient at [x, y, z] = [1, 1, 1]
    subs_dict = {x: 1, y: 1, z: 1}
    value_at_zero = expanded_form.subs(subs_dict)
    grad_at_zero = grad.subs(subs_dict)
    print("\nValue at: [1, 1, 1]", value_at_zero)
    print("Gradient at [1, 1, 1]:")
    sp.pprint(grad_at_zero)

def main():
    cubic_1d()
    # quadratic_form3()






if __name__ == "__main__":
    main()


