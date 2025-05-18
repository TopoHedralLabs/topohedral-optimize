
import sympy as sp


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



def main():
    quadratic_form1()






if __name__ == "__main__":
    main()


