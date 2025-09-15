# matrices

A lightweight pure-Python matrix math library written before I started using NumPy.  
This project implements common matrix operations from scratch, focusing on clarity and fundamentals without external dependencies.

## Installation

Install directly from GitHub:

    pip install git+https://github.com/FINN-2005/matrices.git

## Features

- Matrix creation with nested lists, supporting int and float elements  
- Arithmetic operations: addition, subtraction, multiplication (including matrix multiplication), division, power  
- Determinant and inverse for square matrices  
- Solving linear systems via matrix inverse  
- Transpose, element-wise function application, and utility methods like copy, mean, min/max clamping  
- Static methods for generating random matrices and identity matrices  
- Supports iteration, item access, and formatted printing  

## Example Usage

    from matrices import Mat

    # Create matrices
    A = Mat([, ])
    B = Mat([, ])

    # Addition
    C = A + B

    # Matrix multiplication
    D = A * B

    # Element-wise multiplication
    E = A @ B

    # Determinant
    det_A = A.det()

    # Inverse matrix
    A_inv = A.inverse()

    # Solve linear system Ax = B
    x = A.solve_linear_system(B)

    # Transpose
    A_T = A.transpose()

    # Generate random matrix
    R = Mat.random(input_range=(0, 10), shape=(3, 3), int=True)

## Motivation

Built to learn and experiment with matrix operations systematically without relying on NumPy’s abstraction.  
This library aims to help understand the core algorithms behind linear algebra operations.
