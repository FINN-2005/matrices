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
```python
from matrices import Mat

# Create matrices
A = Mat([
    [1,2],
    [3,4]
])
B = Mat([
    [5,6],
    [7,8]
])

# Addition
print(A + B)

# Matrix multiplication
print(A * B)

# Element-wise multiplication
print(A @ B)

# Determinant
print(A.det())

# Inverse matrix
print(A.inverse())

# Solve linear system Ax = B
print(A.solve_linear_system(B))

# Transpose
print(A.transpose())

# Generate random matrix
print(Mat.random(input_range=(0, 10), shape=(3, 3), int=True))
```
## Motivation

Built to learn and experiment with matrix operations systematically without relying on NumPy’s abstraction.  
This library aims to help understand the core algorithms behind linear algebra operations.
