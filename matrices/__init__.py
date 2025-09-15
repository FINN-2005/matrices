from typing import Tuple, Union
import random
import math

class Mat:
    e = math.e

    def __init__(self, data: list[list[Union[int, float]]]) -> None:
        if not data or not all(isinstance(row, list) for row in data):
            raise ValueError("Data must be a list of lists.")
        
        self.data = data
        self.rows = len(data)
        self.columns = len(data[0]) if self.rows > 0 else 0
        self.shape = (self.rows, self.columns)

    def __len__(self) -> int:
        """Return the number of rows in the matrix."""
        return self.rows

    def __repr__(self) -> str:
        col_widths = [max(len(str(item)) for item in col) for col in zip(*self.data)] if self.data else []
        formatted_rows = []
        for row in self.data:
            formatted_row = ' | '.join(f"{str(item).rjust(width)}" for item, width in zip(row, col_widths))
            formatted_rows.append(f"| {formatted_row} |")
        matrix_str = '\n'.join(formatted_rows)        
        return matrix_str
    
    def __add__(self, other: Union['Mat', int, float]) -> 'Mat':
        if isinstance(other, Mat):
            if self.shape != other.shape:
                raise ValueError("Matrices must have the same dimensions for addition.")
            return Mat([[self.data[i][j] + other.data[i][j] for j in range(self.columns)] for i in range(self.rows)])
        elif isinstance(other, (int, float)):
            return Mat([[self.data[i][j] + other for j in range(self.columns)] for i in range(self.rows)])
        else:
            raise TypeError("Unsupported operand type for +: '{}'".format(type(other).__name__))
         
    def __sub__(self, other: Union['Mat', int, float]) -> 'Mat':
        if isinstance(other, Mat):
            if self.shape != other.shape:
                raise ValueError("Matrices must have the same dimensions for subtraction.")
            return Mat([[self.data[i][j] - other.data[i][j] for j in range(self.columns)] for i in range(self.rows)])
        elif isinstance(other, (int, float)):
            return Mat([[self.data[i][j] - other for j in range(self.columns)] for i in range(self.rows)])
        else:
            raise TypeError("Unsupported operand type for -: '{}'".format(type(other).__name__))
        
    def __mul__(self, other: Union['Mat', int, float]) -> 'Mat':
        if isinstance(other, Mat):
            if self.columns != other.rows:
                raise ValueError("Number of columns in the first matrix must be equal to the number of rows in the second matrix.")
            result_data = []
            for i in range(self.rows):
                result_row = []
                for j in range(other.columns):
                    sum_product = sum(self.data[i][k] * other.data[k][j] for k in range(self.columns))
                    result_row.append(sum_product)
                result_data.append(result_row)
            return Mat(result_data)
        elif isinstance(other, (int, float)):
            return Mat([[element * other for element in row] for row in self.data])
        else:
            raise TypeError("Unsupported operand type for *: '{}'".format(type(other).__name__))
    
    def __truediv__(self, other: Union['Mat', int, float]) -> 'Mat':
        if isinstance(other, (int, float)):
            if other == 0:
                raise ValueError("Division by zero is not allowed.")
            return Mat([[element / other for element in row] for row in self.data])
        elif isinstance(other, Mat):
            if self.shape != other.shape:
                raise ValueError("Matrices must have the same dimensions for element-wise division.")
            return Mat([[self.data[i][j] / other.data[i][j] if other.data[i][j] != 0 else float('inf') 
                         for j in range(self.columns)] for i in range(self.rows)])
        else:
            raise TypeError("Unsupported operand type(s) for /: '{}'".format(type(other).__name__))
        
    def __pow__(self, power: int) -> 'Mat':
        if self.rows != self.columns:
            raise ValueError("Matrix must be square to raise to a power.")
        if power < 0:
            raise ValueError("Negative powers are not supported.")
        if power == 0:
            return Mat([[1 if i == j else 0 for j in range(self.columns)] for i in range(self.rows)])  # Identity matrix
        elif power == 1:
            return self
        result = Mat([[1 if i == j else 0 for j in range(self.columns)] for i in range(self.rows)])  # Identity matrix
        base = self
        while power > 0:
            if power % 2 == 1:
                result = result * base
            base = base * base
            power //= 2
        return result
        
    def __round__(self, n: int = 5) -> 'Mat':
        return Mat([[round(element, n) for element in row] for row in self.data])    
        
    def __neg__(self) -> 'Mat':
        return Mat([[-element for element in row] for row in self.data])
        
    def apply_function_element_wise(self, func) -> 'Mat':
        return Mat([[func(element) for element in row] for row in self.data])

    def apply_function_row_wise(self, func) -> 'Mat':
        return Mat([func(row) for row in self.data])

    def transpose(self) -> 'Mat':
        return Mat(list(map(list, zip(*self.data))))
    
    def det(self) -> float:
        """ Calculate the determinant of a square matrix using Laplace expansion. """
        if self.rows != self.columns:
            raise ValueError("Matrix must be square to compute its determinant.")
        if self.rows == 1:
            return self.data[0][0]
        if self.rows == 2:
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        
        det = 0
        for c in range(self.columns):
            sub_matrix = [row[:c] + row[c+1:] for row in self.data[1:]]
            det += ((-1) ** c) * self.data[0][c] * Mat(sub_matrix).det()
        return det   
    
    def inverse(self) -> 'Mat':
        if self.rows != self.columns:
            raise ValueError("Matrix must be square to compute its inverse.")
        det = self.det()
        if det == 0:
            raise ValueError("Matrix is singular and cannot be inverted.")
        def _cofactor(matrix, row, col):
            return [r[:col] + r[col+1:] for r in (matrix[:row] + matrix[row+1:])]
        def _adjugate(matrix):
            size = len(matrix)
            adjugate = []
            for i in range(size):
                adjugate_row = []
                for j in range(size):
                    minor = _cofactor(matrix, i, j)
                    adjugate_row.append(((-1) ** (i + j)) * Mat(minor).det())
                adjugate.append(adjugate_row)
            return Mat(list(map(list, zip(*adjugate)))).apply_function_element_wise(lambda x: x / det)
        return _adjugate(self.data)

    def solve_linear_system(self, other: 'Mat') -> 'Mat':
        """ Solve the linear system Ax = other for x """
        if self.rows != self.columns:
            raise ValueError("Matrix must be square to solve a linear system.")
        if other.rows != self.rows:
            raise ValueError("Right-hand side matrix dimensions do not match.")
        return self.inverse() * other  # Use inverse to solve Ax = B => x = A^-1 * B

    @staticmethod
    def random(input_range: Tuple[float, float], shape: Tuple[int, int], int: bool = False) -> 'Mat':
        min_val, max_val = input_range
        rows, cols = shape
        if int:
            return Mat([[random.randint(min_val, max_val) for _ in range(cols)] for _ in range(rows)])
        else:
            return Mat([[random.uniform(min_val, max_val) for _ in range(cols)] for _ in range(rows)])
    
    def min_max(self, min_val, max_val):
        return self.apply_function_element_wise(lambda x: max(min_val, min(x, max_val)))
    
    @staticmethod
    def random_seed(seed: int):
        random.seed(seed)
        
    def copy(self) -> 'Mat':
        """ Return a new Mat object with the same data. """
        return Mat([row[:] for row in self.data])
    
    def mean(self) -> float:
        """ Calculate the mean of all elements in the matrix. """
        total_sum = sum(sum(row) for row in self.data)
        num_elements = self.rows * self.columns
        return total_sum / num_elements if num_elements > 0 else 0.0
        
    def __iter__(self):
        for row in self.data:
            for element in row:
                yield element
    
    def __getitem__(self, index: int) -> list[Union[int, float]]:
        if index >= self.rows or index < 0:
            raise IndexError("Index out of range.")
        return self.data[index]
    
    def iter_row_wise(self):
        for row in self.data:
            yield row
    
    @staticmethod
    def identity(size: int) -> 'Mat':
        """Generate an identity matrix of the given size."""
        if size <= 0:
            raise ValueError("Size of identity matrix must be a positive integer.")
        return Mat([[1 if i == j else 0 for j in range(size)] for i in range(size)])
    
    def __matmul__(self, other: Union['Mat', int, float]) -> 'Mat':
        if isinstance(other, Mat):
            if self.shape != other.shape:
                raise ValueError("Matrices must have the same dimensions for element-wise multiplication.")
            result_data = [[self.data[i][j] * other.data[i][j] for j in range(self.columns)] for i in range(self.rows)]
            return Mat(result_data)
        elif isinstance(other, (int, float)):
            return Mat([[element * other for element in row] for row in self.data])
        else:
            raise TypeError("Unsupported operand type(s) for @: '{}'".format(type(other).__name__))
