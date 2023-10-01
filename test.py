import pytest
from matrix_multiplication import MatrixMultiplication
import random
from time import *

n = 512 #define the matrix size here

@pytest.fixture(scope="function")
def operands():
    a = [[random.random() for _ in range(n)] for _ in range(n)]
    b = [[random.random() for _ in range(n)] for _ in range(n)]
    return a, b

@pytest.mark.benchmark(
    min_rounds=5,
    min_time=1.0,  
    disable_gc=True,  
)

def test_matrix_multiplication(benchmark, operands):
    a, b = operands
    matrix_multiplier = MatrixMultiplication()
    benchmark(matrix_multiplier.execute, a, b)
