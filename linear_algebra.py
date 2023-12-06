import numpy as np

# just the bare basics

# scalars, vectors, matrices, tensors
age = 22 # just a single numerical value
CSgroupAges = np.array([19, 25, 52, 44, 21]) # a vector, so just a single sequence of values
CSgroupStudents = np.array([ # a matrix, so kinda "2D vector"
    # col one is student ID and col 2 is their age
    [1, CSgroupAges[0]],
    [2, CSgroupAges[1]],
    [3, CSgroupAges[2]],
    [4, CSgroupAges[3]],
    [5, CSgroupAges[4]]
])
# what about their classes? well ...
classes = np.array(["CS", "Law", "Medicine", "Data science", "CS"]).reshape(1, 5, 1) # create a tensor with classes (a "3D vector")
CSgroupStudents = np.concatenate((CSgroupStudents.reshape(1, CSgroupStudents.shape[0], CSgroupStudents.shape[1]), classes), axis=2) # take the original students matrix, take the classes tensor, and merge them through concat
ages = CSgroupStudents[0, :, 1].astype(int) # store the ages from the tensor of students as a single vector of integers
print(f" Mean age: {np.mean(ages)}") # calc the mean age using the ages vector
# matrix operations
# addition
A = np.array([
    [22, 1],
    [1, 22]
])
B = np.array([
    [3, 3],
    [3, 3]
])
A + B # element-wise addition
# subtraction
A - B # element-wise subtraction
# multiplication
A * B # element-wise multiplication
# division
A / B # element-wise division
# matrix multiplication
print(A @ B) # matrix multiplication (dot product) - not elem-wise !!
print("\nMore advanced:")
# transpose
C = np.array([
    [1, 2, 3],
    [4, 5, 6]
])
C.T # transpose of a matrix
# determinant
def determinant(mat):
    if mat.shape == (2, 2):
        return mat[0, 0] * mat[1, 1] - mat[0, 1] * mat[1, 0]
    else:
        det = 0
        
        return det
print(determinant(A)) # determinant of a matrix (only for square matrices)

# inverse
def inverse(mat):
    if mat.shape == (2, 2):
        det = determinant(mat)
    raise NotImplementedError
print(inverse(A)) # inverse of a matrix (only for square matrices



# print(np.linalg.det(A)) # determinant of a matrix (only for square matrices)
# # inverse
# print(np.linalg.inv(A)) # inverse of a matrix (only for square matrices)
# print(np.around(A @ np.linalg.inv(A))) # identity matrix (rounded up due to floating point errors)
# # trace
# print(np.trace(A)) # trace of a matrix (sum of diagonal elements)
# # norm
# print(np.linalg.norm(A)) # norm of a matrix (square root of sum of squared elements)
# # rank
# print(np.linalg.matrix_rank(A)) # rank of a matrix (number of linearly independent rows/columns)