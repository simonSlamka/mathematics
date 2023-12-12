import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import mplcyberpunk

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
3 * A # scalar multiplication
A + B # element-wise addition
# subtraction
A - B # element-wise subtraction
# multiplication
A * B # element-wise multiplication
# division
A / B # element-wise division
# matrix multiplication
def matmul(A, B):
    if A.shape[1] == B.shape[0]:
        n = np.zeros((A.shape[0], B.shape[1]))
        for row in range(A.shape[0]):
            for col in range(B.shape[1]):
                for i in range(A.shape[1]):
                    n[row,col] += A[row,i] * B[i,col]
        return n
    else:
        raise ValueError("Cannot multiply matrices of these shapes")
if np.array_equal(A @ B, matmul(A, B)):
    print("matmul understood!")
# transpose
C = np.array([
    [1, 2, 3],
    [4, 5, 6]
])
def transpose(mat):
    n = np.zeros((mat.shape[1], mat.shape[0]))
    for row in range(mat.shape[0]):
        for col in range(mat.shape[1]):
            n[col,row] = mat[row,col]
    return n
if np.array_equal(C.T, transpose(C)):
    print("transpose understood!")
# C.T # transpose of a matrix

# determinant
def determinant(mat):
    if mat.shape == (2, 2):
        return mat[0, 0] * mat[1, 1] - mat[0, 1] * mat[1, 0]
    else:
        det = 0
        return det
if np.around(determinant(A)) == np.around(np.linalg.det(A)):
    print("determinant understood!")
# print(determinant(A)) # determinant of a matrix (only for square matrices)

# inverse
def inverse(mat):
    if mat.shape == (2, 2):
        det = determinant(mat)
        return np.array([
            [mat[1, 1], -mat[0, 1]],
            [-mat[1, 0], mat[0, 0]]
        ]) / det
    else:
        raise ValueError("Cannot invert matrices of these shapes")
I = np.around(matmul(A, inverse(A))) # the identity matrix of A, rounded up due to floating point errors
if np.array_equal(I, np.eye(2)):
    print("inverse understood!")

# eigenvalues and eigenvectors
O = np.array([
    [6, 9],
    [9, 6]
])
def eigenvalues(mat):
    if mat.shape == (2, 2):
        a, b, c, d = mat[0, 0], mat[0, 1], mat[1, 0], mat[1, 1]

        x = 1
        y = -(a + d)
        z = determinant(mat)

        discr = y ** 2 - 4 * x * z

        lambda1 = (-y + np.sqrt(discr)) / (2 * x)
        lambda2 = (-y - np.sqrt(discr)) / (2 * x)

        return lambda1, lambda2
if np.array_equal(eigenvalues(O), np.linalg.eigvals(O)):
    print("eigenvalues understood!")

def eigenvectors(mat):
    # if np.array_equal(inverse(mat) * mat, np.eye(2)) and eigenvalues(mat)[0] != eigenvalues(mat)[1]: # check if the matrix is invertible and has distinct eigenvalues
    def pick_eigenvector(augMat):
        v1 = 1
        v2 = -augMat[0, 1] / augMat[0, 0]
        return np.array([v1, v2])
    def gaussian(mat):
        rows, cols = mat.shape
        for i in range(rows):
            # fwd elimination
            if mat[i, i] == 0:
                for j in range(i + 1, rows):
                    if mat[j, i] != 0:
                        mat[[i, j]] = mat[[j, i]]
                        break
                else:
                    raise ValueError("No unique solution found!")
            for j in range(i + 1, rows):
                f = mat[j, i] / mat[i, i]
                if mat[i, i] == 0:
                    raise ValueError("No unique solution found!")
                mat[j, :] -= f * mat[i, :]
        # bwd substitution
        for i in range(rows - 1, -1, -1):
            mat[i, :] /= mat[i, i]
            for j in range(i - 1, -1, -1):
                f = mat[j, i]
                mat[j, :] -= f * mat[i, :]

        return mat[:, -1]
    if mat.shape == (2, 2):
        lambdas = eigenvalues(mat)
        eigenvectors = []
        for v in lambdas:
            augMat = mat - v * np.eye(2)
            eigenvectors.append(pick_eigenvector(augMat))

        return eigenvectors
        if (determinant(lambdas[0]*np.eye(2)-mat)) or (determinant(lambdas[1]*np.eye(2)-mat)):
            aug1 = np.concatenate((mat - lambdas[0] * np.eye(2), np.zeros((2, 1))), axis=1)
            aug1 = aug1[:-1, :]
            print(aug1)
            breakpoint()
            aug2 = np.concatenate((mat - lambdas[1] * np.eye(2), np.zeros((2, 1))), axis=1)
            # check if either of the augmented matrices contains multiples 

            return gaussian(aug1), gaussian(aug2)

def plot_eigenvectors(mat):
    eigvs = eigenvectors(mat)
    plt.plot([0, eigvs[0][0]], [0, eigvs[0][1]], color="red")
    plt.plot([0, eigvs[1][0]], [0, eigvs[1][1]], color="blue")
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.grid()
    plt.show()
def transformImg(img: str, mat):
    img = Image.open(img)
    img = img.convert("L")
    data = np.asarray(img)
    cov = np.cov(data)
    eigsvals, eigs = np.linalg.eig(cov)
    trans = np.dot(data, eigs)
    trans = np.real(trans)
    transImg = Image.fromarray(trans)
    transImg.show()
# transformImg("test.png", O)
plot_eigenvectors(O)
eigvals, eigvects = eigenvalues(O), eigenvectors(O)
def is_eigenvactor_valid(mat, eigv, eigval):
    Av = np.dot(mat, eigv)
    lambdav = eigval * eigv
    return np.allclose(Av, lambdav)
for i in range(len(eigvals)):
    if is_eigenvactor_valid(O, eigvects[i], eigvals[i]):
        print("eigenvectors understood!")
    else:
        print("eigenvectors not understood!")
        print(f"eigenvec: {eigenvectors[i]}")

# mean
def mean(x): # average
    return sum(x) / len(x)

# covariance
def cov(x, y):
    if len(x) != len(y):
        raise ValueError("can't get cov of vects of diff lens")
    else:
        n = len(x)
        xmean, ymean = mean(x), mean(y)
        cov = sum((x[i] - xmean) * (y[i] - ymean) for i in range(n)) / (n - 1)

        return cov

# covariance matrix
def cov_mat(data):
    n = data.shape[1]
    covMat = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            covMat[i, j] = cov(data[:, i], data[:, j])

    return covMat

# let's try this on the iris dataset from sklearn
iris = load_iris()
X = iris.data
Xstd = StandardScaler().fit_transform(X)
covMat = cov_mat(Xstd)
if np.allclose(covMat, np.cov(Xstd.T)):
    print("covariance matrix understood!")

# PCA
def pca(X):
    X = StandardScaler().fit_transform(X)
    covMat = cov_mat(X)
    eigvals, eigvecs = np.linalg.eig(covMat)
    k = 3
    idx = eigvals.argsort()[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    eigvecs = eigvecs[:, :k]

    Xpca = np.dot(X, eigvecs)

    plt.figure(figsize=(8, 8))
    plt.scatter(Xpca[:, 0], Xpca[:, 1])
    plt.xlabel(f"PC1 ({eigvals[0] / sum(eigvals) * 100:.2f}%)")
    plt.ylabel(f"PC2 ({eigvals[1] / sum(eigvals) * 100:.2f}%)")
    plt.grid()
    plt.show()

    # scree
    totVar = sum(eigvals)
    varExpl = [(i / totVar) for i in sorted(eigvals, reverse=True)]
    cumVarExpl = np.cumsum(varExpl)

    plt.figure(figsize=(8, 8))
    plt.bar(range(1, len(varExpl) + 1), varExpl, alpha=0.5, align="center", label="individual explained variance")
    plt.step(range(1, len(cumVarExpl) + 1), cumVarExpl, where="mid", label="cumulative explained variance")
    plt.ylabel("Explained variance ratio")
    plt.xlabel("Principal components")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

pca(X)


def median(x): # middle value
    n = len(x)
    x = sorted(x)
    if n % 2 == 0:
        return (x[n // 2 - 1] + x[n // 2]) / 2
    else:
        return x[n // 2]

def mode(x): # most frequent value
    x = sorted(x)
    return max(set(x), key=x.count)

def variance(x): # how spread out the data is
    n = len(x)
    xmean = mean(x)
    var = sum((x[i] - xmean) ** 2 for i in range(n)) / (n - 1)
    return var

def std(x): # not *that* std. this std is the square root of the variance
    return np.sqrt(variance(x))


# linear regression
# Mark's monthly savings
x = np.array([1, 2, 3, 4, 5, 6, 7, 8]) # Jan to Aug
y = np.array([124, 164, 852, 593, 921, 109, 102, 123]) # savings in caps (lol)
y = np.log(y) # log transform

def mse(y, ypred):
    return sum((y[i] - ypred[i]) ** 2 for i in range(len(y))) / len(y)

m = 0
b = 0

eta = 0.00001
epochs = 10000

for epoch in range(epochs):
    ypred = m * x + b
    dm = (-2 / len(x)) * sum(x * (y - ypred))
    db = (-2 / len(x)) * sum(y - ypred)
    m = m - eta * dm
    b = b - eta * db
    print(f"epoch {epoch + 1}: m = {m}, b = {b}, mse = {mse(y, ypred)}")

print(f"m: {m}, b: {b}, mse: {mse(y, ypred)}")

xPred = np.array([9, 10, 11, 12]) # Sep to Dec
yPred = m * xPred + b

for month, savings in zip(xPred, yPred):
    print(f"month {month}: {savings}")

plt.figure(figsize=(12, 12))
plt.plot(x, y, "ro", label="actual")
plt.plot(xPred, yPred, "bo", label="predicted")
plt.plot(x, m * x + b, label="regression line")
for i in range(len(x)):
    plt.plot([x[i], x[i]], [y[i], m * x[i] + b], color="black", linestyle="dashed", label="error")
plt.title("Mark's savings")
plt.xlabel("month")
plt.ylabel("savings")
plt.grid()
plt.legend()


# logistic regression
# sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def J(y, ypred): # cross entropy
    epsilon = 1e-15
    ypred = np.clip(ypred, epsilon, 1 - epsilon)
    return (-1 / len(y)) * sum(y * np.log(ypred) + (1 - y) * np.log(1 - ypred))

def dJ(ytrue, ypred):
    return ypred - ytrue

# Mark's frequency of intermingling with Aurora
x = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12]]) # Jan to Dec
y = np.array([0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0]) # 0 means no intermingling, 1 means intermingling (she wanted to try at first, but then decided against it, up until July when she decided to try again (LOL))

w = np.zeros((1, 1))
b = 0

eta = 0.01
epochs = 10000

for epoch in range(epochs):
    z = np.dot(x, w) + b
    ypred = sigmoid(z).flatten()

    loss = J(y, ypred)

    grad = dJ(y, ypred)
    dw = np.dot(x.T, grad) / len(y) # derivative of loss with respect to w
    db = np.mean(grad) # derivative of loss with respect to b

    w -= eta * dw # update w
    b -= eta * db # update b

    if epoch % 1000 == 0:
        print(f"epoch {epoch + 1}: w = {w}, b = {b}, loss = {loss}")

print(f"final loss: {loss}")

xTest = np.array([[i] for i in range(1, 25)])
yTest = sigmoid(np.dot(xTest, w) + b)
yClass = (yTest >= 0.5).astype(int) # convert to 0 or 1

plt.style.use("cyberpunk")
plt.figure(figsize=(12, 8))
plt.scatter(x, y, color="yellow", label="Actual intermingling")
plt.plot(xTest, yTest, color="magenta", linestyle="dashed", label="Predicted probability of intermingling")
for i in range(len(xTest)):
    if yClass[i] == 1:
        plt.scatter(xTest[i], yClass[i], color="blue", label="Predicted intermingling bool")
    else:
        plt.scatter(xTest[i], yClass[i], color="red", label="Predicted intermingling bool")
plt.title("Mark's intermingling with Aurora over two years", fontsize=16)
plt.xlabel("Month", fontsize=14)
plt.ylabel("Intermingling (bool)", fontsize=14)
plt.xticks(np.arange(1, 25, 1), ["Jan '24", "Feb '24", "Mar '24", "Apr '24", "May '24", "Jun '24", "Jul '24", "Aug '24", "Sep '24", "Oct '24", "Nov '24", "Dec '24", "Jan '25", "Feb '25", "Mar '25", "Apr '25", "May '25", "Jun '25", "Jul '25", "Aug '25", "Sep '25", "Oct '25", "Nov '25", "Dec '25"], fontsize=5)
plt.yticks([0, 1], ["No", "Yes"])
plt.grid(True)
plt.legend(loc="center left", fontsize=8)
mplcyberpunk.add_glow_effects()
plt.show()


# L1 and L2 reg



# SVM
# Mark and Aurora's relationship
# Mark's daily frequency of intermingling with Aurora during 2023, daily
x = np.array([[i] for i in range(1, 366)]) # 1 to 365
y = np.random.randint(0, 2, (365, 1)) # 0 or 1 for each day

class SVM: # support vector machine
    def __init__(self, epochs, eta, lambda_): # lambda is a reserved keyword so I used lambda_
        self.epochs = epochs
        self.eta = eta
        self.lambda_ = lambda_
        self.b = 0

    def hinge(self, y, ypred):
        return max(0, 1 - y * ypred)

    def rbf(self, x, y, gamma=0.1):
        return np.exp(-gamma * np.linalg.norm(x - y) ** 2)

    def fit(self, x, y): # train
        obsCount, featureCount = x.shape # 365, 1 (one feature, which is the day (not ideal, but it's just an example. in reality, you'd have more features))

        y_ = np.where(y <= 0, -1, 1) # convert 0 to -1

        alphas = np.zeros(obsCount) # 365

        for epoch in range(self.epochs):
            for i in range(obsCount):
                x_i, y_i = x[i], y_[i] # get the i-th (current) observation

                decision = sum(alphas[j] * y_[j] * self.rbf(x_i, x[j]) for j in range(obsCount)) - self.b # decision boundary

                if y_i * decision < 1:
                    alphas[i] += self.eta * (1 - y_i * decision)
                    self.b += self.eta * y_i
                else:
                    alphas[i] -= self.eta * 2 * self.lambda_ * alphas[i]

            print(f"epoch {epoch + 1}: b = {self.b}")

            self.supportVectors = x[alphas > 0]
            self.supportAlphas = alphas[alphas > 0]
            self.supportLabels = y_[alphas > 0]

    def predict(self, x):
        decision = sum(self.supportAlphas[i] * self.supportLabels[i] * self.rbf(x, self.supportVectors[i]) for i in range(len(self.supportVectors))) - self.b
        return np.sign(decision)


svm = SVM(epochs=10, eta=0.01, lambda_=0.01)
svm.fit(x, y)

xPred = np.array([[i] for i in range(366, 731)]) # (365, 1)
yPred = np.array([svm.predict(np.array([x])) for x in xPred]) # (365, 1)
yPred = np.where(yPred < 0, 0, 1) # convert -1 to 0

deltaIdx = np.where(np.diff(yPred.ravel()) != 0)[0]
deltas = xPred[deltaIdx + 1]

plt.style.use("cyberpunk")
plt.figure(figsize=(12, 8))

plt.scatter(x, y, color="yellow", label="Actual intermingling")
plt.scatter(xPred, yPred, color="magenta", label="Predicted intermingling")
plt.scatter(svm.supportVectors, np.where(svm.supportLabels < 0, 0, 1), color="red", label="Support vectors", edgecolors="black", s=100)
for day in deltas:
    plt.axvline(day, color="cyan", linestyle="dashed", label="Change in intermingling" if day == deltas[0] else "")
plt.title("Mark and Aurora's intermingling", fontsize=16)
plt.xlabel("Day", fontsize=14)
plt.ylabel("Intermingling (bool)", fontsize=14)
plt.legend(loc="center left", fontsize=8)
plt.ylim(-0.1, 1.1)
plt.show()


# 