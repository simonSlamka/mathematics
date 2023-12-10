import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D


# ^ redefining a lot throughout, but that's only for clarity. this is just a training file, not a production one

# vars, funcs, graphs, lims, continuity
def f(x):
    return x**2

def g(x):
    return np.sin(x)

x = np.linspace(-10, 10, 500)
y = f(x)

plt.figure(figsize=(8, 8))
plt.plot(x, y, label="f(x) = x^2")
plt.title("f(x) = x^2")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.legend()

def lim(x):
    return (1/x)

xLim = 0
epsilon = 0.0000001

left = lim(xLim - epsilon)
right = lim(xLim + epsilon)

print(f"left: {left}\nright: {right}")

x = np.linspace(-10, 10, 500)
x = x[np.abs(x) > epsilon]
y = lim(x)

plt.figure(figsize=(8, 8))
plt.plot(x, y, label="lim x->0 1/x")
plt.scatter([xLim - epsilon], [left], color="blue", label="left")
plt.scatter([xLim + epsilon], [right], color="red", label="right")
plt.title("lim x->0 1/x")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.ylim(-50, 50)
plt.grid()
plt.legend()


# derivatives
def df(x):
    return 2*x

def dfg(x):
    return np.cos(x)

a = 69

dt = df(a)
print(f"df: {dt}")

x = np.linspace(-10, 10, 500)
y = f(x)
yDt = df(x)

x2 = np.linspace(-10, 10, 500)
y2 = g(x2)
yDt2 = dfg(x2)

plt.figure(figsize=(8, 8))
plt.plot(x, y, label="f(x) = x^2")
plt.plot(x, yDt, label="df(x) = 2x")
plt.plot(x2, y2, color="red", label="g(x) = sin(x)")
plt.plot(x2, yDt2, color="blue", label="dg(x) = cos(x)")
plt.title("f(x) = x^2 and g(x) = sin(x)")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.legend()


t = 2
hAt2Sec = -5 * t ** 2 + 20 * t
print(f"height at 2 seconds: {hAt2Sec}")

def h(t):
    return -5 * t ** 2 + 20 * t

def v(t): # velocity, which is the derivative of h(t)
    return -10 * t + 20

def a(t): # acceleration, which is the derivative of v(t)
    return np.full_like(t, -10) # g = 9.8 m/s^2, so this actually works

t = np.linspace(0, 4, 500)

hs = h(t)
vs = v(t)
as_ = a(t)

plt.figure(figsize=(12, 12))
plt.subplot(311)
plt.plot(t, hs, label="h(t) = -5t^2 + 20t")
plt.title("h(t) = -5t^2 + 20t")
plt.xlabel("t")
plt.ylabel("h")
plt.grid()
plt.legend()

plt.subplot(312)
plt.plot(t, vs, label="v(t) = -10t + 20")
plt.title("v(t) = -10t + 20")
plt.xlabel("t")
plt.ylabel("v")
plt.grid()
plt.legend()

# plt.subplot(313)
# plt.plot(t, as_, label="a(t) = -10")
# plt.title("a(t) = -10")
# plt.xlabel("t")
# plt.ylabel("a")
# plt.grid()
# plt.legend()

plt.tight_layout()

fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 12))

line1, = ax1.plot([], [], label="h(t) = -5t^2 + 20t")
ax1.set_title("h(t) = -5t^2 + 20t")
ax1.set_xlabel("t")
ax1.set_ylabel("h")
ax1.grid()
ax1.legend()

line2, = ax2.plot([], [], label="v(t) = -10t + 20")
ax2.set_title("v(t) = -10t + 20")
ax2.set_xlabel("t")
ax2.set_ylabel("v")
ax2.grid()
ax2.legend()

def init():
    ax1.set_xlim(0, 4)
    ax1.set_ylim(0, 25)
    line1, = ax1.plot([], [], "b-", label="h(t) = -5t^2 + 20t")

    ax2.set_xlim(0, 4)
    ax2.set_ylim(-40, 25)
    line2, = ax2.plot([], [], "r-", label="v(t) = -10t + 20")
    return line1, line2,

def update(frame):
    t = np.linspace(0, frame, 500)
    hs = h(t)
    vs = v(t)
    line1.set_data(t, hs)
    line2.set_data(t, vs)
    return line1, line2,

ani = FuncAnimation(fig, update, frames=np.linspace(0, 4, 50), init_func=init, blit=True)
plt.show()

def x(t):
    return t ** 2
def y(t):
    return 2 * t
def z(t):
    return t ** 3 - 3 * t

def pos(t):
    return (x(t), y(t), z(t))

def dpos(t):
    return (2 * t, 2, 3 * t ** 2 - 3)

def isInIllegalArea(x, y, z):
    return (x < 0 or x > 50) or (y < 0 or y > 30) or (z < 0 or z > 70)

def isGoingIntoIllegalArea(position, velocity):
    x, y, z = position
    dx, dy, dz = velocity

    return ((x < 0 and dx < 0) or (x > 50 and dx > 0)) or ((y < 0 and dy < 0) or (y > 30 and dy > 0)) or ((z < 0 and dz < 0) or (z > 70 and dz > 0))

def k(x, y, z):
    return x * y + y * z + z * x

def dfdx(x, y, z):
    return y + z

def dfdy(x, y, z):
    return x + z

def dfdz(x, y, z):
    return x + y

t = np.linspace(-10, 10, 50)
positions = np.array([pos(ti) for ti in t])
velocities = np.array([dpos(ti) for ti in t])

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], label="Drone path")

for i in range(0, len(t), 5):
    if isInIllegalArea(*positions[i]):
        ax.scatter(*positions[i], color="red")
    elif isGoingIntoIllegalArea(positions[i], velocities[i]):
        ax.scatter(*positions[i], color="yellow")
    else:
        ax.scatter(*positions[i], color="blue")

    ax.quiver(*positions[i], *velocities[i], color="magenta")

ax.quiver(0, 0, 0, 0, 0, 0, color="red", label="Illegal area")
ax.quiver(0, 0, 0, 0, 0, 0, color="blue", label="Legal area")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.legend()

x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
x, y = np.meshgrid(x, y)

z = np.linspace(-10, 10, 5)

fig = plt.figure(figsize=(12, 8))
fig.suptitle("k(x, y, z) = xy + yz + zx\ndf/dx = y + z\ndf/dy = x + z\ndf/dz = x + y")

for i, z in enumerate(z, 1):
    ax = fig.add_subplot(1, 5, i, projection="3d")

    ks = k(x, y, z)

    norm = plt.Normalize(ks.min(), ks.max())
    colors = plt.cm.jet(norm(ks))

    surf = ax.plot_surface(x, y, ks, facecolors=colors, rcount=100, ccount=100, shade=False)
    surf.set_facecolor((0, 0, 0, 0))

    ax.set_title(f"z = {z}")


# integrals
def f(x):
    return x**2

def intf(x):
    return (1/3) * x**3 + 1

x = np.linspace(-10, 10, 500)
y = f(x)

fig, ax = plt.subplots()
ax.plot(x, y, label="f(x) = x^2", linewidth=3)

xFill = np.linspace(-10, 10, 500)
yFill = f(xFill)
ax.fill_between(xFill, yFill, color="red", alpha=0.5, label="Area under curve")

ax.set_title("Area under curve")
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.grid()
ax.legend()


# chain rule
def h(x):
    return x**2

def g(x):
    return np.sin(x)

def f(x):
    return h(g(x))

def df(x):
    return 2 * np.sin(x) * np.cos(x)

x = np.linspace(-10, 10, 500)
y = f(x)
yDt = df(x)

plt.figure(figsize=(8, 8))
plt.plot(x, y, label="f(x) = x^2")
plt.plot(x, yDt, label="df(x) = 2sin(x)cos(x)")
plt.title("f(x) = (sin(x))^2 and derivative")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.legend()


# gradient descent
def f(x):
    return x**2

def df(x):
    return 2 * x

def gradient_descent(x, eta, epochs):
    xs = []
    for i in range(epochs):
        grad = df(x)
        x = x - eta * grad
        print(f"epoch {i + 1}: x = {x}")
        xs.append(x)
    return x, xs

x = 100
eta = 0.0001
epochs = 100000

minX, xs = gradient_descent(x, eta, epochs)
print(f"min x: {minX}")

x = np.linspace(-10, 10, 500)
y = f(x)

fig, axs = plt.subplots(2, figsize=(12, 12))  # Create two subplots

axs[0].plot(xs, label="x")
axs[0].set_title(f"gradient descent - eta: {eta}, epochs: {epochs}, start x: 100, min x: {minX}")
axs[0].set_xlabel("epoch")
axs[0].set_ylabel("x")
axs[0].grid()
axs[0].legend()

axs[1].plot(x, y, "ro", label="f(x) = x^2")
axs[1].set_title("min x")
axs[1].set_xlabel("epoch")
axs[1].set_ylabel("x")
axs[1].grid()
axs[1].legend()


# backprop
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

def mse(y, yPred):
    return ((y - yPred) ** 2).mean()

def dmse(y, yPred):
    return 2 * (yPred - y) / y.size

# Mark's monthly savings
x = np.array([[1], [2], [3], [4], [5], [6], [7], [8]]) # Jan to Aug
y = np.array([[124], [164], [852], [593], [921], [109], [102], [123]]) # savings in caps (lol)
y = np.log(y) # log to make it easier to work with

inputLayerSize = x.shape[1] # one neuron
hiddenLayerSize = 2 # two neurons
outputLayerSize = y.shape[1] # one neuron

np.random.seed(69)
weights1 = np.random.randn(inputLayerSize, hiddenLayerSize) # (1, 3)
weights2 = np.random.randn(hiddenLayerSize, outputLayerSize) # (3, 1)

eta = 0.01
epochs = 1000000

for epoch in range(epochs):
    # fwd
    hiddenLayer = sigmoid(np.dot(x, weights1)) # (8, 3)

    outputLayer = np.dot(hiddenLayer, weights2) # (8, 1)
    yPred = outputLayer

    # loss
    loss = mse(y, yPred)
    if epoch % 10000 == 0:
        print(f"epoch {epoch}: loss = {loss}")

    # bwd
    grad = dmse(y, yPred) # (8, 1)
    gradw2 = np.dot(hiddenLayer.T, grad) # (3, 1)

    grad = np.dot(grad, weights2.T) # (8, 3)
    gradw1 = np.dot(x.T, grad) # (1, 3)

    weights1 -= eta * gradw1
    weights2 -= eta * gradw2

print(f"pred: {yPred}")
print(f"final prediction: {np.exp(yPred)}")

xPred = np.array([[9], [10], [11], [12]]) # Sep to Dec
yPred2 = np.log(np.exp(np.dot(sigmoid(np.dot(xPred, weights1)), weights2)))
print(f"pred: {yPred2}")


plt.figure(figsize=(12, 12))
plt.plot(x, y, "ro", label="Actual savings")
plt.plot(yPred, "b-", label="Predicted savings for Jan to Aug")
plt.scatter(np.arange(9,13,1), yPred2, color="g", label="Predicted savings for Sep to Dec")
plt.title("Mark's monthly savings")
plt.xlabel("Month")
plt.ylabel("Savings (in log scale)")
plt.grid()
plt.legend()
plt.show()