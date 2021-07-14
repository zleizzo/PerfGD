import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import sem
from collections import deque
import time
import csv
import sys

matplotlib.rcParams['pdf.fonttype'] = 42

# Problem instance constants.
mu0 = 1
s0 = 0.5

mu1 = -1
s1 = 0.5
eps = 3.

g = 0.5
reg = 1e-2

R = 10.

def clip_coord(z):
    if z > R:
        return R
    elif z < -R:
        return -R
    else:
        return z


def clip(theta):
    return np.array([clip_coord(z) for z in theta])


def mean0(theta):
    return mu0


def mean1(theta):
    return mu1 - eps * theta[1]


def shift_dist(n, theta):
    X = np.ones((n, 2))
    Y = np.ones(n)
    for i in range(n):
        if np.random.rand() <= g:
            X[i, 1] = s1 * np.random.randn() + mean1(theta)
            Y[i] = 1
        else:
            X[i, 1] = s0 * np.random.randn() + mean0(theta)
            Y[i] = 0
    return X, Y


def h(x, theta):
    """
    Logistic model output on x with params theta.
    x should have a bias term appended in the 0-th coordinate.
    1 / (1 + exp{-x^T theta})
    """
    return 1. / (1. + np.exp(-np.dot(x, theta)))


def loss(x, y, theta, reg = reg):
    """
    Cross entropy loss on (x, y) with params theta.
    x should have a bias term appended in the 0-th coordinate.
    """
    return -y * np.log(h(x, theta)) - (1 - y) * np.log(1 - h(x, theta)) + (reg / 2) * np.linalg.norm(theta) ** 2


def est_performative_loss(N, theta):
    X, Y = shift_dist(N, theta)
    return np.mean([loss(x, y, theta) for x, y in zip(X, Y)])


def ce_grad(x, y, theta, reg = reg):
    return (h(x, theta) - y) * x + reg * theta


def grad1(X, Y, theta):
    n = len(Y)
    grad = np.zeros(2)
    for x, y in zip(X, Y):
        grad += ce_grad(x, y, theta)
    return grad / n


def RGD(n, theta0, lr, tol, max_iter):
    history = deque()
    
    theta = theta0.copy()
    X, Y = shift_dist(n, theta)
    history.append(theta)
    
    for t in range(max_iter - 1):
        grad = grad1(X, Y, theta)
        if np.linalg.norm(grad) < tol:
            print(f'RGD converged in {t + 1} iterations.')
            return history
        else:
            theta = clip(theta - lr * grad).copy()
            history.append(theta)
            
            X, Y = shift_dist(n, theta)
    
    print(f'RGD failed to converge in {max_iter} iterations.')
    return history


def hessian(X, theta, reg):
    """
    Computes the Hessian of the loss on X at theta with ridge regularization reg.
    """
    n = len(X)
    d = len(X[0, :])
    h_vec = np.array([h(x, theta) for x in X])
    w = h_vec * (1 - h_vec)
    
    hess = np.zeros((d, d))
    for i in range(n):
        hess += np.outer(w[i] * X[i], X[i])
    hess += n * reg * np.eye(d)
    return hess


def gradient(X, Y, theta, reg):
    """
    Computes the gradient of the loss on X, Y at theta with ridge regularization reg.
    """
    n = len(Y)
    h_vec = np.array([h(x, theta) for x in X])
    grad = X.T @ (h_vec - Y) + n * reg * theta
    return grad


def fit(X, Y, theta0 = None, reg = reg, lr = 1, tol = 0.001, max_iter = 1000):
    """
    Fits a logistic model to X, Y via Newton's method, without the aid of Pytorch.
    """
    if theta0 is None:
        theta0 = np.random.randn(len(X[0, :]))
        
    theta = theta0.copy()
    grad = gradient(X, Y, theta, reg)
    grad_norm = np.linalg.norm(grad)
    count = 1
    while grad_norm > tol:
        if count % 500 == 0:
            print(f'Iteration {count}: |grad| = {grad_norm}, lr = {lr}')
        hess = hessian(X, theta, reg)
        step = np.linalg.solve(hess, grad)
        theta -= lr * step
        grad = gradient(X, Y, theta, reg)
        old_grad_norm = grad_norm
        grad_norm = np.linalg.norm(grad)
        if grad_norm > old_grad_norm:
            lr *= 0.9
        count += 1
        if count > max_iter:
            print(f'Warning: Optimization failed to converge. Aborting with |grad| = {grad_norm}.')
            break
    return theta


def RRM(n, theta0, tol, max_iter):
    print('Starting RRM.')
    history = deque()
    history.append(theta0.copy())
    
    theta = theta0.copy()
    
    for t in range(max_iter - 1):
        X, Y = shift_dist(n, theta)
        new_theta = clip(fit(X, Y, theta0 = theta))
        
        if np.linalg.norm(theta - new_theta) < tol:
            return history
        
        else:
            history.append(new_theta.copy())
            theta = new_theta.copy()
    
    print(f'RRM failed to converge in {max_iter} iterations.')
    return history


def approx_f(X, Y):
    return np.mean(X[Y == 1][:, 1])


def approx_grad_f(means, thetas):
    dmeans = np.array([m - means[-1] for m in means])
    dthetas = np.array([t - thetas[-1] for t in thetas])
    
    return np.linalg.pinv(dthetas) @ dmeans


def grad2(X, Y, means, thetas, s1):
    """
    X, Y should be the data resulting from thetas[-1]
    """
    n      = len(Y)
    theta  = thetas[-1]
    grad_f = approx_grad_f(means, thetas)
    
    pos_X = X[Y == 1]
    loss_vec  = np.array([loss(x, 1, theta) for x in pos_X])
    
    x_minus_f = np.array([x - means[-1] for x in pos_X[:, 1]])
    
    return (np.dot(loss_vec, x_minus_f) / (n * s1 ** 2)) * grad_f, grad_f


def PGD(n, theta0, H, lr, tol, max_iter):
    print('Starting PGD.')
    history = deque()
    grad_fs = deque()
    g2s     = deque()
    
    thetas = deque(maxlen = H + 1)
    means  = deque(maxlen = H + 1)
    
    start = time.time()
    
    theta = theta0.copy()
    X, Y = shift_dist(n, theta)
    
    history.append(theta.copy())
    thetas.append(theta.copy())
    means.append(approx_f(X, Y))
    
    grad = grad1(X, Y, theta)
    theta = clip(theta - lr * grad).copy()
    X, Y = shift_dist(n, theta)
    
    history.append(theta.copy())
    thetas.append(theta.copy())
    means.append(approx_f(X, Y))
    
    for t in range(max_iter - 2):
        if t % 100 == 0:
            print(f'ETA: {round((time.time() - start) * (max_iter - t - 2) / (60 * (t + 2)), 1)} min')
        
        g2, grad_f = grad2(X, Y, means, thetas, s1)
        grad = grad1(X, Y, theta) + g2
        grad_fs.append(grad_f)
        g2s.append(g2)
        
        if np.linalg.norm(grad) < tol:
            print(f'PGD converged in {t + 1} iterations.')
            return history
        else:
            theta = clip(theta - lr * grad).copy()
            X, Y = shift_dist(n, theta)
            
            history.append(theta.copy())
            thetas.append(theta.copy())
            means.append(approx_f(X, Y))
    
    print(f'PGD failed to converge in {max_iter} iterations.')
    return history, grad_fs, means, g2s


def flaxman(n, theta0, lr, delta, max_iter):
    d = 2
    history = deque()
    queries = deque()
    
    theta = theta0.copy()
    
    for t in range(max_iter):
        # Sample u ~ Unif(sphere)
        u = np.random.randn(d)
        u /= np.linalg.norm(u)
        
        history.append(theta)
        perturbed_theta = (theta + delta * u).copy()
        queries.append(perturbed_theta)
        
        loss_estimate = est_performative_loss(n, perturbed_theta)
        grad_estimate = (d * loss_estimate / delta) * u
        
        theta = clip(theta - lr * grad_estimate).copy()
    
    return history, queries


s = 0
n  = 500
H  = 500
lr = 0.1
tol = 0
max_iter = 100
delta = 1.1
# delta = float(sys.argv[1])

runs = 10

# rgd_hist = np.zeros((runs, max_iter))
# pgd_hist = np.zeros((runs, max_iter))
# rrm_hist = np.zeros((runs, max_iter))
# flx_hist = np.zeros((runs, max_iter))
# flq_hist = np.zeros((runs, max_iter))

rgd_loss = np.zeros((runs, max_iter))
pgd_loss = np.zeros((runs, max_iter))
rrm_loss = np.zeros((runs, max_iter))
flx_loss = np.zeros((runs, max_iter))
flq_loss = np.zeros((runs, max_iter))




for r in range(runs):
    print(f'Starting experiment {r}.')
    np.random.seed(r)
    
    theta0 = 2 * np.random.rand(2) - 1
    
    rgd_hist = RGD(n, theta0, lr, tol, max_iter)
    pgd_hist, _, _, _ = PGD(n, theta0, H, lr, tol, max_iter)
    # rrm_hist = RRM(n, theta0, tol, max_iter)
    # flx_hist, flq_hist = flaxman(n, theta0, lr, delta, max_iter)
    
    rgd_loss[r, :] = np.array([est_performative_loss(100, t) for t in rgd_hist])
    pgd_loss[r, :] = np.array([est_performative_loss(100, t) for t in pgd_hist])
    # rrm_loss[r, :] = np.array([est_performative_loss(100, t) for t in rrm_hist])
    # flx_loss[r, :] = np.array([est_performative_loss(100, t) for t in flx_hist])
    # flq_loss[r, :] = np.array([est_performative_loss(100, t) for t in flq_hist])

plt.style.use('ggplot')
plt.rc('font', size = 20)

rgd_loss_mean = np.mean(rgd_loss, axis = 0)
rgd_loss_sem  = sem(rgd_loss, axis = 0)

rrm_loss_mean = np.mean(rrm_loss, axis = 0)
rrm_loss_sem  = sem(rrm_loss, axis = 0)

pgd_loss_mean = np.mean(pgd_loss, axis = 0)
pgd_loss_sem  = sem(pgd_loss, axis = 0)

flx_loss_mean = np.mean(flx_loss, axis = 0)
flx_loss_sem  = sem(flx_loss, axis = 0)

flq_loss_mean = np.mean(flq_loss, axis = 0)
flq_loss_sem  = sem(flq_loss, axis = 0)


colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
x = range(max_iter)

plt.figure()
ax = plt.subplot(2, 1, 1)
plot = plt.plot(x, rgd_loss_mean, label = 'RGD', c=colors[2])
plt.fill_between(x, rgd_loss_mean + rgd_loss_sem, rgd_loss_mean - rgd_loss_sem, color = plot[0].get_color(), alpha = 0.3)

# plot = plt.plot(x, rrm_loss_mean, label = 'RRM', c='#777777')
# plt.fill_between(x, rrm_loss_mean + rrm_loss_sem, rrm_loss_mean - rrm_loss_sem, color = plot[0].get_color(), alpha = 0.3)

# plot = plt.plot(x, flx_loss_mean, label = 'FLX', c=colors[3])
# plt.fill_between(x, flx_loss_mean + flx_loss_sem, flx_loss_mean - flx_loss_sem, color = plot[0].get_color(), alpha = 0.3)

# plot = plt.plot(x, flq_loss_mean, label = 'FLX queries', c=colors[5])
# plt.fill_between(x, flq_loss_mean + flq_loss_sem, flq_loss_mean - flq_loss_sem, color = plot[0].get_color(), alpha = 0.3)

plot = plt.plot(x, pgd_loss_mean, label = 'PerfGD', c=colors[4])
plt.fill_between(x, pgd_loss_mean + pgd_loss_sem, pgd_loss_mean - pgd_loss_sem, color = plot[0].get_color(), alpha = 0.3)

# plt.title('Performative loss vs. training iterations')
plt.xlabel('Training iteration')
plt.ylabel('Performative loss')
plt.legend()


ratio = 0.5
xleft, xright = ax.get_xlim()
ybottom, ytop = ax.get_ylim()
ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)


