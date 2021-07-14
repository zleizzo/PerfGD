import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import sem
from collections import deque
import time
import csv
import sys

matplotlib.rcParams['pdf.fonttype'] = 42

R = 10
# Problem instance constants.
sigma = 1.
mu = 1.666666666666667
error_var = 2.033333333333333
a0 = 1.666666666666667
a1 = 1.666666666666667
reg = 3.3333333333333335


loss_reg = reg
fit_reg = 0

c = mu ** 2 + sigma ** 2
opt_theta = c * a0 * (1 - a1) / (c * (1 - a1) ** 2 + loss_reg)
fixed_theta = c * a0 / (c * (1 - a1) + loss_reg)

def clip(theta):
    if theta > R:
        return R
    elif theta < -R:
        return -R
    else:
        return theta


def beta(theta):
    return a0 + a1 * theta


def shift_dist(n, theta):
    b = beta(theta)
    X = sigma * np.random.randn(n) + mu
    Y = b * X + error_var * np.random.randn(n)
    return X, Y


def loss(x, y, theta, reg = loss_reg):
    """
    Squared loss on (x, y) with params theta.
    x should have a bias term appended in the 0-th coordinate.
    """
    return 0.5 * ((theta * x - y) ** 2 + reg * theta ** 2)


def performative_loss(theta, reg = loss_reg):
    return (((theta - beta(theta)) ** 2) * (mu ** 2 + sigma ** 2) + (error_var ** 2) + reg * (theta ** 2)) / 2

opt_loss = performative_loss(opt_theta)
fixed_loss = performative_loss(fixed_theta)

def grad1(X, Y, theta, reg = loss_reg):
    return np.mean([(theta * x - y) * x for x, y in zip(X, Y)]) + reg * theta


def RGD(n, theta0, lr, tol, max_iter):
    history = deque()
    
    theta = theta0
    X, Y = shift_dist(n, theta)
    history.append(theta)
    
    for t in range(max_iter - 1):
        grad = grad1(X, Y, theta)
        if np.linalg.norm(grad) < tol:
            print(f'RGD converged in {t + 1} iterations.')
            return history
        else:
            theta = clip(theta - lr * grad)
            history.append(theta)
            
            X, Y = shift_dist(n, theta)
    
    print(f'RGD failed to converge in {max_iter} iterations.')
    return history


def approx_beta(X, Y, reg = fit_reg):
    mean_xy = np.mean(X * Y)
    mean_xx = np.mean(X ** 2)
    return mean_xy / (mean_xx + reg)


def RRM(n, theta0, tol, max_iter):
    history = deque()
    
    theta = theta0
    history.append(theta)
    
    for t in range(max_iter - 1):
        X, Y = shift_dist(n, theta)
        new_theta = clip(approx_beta(X, Y, reg = loss_reg))
        if abs(new_theta - theta) < tol:
            return history
        else:
            history.append(new_theta)
            theta = new_theta
    
    print(f'RRM failed to converge in {max_iter} iterations.')
    return history


def approx_grad_beta(betas, thetas):
    dbetas = np.array([b - betas[-1] for b in betas])
    dthetas = np.array([t - thetas[-1] for t in thetas], dtype = float)
    
    return np.linalg.pinv(dthetas.reshape(-1, 1)) @ dbetas


def grad2(X, Y, betas, thetas):
    """
    X, Y should be the data resulting from thetas[-1]
    """
    theta = thetas[-1]
    grad_beta = approx_grad_beta(betas, thetas)
    
    return -grad_beta * np.mean([(theta * x - y) * x for x, y in zip(X, Y)]), grad_beta


def PGD(n, theta0, H, lr, tol, max_iter):
    print('Starting PGD.')
    history    = deque()
    
    thetas = deque(maxlen = H + 1)
    betas  = deque(maxlen = H + 1)
    
    start = time.time()
    
    theta = theta0
    X, Y = shift_dist(n, theta)
    
    history.append(theta)
    thetas.append(theta)
    betas.append(approx_beta(X, Y))
    
    grad = grad1(X, Y, theta)
    theta = clip(theta - lr * grad)
    X, Y = shift_dist(n, theta)
    
    history.append(theta)
    thetas.append(theta)
    betas.append(approx_beta(X, Y))
    
    for t in range(max_iter - 2):
        if t % 100 == 0:
            print(f'ETA: {round((time.time() - start) * (max_iter - t - 2) / (60 * (t + 2)), 1)} min')
        
        g2, grad_beta = grad2(X, Y, betas, thetas)
        grad = grad1(X, Y, theta) + g2
        
        if np.linalg.norm(grad) < tol:
            print(f'PGD converged in {t + 1} iterations.')
            return history
        else:
            theta = clip(theta - lr * grad)
            X, Y = shift_dist(n, theta)
            
            history.append(theta)
            thetas.append(theta)
            betas.append(approx_beta(X, Y))
    
    print(f'PGD failed to converge in {max_iter} iterations.')
    return history


def flaxman(n, theta0, lr, delta, max_iter):
    d = 1
    history = deque()
    queries = deque()
    
    theta = theta0
    
    for t in range(max_iter):
        # Sample u ~ Unif(sphere)
        u = np.random.randn(d)
        u /= np.linalg.norm(u)
        
        history.append(theta)
        perturbed_theta = np.array(theta + delta * u).reshape((1, 1))
        queries.append(perturbed_theta)
        
        X, Y = shift_dist(n, perturbed_theta)
        X = X.reshape((-1, 1))
        Y = Y.reshape((-1, 1))
        loss_estimate = 0.5 * (sum((X @ perturbed_theta - Y) ** 2) / n + loss_reg * theta ** 2)
        grad_estimate = (d * loss_estimate / delta) * u
        grad_estimate = grad_estimate[0]
        
        theta = clip(theta - lr * grad_estimate)
    
    return history, queries


runs = 10
n  = 500
H  = 500
lr = 0.1
tol = 0
max_iter = 40
delta = 1.5

rgd = np.zeros((runs, max_iter))
pgd = np.zeros((runs, max_iter))
rrm = np.zeros((runs, max_iter))
flx = np.zeros((runs, max_iter))
flq = np.zeros((runs, max_iter))

rgd_loss = np.zeros((runs, max_iter))
pgd_loss = np.zeros((runs, max_iter))
rrm_loss = np.zeros((runs, max_iter))
flx_loss = np.zeros((runs, max_iter))
flq_loss = np.zeros((runs, max_iter))
for r in range(runs):
    print(f'Running experiment {r + 1}.')
    np.random.seed(r)
    
    theta0 = 2 * np.random.rand() + 2
    
    rgd[r, :] = RGD(n, theta0, lr, tol, max_iter)
    pgd[r, :] = PGD(n, theta0, H, lr, tol, max_iter)
    rrm[r, :] = RRM(n, theta0, tol, max_iter)
    flx[r, :], flq[r, :] = flaxman(n, theta0, lr, delta, max_iter)
    
    rgd_loss[r, :] = np.array([performative_loss(t) for t in rgd[r, :]])
    pgd_loss[r, :] = np.array([performative_loss(t) for t in pgd[r, :]])
    rrm_loss[r, :] = np.array([performative_loss(t) for t in rrm[r, :]])
    flx_loss[r, :] = np.array([performative_loss(t) for t in flx[r, :]])
    flq_loss[r, :] = np.array([performative_loss(t) for t in flq[r, :]])


rgd_mean = np.mean(rgd, axis = 0)
rgd_sem  = sem(rgd, axis = 0)

rgd_loss_mean = np.mean(rgd_loss, axis = 0)
rgd_loss_sem  = sem(rgd_loss, axis = 0)

pgd_mean = np.mean(pgd, axis = 0)
pgd_sem  = sem(pgd, axis = 0)

flx_mean = np.mean(flx, axis = 0)
flx_sem  = sem(flx, axis = 0)

flq_mean = np.mean(flq, axis = 0)
flq_sem  = sem(flq, axis = 0)

pgd_loss_mean = np.mean(pgd_loss, axis = 0)
pgd_loss_sem  = sem(pgd_loss, axis = 0)

rrm_mean = np.mean(rrm, axis = 0)
rrm_sem  = sem(rrm, axis = 0)

rrm_loss_mean = np.mean(rrm_loss, axis = 0)
rrm_loss_sem  = sem(rrm_loss, axis = 0)

flx_loss_mean = np.mean(flx_loss, axis = 0)
flx_loss_sem  = sem(flx_loss, axis = 0)

flq_loss_mean = np.mean(flq_loss, axis = 0)
flq_loss_sem  = sem(flq_loss, axis = 0)


plt.style.use('ggplot')
plt.rc('font', size = 20)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

x = range(max_iter)


ratio = 1.

plt.figure()
ax = plt.subplot(1, 2, 1)
plt.plot(opt_theta * np.ones(len(rgd_mean)), label = 'OPT')
plt.plot(fixed_theta * np.ones(len(rgd_mean)), label = 'STAB')

plot = plt.plot(rgd_mean, label = 'RGD', c = colors[2])
plt.fill_between(x, rgd_mean - rgd_sem, rgd_mean + rgd_sem, color = plot[0].get_color(), alpha = 0.3)

# plot = plt.plot(rrm_mean, label = 'RRM')
# plt.fill_between(x, rrm_mean - rrm_sem, rrm_mean + rrm_sem, color = plot[0].get_color(), alpha = 0.3)

plot = plt.plot(flx_mean, label = 'FLX', c = colors[3])
plt.fill_between(x, flx_mean - flx_sem, flx_mean + flx_sem, color = plot[0].get_color(), alpha = 0.3)

plot = plt.plot(flq_mean, label = 'FLX queries', c = colors[5])
plt.fill_between(x, flq_mean - flq_sem, flq_mean + flq_sem, color = plot[0].get_color(), alpha = 0.3)

plot = plt.plot(pgd_mean, label = 'PerfGD', c = colors[4])
plt.fill_between(x, pgd_mean - pgd_sem, pgd_mean + pgd_sem, color = plot[0].get_color(), alpha = 0.3)

plt.xlabel('Training iteration')
plt.ylabel('θ')
leg = plt.legend()
leg.set_draggable(state = True)

xleft, xright = ax.get_xlim()
ybottom, ytop = ax.get_ylim()
ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)


ax = plt.subplot(1, 2, 2)
plot = plt.plot(opt_loss * np.ones(len(rgd_loss_mean)), label = 'OPT')

plot = plt.plot(fixed_loss * np.ones(len(rgd_loss_mean)), label = 'STAB')

plot = plt.plot(rgd_loss_mean, label = 'RGD', c = colors[2])
plt.fill_between(x, rgd_loss_mean - rgd_loss_sem, rgd_loss_mean + rgd_loss_sem, color = plot[0].get_color(), alpha = 0.3)

# plot = plt.plot(rrm_loss_mean, label = 'RRM')
# plt.fill_between(x, rrm_loss_mean - rrm_loss_sem, rrm_loss_mean + rrm_loss_sem, color = plot[0].get_color(), alpha = 0.3)
# print(plot[0].get_color())

plot = plt.plot(flx_loss_mean, label = 'FLX', c = colors[3])
plt.fill_between(x, flx_loss_mean - flx_loss_sem, flx_loss_mean + flx_loss_sem, color = plot[0].get_color(), alpha = 0.3)

plot = plt.plot(flq_loss_mean, label = 'FLX queries', c = colors[5])
plt.fill_between(x, flq_loss_mean - flq_loss_sem, flq_loss_mean + flq_loss_sem, color = plot[0].get_color(), alpha = 0.3)

plot = plt.plot(pgd_loss_mean, label = 'PerfGD', c = colors[4])
plt.fill_between(x, pgd_loss_mean - pgd_loss_sem, pgd_loss_mean + pgd_loss_sem, color = plot[0].get_color(), alpha = 0.3)

plt.xlabel('Training iteration')
plt.ylabel('Performative loss')
plt.legend()

xleft, xright = ax.get_xlim()
ybottom, ytop = ax.get_ylim()
ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)