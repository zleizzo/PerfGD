import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import sem
from collections import deque
from sklearn.datasets import make_spd_matrix
import time
import csv
import sys

matplotlib.rcParams['pdf.fonttype'] = 42

np.random.seed(0)

R = 5
# Problem instance constants.
d = 5
sigma = 1
Sigma = (sigma ** 2) * np.eye(d)

mu0 = np.random.rand(d) + 6

eps = 1.5


opt_theta = mu0 / (2 * eps)
fixed_theta = mu0 / eps


def clip_coord(z):
    if z > R:
        return R
    elif z < 0:
        return 0
    else:
        return z


def clip(theta):
    return np.array([clip_coord(z) for z in theta])


def mu(theta):
    return mu0 - eps * theta


def shift_dist(n, theta):
    X = np.random.multivariate_normal(mu(theta), Sigma, n)
    return X


def loss(x, theta):
    return np.dot(x, theta)


def performative_loss(theta):
    return np.dot(theta, mu0 - eps * theta)

opt_loss = performative_loss(opt_theta)
fixed_loss = performative_loss(fixed_theta)


def grad1(X, theta):
    return np.mean(X, axis = 0)


def RGD(n, theta0, lr, tol, max_iter):
    history = deque()
    
    theta = theta0
    X = shift_dist(n, theta)
    history.append(theta)
    
    for t in range(max_iter - 1):
        grad = grad1(X, theta)
        if np.linalg.norm(grad) < tol:
            print(f'RGD converged in {t + 1} iterations.')
            return history
        else:
            theta = clip(theta + lr * grad)
            history.append(theta)
            
            X = shift_dist(n, theta)
    
    print(f'RGD failed to converge in {max_iter} iterations.')
    return history


def flaxman(n, theta0, lr, delta, max_iter):
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
        
        X = shift_dist(n, perturbed_theta)
        loss_estimate = sum(X @ perturbed_theta) / n
        grad_estimate = (d * loss_estimate / delta) * u
        
        theta = clip(theta + lr * grad_estimate).copy()
    
    return history, queries


def approx_mu(X):
    return np.mean(X, axis = 0)


def approx_grad_mu(mus, thetas):
    dmus = np.array([m - mus[-1] for m in mus]).T
    dthetas = np.array([t - thetas[-1] for t in thetas], dtype = float).T
    
    return dmus @ np.linalg.pinv(dthetas)


def grad2(X, mus, thetas):
    """
    X, Y should be the data resulting from thetas[-1]
    """
    theta = thetas[-1]
    grad_mu = approx_grad_mu(mus, thetas)
    
    mean = approx_mu(X)
    return grad_mu.T @ np.mean(np.array([loss(x, theta) * (x - mean) for x in X]), axis = 0)


def PGD(n, theta0, H, lr, tol, max_iter, init = d, verbose = False):
    print('Starting PGD.')
    history    = deque()
    grad_betas = deque()
    min_sing_vals = deque()
    
    thetas = deque(maxlen = H + 1)
    mus    = deque(maxlen = H + 1)
    
    start = time.time()
    
    theta = theta0.copy()
    X = shift_dist(n, theta)
    
    history.append(theta)
    thetas.append(theta)
    mus.append(approx_mu(X))
    
    for t in range(init - 1):
        grad = grad1(X, theta)
        theta = clip(theta + lr * grad)
        X = shift_dist(n, theta)
        
        history.append(theta)
        thetas.append(theta)
        mus.append(approx_mu(X))
    
    for t in range(max_iter - init):
        if t % 100 == 0:
            print(f'ETA: {round((time.time() - start) * (max_iter - t - 2) / (60 * (t + 2)), 1)} min')
        
        g2 = grad2(X, mus, thetas)
        
        if verbose:
            dthetas = np.array([t - thetas[-1] for t in thetas], dtype = float).T
            _, sing_vals, _ = np.linalg.svd(dthetas)
            min_sing_vals.append(sing_vals[-1])
        
        grad = grad1(X, theta) + g2
        
        if np.linalg.norm(grad) < tol:
            print(f'PGD converged in {t + 1} iterations.')
            return history
        else:
            theta = clip(theta + lr * grad)
            X = shift_dist(n, theta)
            
            history.append(theta)
            thetas.append(theta)
            mus.append(approx_mu(X))
    
    print(f'PGD failed to converge in {max_iter} iterations.')
    return history, min_sing_vals#, betas, g2s, grad_betas



runs = 10 #10
n  = 500 #1000
H  = 25 #500
# H = d
lr = 0.1 #0.1
delta = 4.75 #4.75  #Best so far: 5, theta0 = 1 + rand [3, 4, 5, 6] --> [4.5, 4.75]
tol = 0
max_iter = 30
init = d
test_sing_vals = False


rgd = [None for _ in range(max_iter)]
pgd = [None for _ in range(max_iter)]
flx = [None for _ in range(max_iter)]
flx_quer = [None for _ in range(max_iter)]

min_sing_vals = np.zeros((runs, max_iter - init))

rgd_loss = np.zeros((runs, max_iter))
pgd_loss = np.zeros((runs, max_iter))
flx_loss = np.zeros((runs, max_iter))
flx_quer_loss = np.zeros((runs, max_iter))
# rrm_loss = np.zeros((runs, max_iter))

rgd_dist = np.zeros((runs, max_iter))
pgd_dist = np.zeros((runs, max_iter))
flx_dist = np.zeros((runs, max_iter))
flx_quer_dist = np.zeros((runs, max_iter))

for r in range(runs):
    print(f'Running experiment {r + 1}.')
    np.random.seed(r)
    
    theta0 = (opt_theta + fixed_theta) / 2 + np.random.randn(d)
    # theta0 = 1 + np.random.rand(d)
    
    rgd[r] = RGD(n, theta0, lr, tol, max_iter)
    if test_sing_vals:
        pgd[r], min_sing_vals[r] = PGD(n, theta0, H, lr, tol, max_iter, init = init, verbose = True)
    else:
        pgd[r], _ = PGD(n, theta0, H, lr, tol, max_iter, init = init)
    flx[r], flx_quer[r] = flaxman(n, theta0, lr, delta, max_iter)
    # rrm[r, :] = RRM(n, theta0, tol, max_iter)

for r in range(runs):
    rgd_loss[r, :] = np.array([performative_loss(t) for t in rgd[r]])
    pgd_loss[r, :] = np.array([performative_loss(t) for t in pgd[r]])
    flx_loss[r, :] = np.array([performative_loss(t) for t in flx[r]])
    flx_quer_loss[r, :] = np.array([performative_loss(t) for t in flx_quer[r]])
    
    rgd_dist[r, :] = np.array([np.linalg.norm(t - opt_theta) for t in rgd[r]])
    pgd_dist[r, :] = np.array([np.linalg.norm(t - opt_theta) for t in pgd[r]])
    flx_dist[r, :] = np.array([np.linalg.norm(t - opt_theta) for t in flx[r]])
    flx_quer_dist[r, :] = np.array([np.linalg.norm(t - opt_theta) for t in flx_quer[r]])
    


rgd_loss_mean = np.mean(rgd_loss, axis = 0)
rgd_loss_sem  = sem(rgd_loss, axis = 0)

pgd_loss_mean = np.mean(pgd_loss, axis = 0)
pgd_loss_sem  = sem(pgd_loss, axis = 0)

flx_loss_mean = np.mean(flx_loss, axis = 0)
flx_loss_sem  = sem(flx_loss, axis = 0)

flx_quer_loss_mean = np.mean(flx_quer_loss, axis = 0)
flx_quer_loss_sem  = sem(flx_quer_loss, axis = 0)



rgd_dist_mean = np.mean(rgd_dist, axis = 0)
rgd_dist_sem  = sem(rgd_dist, axis = 0)

pgd_dist_mean = np.mean(pgd_dist, axis = 0)
pgd_dist_sem  = sem(pgd_dist, axis = 0)

flx_dist_mean = np.mean(flx_dist, axis = 0)
flx_dist_sem  = sem(flx_dist, axis = 0)

flx_quer_dist_mean = np.mean(flx_quer_dist, axis = 0)
flx_quer_dist_sem  = sem(flx_quer_dist, axis = 0)

# rrm_mean = np.mean(rrm, axis = 0)
# rrm_sem  = sem(rrm, axis = 0)

# rrm_loss_mean = np.mean(rrm_loss, axis = 0)
# rrm_loss_sem  = sem(rrm_loss, axis = 0)

num_plots = 2

plt.style.use('ggplot')
plt.rc('font', size = 25)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

x = range(max_iter)


ratio = 0.5

plt.figure()
ax = plt.subplot(1, num_plots, 1)
ax.set_ylim(-0.2, 6.5)
plt.plot(np.zeros(max_iter), label = 'OPT')
plt.plot(np.linalg.norm(opt_theta - fixed_theta) * np.ones(max_iter), label = 'STAB')

plot = plt.plot(rgd_dist_mean, label = 'RGD', c = colors[2])
plt.fill_between(x, rgd_dist_mean - rgd_dist_sem, rgd_dist_mean + rgd_dist_sem, color = plot[0].get_color(), alpha = 0.3)

plot = plt.plot(flx_dist_mean, label = 'DFO', c = colors[3])
plt.fill_between(x, flx_dist_mean - flx_dist_sem, flx_dist_mean + flx_dist_sem, color = plot[0].get_color(), alpha = 0.3)

plot = plt.plot(flx_quer_dist_mean, label = 'DFO queries', c = colors[5])
plt.fill_between(x, flx_quer_dist_mean - flx_quer_dist_sem, flx_quer_dist_mean + flx_quer_dist_sem, color = plot[0].get_color(), alpha = 0.3)

plot = plt.plot(pgd_dist_mean, label = 'PerfGD', c = colors[4])
plt.fill_between(x, pgd_dist_mean - pgd_dist_sem, pgd_dist_mean + pgd_dist_sem, color = plot[0].get_color(), alpha = 0.3)


# plot = plt.plot(rrm_mean, label = 'RRM')
# plt.fill_between(x, rrm_mean - rrm_sem, rrm_mean + rrm_sem, color = plot[0].get_color(), alpha = 0.3)


plt.xlabel('Training iteration')
plt.ylabel('Distance to OPT')
leg = plt.legend()
leg.set_draggable(state = True)

xleft, xright = ax.get_xlim()
ybottom, ytop = ax.get_ylim()
ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)


ax = plt.subplot(1, num_plots, 2)
plot = plt.plot(opt_loss * np.ones(len(rgd_loss_mean)), label = 'OPT')
plot = plt.plot(fixed_loss * np.ones(len(rgd_loss_mean)), label = 'STAB')

plot = plt.plot(rgd_loss_mean, label = 'RGD', c = colors[2])
plt.fill_between(x, rgd_loss_mean - rgd_loss_sem, rgd_loss_mean + rgd_loss_sem, color = plot[0].get_color(), alpha = 0.3)

plot = plt.plot(flx_loss_mean, label = 'FLX', c = colors[3])
plt.fill_between(x, flx_loss_mean - flx_loss_sem, flx_loss_mean + flx_loss_sem, color = plot[0].get_color(), alpha = 0.3)

plot = plt.plot(flx_quer_loss_mean, label = 'FLX queries', c = colors[5])
plt.fill_between(x, flx_quer_loss_mean - flx_quer_loss_sem, flx_quer_loss_mean + flx_quer_loss_sem, color = plot[0].get_color(), alpha = 0.3)

plot = plt.plot(pgd_loss_mean, label = 'PerfGD', c = colors[4])
plt.fill_between(x, pgd_loss_mean - pgd_loss_sem, pgd_loss_mean + pgd_loss_sem, color = plot[0].get_color(), alpha = 0.3)

# plot = plt.plot(rrm_loss_mean, label = 'RRM')
# plt.fill_between(x, rrm_loss_mean - rrm_loss_sem, rrm_loss_mean + rrm_loss_sem, color = plot[0].get_color(), alpha = 0.3)



plt.xlabel('Training iteration')
plt.ylabel('Performative revenue')
leg = plt.legend()
leg.set_draggable(state = True)

xleft, xright = ax.get_xlim()
ybottom, ytop = ax.get_ylim()
ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)


if test_sing_vals:
    plt.figure()
    x = range(d, max_iter)
    min_sing_vals_mean = np.mean(min_sing_vals, axis = 0)
    min_sing_vals_sem  = sem(min_sing_vals, axis = 0)
    
    ax = plt.subplot(1, 1, 1)
    plot = plt.plot(x, min_sing_vals_mean, label = 'd-th singular value')
    plt.fill_between(x, min_sing_vals_mean - min_sing_vals_sem, min_sing_vals_mean + min_sing_vals_sem, color = plot[0].get_color(), alpha = 0.3)
    
    plt.xlabel('Training iteration')
    plt.ylabel('d-th singular value')
    plt.legend()
    
    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*0.5)