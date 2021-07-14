import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
from collections import deque

d = 1

a1 = 1.
a0 = 1.


opt = -(2. / 3.) * (a0 / a1)
stab = -a0 / a1

opt_loss = np.sqrt(a1 * opt + a0) * opt
stab_loss = np.sqrt(a1 * stab + a0) * opt

def clip(theta):
    if theta > 1.:
        return 1.
    elif theta < -1:
        return -1.
    else:
        return theta



beta = 0
def shift_dist(Y0, theta):
    n = len(Y0)
    return np.random.randn(n) + np.sqrt(a1 * theta + a0)


def loss(y, theta):
    return y * theta


def grad(y, theta):
    return y


def grad1(cur_Y, cur_theta):
    return np.mean(cur_Y)


def RGD(Y0, theta0, lr, tol, max_iter):
    history = deque()
    
    cur_theta = theta0
    cur_Y = shift_dist(Y0, cur_theta).copy()
    history.append(cur_theta)
    
    for t in range(max_iter - 1):
        grad = grad1(cur_Y, cur_theta)
        if np.abs(grad) < tol:
            print(f'RGD converged in {t + 1} iterations.')
            return history
        else:
            cur_theta = clip(cur_theta - lr * grad)
            history.append(cur_theta)
            
            cur_Y = shift_dist(Y0, cur_theta)
    
    print(f'RGD failed to converge in {max_iter} iterations.')
    return history


def RRM(Y0, theta0, lr, tol, max_iter):
    history = deque()
    
    prev_theta = theta0
    prev_Y = shift_dist(Y0, prev_theta).copy()
    history.append(prev_theta)
    
    cur_theta = theta0 - lr * grad1(prev_Y, prev_theta)
    cur_Y = shift_dist(Y0, cur_theta)
    history.append(cur_theta)
    
    for t in range(max_iter - 2):
        delta = np.abs(cur_theta - prev_theta)
        if delta < tol:
            print(f'RRM converged in {t + 1} iterations.')
            return history
        else:
            prev_theta = cur_theta
            cur_theta = 1. if sum(cur_Y) < 0 else -1.
            history.append(cur_theta)
            
            prev_Y = cur_Y.copy()
            cur_Y = shift_dist(Y0, cur_theta)
    
    print(f'RRM failed to converge in {max_iter} iterations.')
    return history


def approx_dist_grad(means, thetas):
    dmeans = np.array([m - means[-1] for m in means])
    dthetas = np.array([t - thetas[-1] for t in thetas])
    
    return sum(dmeans * dthetas) / sum(dthetas ** 2)


def grad2(cur_Y, cur_theta, means, thetas):
    f_prime = approx_dist_grad(means, thetas)
    
    loss_vec = np.array([loss(y, cur_theta) for y in cur_Y])
    mean = np.mean(cur_Y)
    
    return f_prime * np.dot(loss_vec, cur_Y - mean) / len(cur_Y)


def PGD(Y0, theta0, H, lr, tol, max_iter):
    history = deque()
    
    thetas = deque(maxlen = H + 1)
    means  = deque(maxlen = H + 1)
    
    cur_theta = theta0
    cur_Y = shift_dist(Y0, cur_theta).copy()
    history.append(cur_theta)
    thetas.append(cur_theta)
    means.append(np.mean(cur_Y))
    
    grad = grad1(cur_Y, cur_theta)
    cur_theta = clip(cur_theta - lr * grad)
    cur_Y = shift_dist(Y0, cur_theta).copy()
    history.append(cur_theta)
    thetas.append(cur_theta)
    means.append(np.mean(cur_Y))
    
    for t in range(max_iter - 2):
        grad = grad1(cur_Y, cur_theta) + grad2(cur_Y, cur_theta, means, thetas)
        
        if np.abs(grad) < tol:
            print(f'PGD converged in {t + 1} iterations.')
            return history
        else:
            cur_theta = clip(cur_theta - lr * grad)
            cur_Y = shift_dist(Y0, cur_theta).copy()
            history.append(cur_theta)
            thetas.append(cur_theta)
            means.append(np.mean(cur_Y))
    
    print(f'PGD failed to converge in {max_iter} iterations.')
    return history


def flaxman(n, theta0, lr, delta, max_iter):
    history = deque()
    queries = deque()
    
    theta = theta0
    
    for t in range(max_iter):
        # Sample u ~ Unif(sphere)
        u = np.random.randn(d)
        u /= np.linalg.norm(u)
        
        history.append(theta)
        perturbed_theta = np.array(clip(theta + delta * u)).reshape((1,1))
        queries.append(perturbed_theta)
        
        X = shift_dist(np.zeros(n), perturbed_theta).reshape((-1, 1))
        loss_estimate = sum(X @ perturbed_theta) / n
        grad_estimate = (d * loss_estimate / delta) * u
        
        theta = clip(theta - lr * grad_estimate)
    
    return history, queries


np.random.seed(0)
plt.style.use('ggplot')

runs = 10
n  = 500
H  = 4
lr = 0.1
tol = 0
max_iter = 30
delta = 0.3


pgd_hists = np.zeros((runs, max_iter))
rgd_hists = np.zeros((runs, max_iter))
rrm_hists = np.zeros((runs, max_iter))
flx_hists = np.zeros((runs, max_iter))
flq_hists = np.zeros((runs, max_iter))

for r in range(runs):
    Y0     = np.random.randn(n)
    theta0 = 0.5 * (np.random.rand() + 1)
    
    pgd_hists[r, :] = np.array(PGD(Y0, theta0, H, lr, tol, max_iter))
    rgd_hists[r, :] = np.array(RGD(Y0, theta0, lr, tol, max_iter))
    rrm_hists[r, :] = np.array(RRM(Y0, theta0, lr, tol, max_iter))
    flx_hists[r, :], flq_hists[r, :] = np.array((flaxman(n, theta0, lr, delta, max_iter)))



def perf_loss(theta):
    return np.sqrt(a1 * theta + a0) * theta

# Convert theta histories to loss histories
rrm_loss = np.zeros(rrm_hists.shape)
for r in range(runs):
    rrm_loss[r, :] = np.array([perf_loss(t) for t in rrm_hists[r, :]])
rrm_lmean = np.mean(rrm_loss, axis = 0)
rrm_lsem  = sem(rrm_loss, axis = 0)

rgd_loss = np.zeros(rgd_hists.shape)
for r in range(runs):
    rgd_loss[r, :] = np.array([perf_loss(t) for t in rgd_hists[r, :]])
rgd_lmean = np.mean(rgd_loss, axis = 0)
rgd_lsem  = sem(rgd_loss, axis = 0)

pgd_loss = np.zeros(pgd_hists.shape)
for r in range(runs):
    pgd_loss[r, :] = np.array([perf_loss(t) for t in pgd_hists[r, :]])
pgd_lmean = np.mean(pgd_loss, axis = 0)
pgd_lsem  = sem(pgd_loss, axis = 0)

flx_loss = np.zeros(flx_hists.shape)
for r in range(runs):
    flx_loss[r, :] = np.array([perf_loss(t) for t in flx_hists[r, :]])
flx_lmean = np.mean(flx_loss, axis = 0)
flx_lsem  = sem(flx_loss, axis = 0)

flq_loss = np.zeros(flq_hists.shape)
for r in range(runs):
    flq_loss[r, :] = np.array([perf_loss(t) for t in flq_hists[r, :]])
flq_lmean = np.mean(flq_loss, axis = 0)
flq_lsem  = sem(flq_loss, axis = 0)

rrm_mean = np.mean(rrm_hists, axis = 0)
rrm_sem  = sem(rrm_hists, axis = 0)

rgd_mean = np.mean(rgd_hists, axis = 0)
rgd_sem  = sem(rgd_hists, axis = 0)

pgd_mean = np.mean(pgd_hists, axis = 0)
pgd_sem  = sem(pgd_hists, axis = 0)

flx_mean = np.mean(flx_hists, axis = 0)
flx_sem  = sem(flx_hists, axis = 0)

flq_mean = np.mean(flq_hists, axis = 0)
flq_sem  = sem(flq_hists, axis = 0)


plt.style.use('ggplot')
plt.rc('font', size = 20)
ratio = 1.

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

x = range(max_iter)
plt.figure()
ax = plt.subplot(1, 2, 1)
plt.plot(x, opt * np.ones(max_iter), label = 'OPT')
plt.plot(x, stab * np.ones(max_iter), label = 'STAB')

# plot = plt.plot(x, rrm_mean, label = 'RRM')
# plt.fill_between(x, rrm_mean + rrm_sem, rrm_mean - rrm_sem, color = plot[0].get_color(), alpha = 0.3)

plot = plt.plot(x, rgd_mean, label = 'RGD', c=colors[2])
plt.fill_between(x, rgd_mean + rgd_sem, rgd_mean - rgd_sem, color = plot[0].get_color(), alpha = 0.3)

plot = plt.plot(x, flx_mean, label = 'FLX', c=colors[3])
plt.fill_between(x, flx_mean + flx_sem, flx_mean - flx_sem, color = plot[0].get_color(), alpha = 0.3)

plot = plt.plot(x, flq_mean, label = 'FLX queries', c=colors[5])
plt.fill_between(x, flq_mean + flq_sem, flq_mean - flq_sem, color = plot[0].get_color(), alpha = 0.3)

plot = plt.plot(x, pgd_mean, label = 'PerfGD', c=colors[4])
plt.fill_between(x, pgd_mean + pgd_sem, pgd_mean - pgd_sem, color = plot[0].get_color(), alpha = 0.3)

plt.xlabel('Training iteration')
plt.ylabel('θ')
plt.legend()

xleft, xright = ax.get_xlim()
ybottom, ytop = ax.get_ylim()
ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)


ax = plt.subplot(1, 2, 2)
plt.plot(x, opt_loss * np.ones(max_iter), label = 'OPT')
plt.plot(x, stab_loss * np.ones(max_iter), label = 'STAB')

# plot = plt.plot(x, rrm_lmean, label = 'RRM')
# plt.fill_between(x, rrm_lmean + rrm_lsem, rrm_lmean - rrm_lsem, color = plot[0].get_color(), alpha = 0.3)

plot = plt.plot(x, rgd_lmean, label = 'RGD', c=colors[2])
plt.fill_between(x, rgd_lmean + rgd_lsem, rgd_lmean - rgd_lsem, color = plot[0].get_color(), alpha = 0.3)

plot = plt.plot(x, flx_lmean, label = 'FLX', c=colors[3])
plt.fill_between(x, flx_lmean + flx_lsem, flx_lmean - flx_lsem, color = plot[0].get_color(), alpha = 0.3)

plot = plt.plot(x, flq_lmean, label = 'FLX queries', c=colors[5])
plt.fill_between(x, flq_lmean + flq_lsem, flq_lmean - flq_lsem, color = plot[0].get_color(), alpha = 0.3)

plot = plt.plot(x, pgd_lmean, label = 'PerfGD', c=colors[4])
plt.fill_between(x, pgd_lmean + pgd_lsem, pgd_lmean - pgd_lsem, color = plot[0].get_color(), alpha = 0.3)

plt.xlabel('Training iteration')
plt.ylabel('Performative loss')
plt.legend()

xleft, xright = ax.get_xlim()
ybottom, ytop = ax.get_ylim()
ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)

