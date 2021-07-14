import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
from collections import deque
from sklearn.cluster import KMeans


# Problem instance constants.
# Currently, we assume the variance for each cluster is known.

d = 1

s1 = 1
a1 = 1.
a0 = -0.5

s2 = 0.5
b1 = -0.3
b0 = 1.

g = 0.5

pl_quad_coeff = g * a1 + (1 - g) * b1
pl_lin_coeff  = g * a0 + (1 - g) * b0

opt_theta = -pl_lin_coeff / (2 * pl_quad_coeff)
opt_loss  = -(pl_lin_coeff ** 2) / (4 * pl_quad_coeff)

fixed_theta = -pl_lin_coeff / pl_quad_coeff
fixed_loss  = pl_quad_coeff * (fixed_theta ** 2) + pl_lin_coeff * fixed_theta


def clip(theta):
    if theta > 1.:
        return 1.
    elif theta < -1:
        return -1.
    else:
        return theta


def mean1(theta):
    return a1 * theta + a0


def mean2(theta):
    return b1 * theta + b0


def shift_dist(n, theta):
    Y = np.zeros(n)
    for i in range(n):
        if np.random.rand() <= g:
            Y[i] = s1 * np.random.randn() + mean1(theta)
        else:
            Y[i] = s2 * np.random.randn() + mean2(theta)
    return Y


def loss(y, theta):
    return y * theta


def grad(y, theta):
    return y


def grad1(cur_Y, cur_theta):
    return np.mean(cur_Y)


def RGD(n, theta0, lr, tol, max_iter):
    history = deque()
    
    cur_theta = theta0
    cur_Y = shift_dist(n, cur_theta).copy()
    history.append(cur_theta)
    
    for t in range(max_iter - 1):
        grad = grad1(cur_Y, cur_theta)
        if np.abs(grad) < tol:
            print(f'RGD converged in {t + 1} iterations.')
            return history
        else:
            cur_theta = clip(cur_theta - lr * grad)
            history.append(cur_theta)
            
            cur_Y = shift_dist(n, cur_theta)
    
    print(f'RGD failed to converge in {max_iter} iterations.')
    return history


def RRM(n, theta0, lr, tol, max_iter):
    history = deque()
    
    prev_theta = theta0
    prev_Y = shift_dist(n, prev_theta).copy()
    history.append(prev_theta)
    
    cur_theta = theta0 - lr * grad1(prev_Y, prev_theta)
    cur_Y = shift_dist(n, cur_theta)
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
            cur_Y = shift_dist(n, cur_theta)
    
    print(f'RRM failed to converge in {max_iter} iterations.')
    return history
    

def approx_dist_grad(means, thetas):
    dmeans = np.array([m - means[-1] for m in means])
    dthetas = np.array([t - thetas[-1] for t in thetas])
    
    return sum(dmeans * dthetas) / sum(dthetas ** 2)


def grad2(cur_Y, cur_theta, means, thetas, s):
    f_prime = approx_dist_grad(means, thetas)
    
    loss_vec = np.array([loss(y, cur_theta) for y in cur_Y])
    mean = np.mean(cur_Y)
    
    return (f_prime / s ** 2) * np.dot(loss_vec.flatten(), cur_Y.flatten() - mean)


def pgd_dist(n, theta):
    Y = np.zeros(n)
    labels = np.zeros(n)
    for i in range(n):
        if np.random.rand() <= g:
            Y[i] = s1 * np.random.randn() + mean1(theta)
        else:
            Y[i] = s2 * np.random.randn() + mean2(theta)
            labels[i] = 1
    return Y, labels


def PGD(n, theta0, H, lr, tol, max_iter):
    history = deque()
    
    thetas = deque(maxlen = H + 1)
    means1 = deque(maxlen = H + 1)
    means2 = deque(maxlen = H + 1)
    
    cur_theta = theta0
    cur_Y, cur_labels = pgd_dist(n, cur_theta)
    cur_Y1 = cur_Y[cur_labels == 0].copy()
    cur_Y2 = cur_Y[cur_labels == 1].copy()
    
    history.append(cur_theta)
    thetas.append(cur_theta)
    means1.append(np.mean(cur_Y1))
    means2.append(np.mean(cur_Y2))
    
    grad = grad1(cur_Y, cur_theta)
    cur_theta = clip(cur_theta - lr * grad)
    cur_Y, cur_labels = pgd_dist(n, cur_theta)
    cur_Y1 = cur_Y[cur_labels == 0].copy()
    cur_Y2 = cur_Y[cur_labels == 1].copy()
    
    history.append(cur_theta)
    thetas.append(cur_theta)
    means1.append(np.mean(cur_Y1))
    means2.append(np.mean(cur_Y2))
    
    for t in range(max_iter - 2):
        grad = grad1(cur_Y, cur_theta) + (grad2(cur_Y1, cur_theta, means1, thetas, s1) + grad2(cur_Y2, cur_theta, means2, thetas, s2)) / n
        
        if np.abs(grad) < tol:
            print(f'PGD converged in {t + 1} iterations.')
            return history
        else:
            cur_theta = clip(cur_theta - lr * grad)
            cur_Y, cur_labels = pgd_dist(n, cur_theta)
            cur_Y1 = cur_Y[cur_labels == 0].copy()
            cur_Y2 = cur_Y[cur_labels == 1].copy()
            
            history.append(cur_theta)
            thetas.append(cur_theta)
            means1.append(np.mean(cur_Y1))
            means2.append(np.mean(cur_Y2))
    
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
        perturbed_theta = np.array(theta + delta * u).reshape((1, 1))
        queries.append(perturbed_theta)
        
        X = shift_dist(n, perturbed_theta).reshape((-1, 1))
        loss_estimate = sum(X @ perturbed_theta) / n
        grad_estimate = (d * loss_estimate / delta) * u
        
        theta = clip(theta - lr * grad_estimate)
    
    return history, queries


np.random.seed(0)
plt.style.use('ggplot')

runs = 10
n  = 1000
H  = 500
lr = 0.1
tol = 0
max_iter = 100
delta = 0.1

pgd_hists = np.zeros((runs, max_iter))
rgd_hists = np.zeros((runs, max_iter))
rrm_hists = np.zeros((runs, max_iter))
flx_hists = np.zeros((runs, max_iter))
flq_hists = np.zeros((runs, max_iter))

for r in range(runs):
    theta0 = (opt_theta + fixed_theta) / 2 + (2 * np.random.rand() - 1)
    
    pgd_hists[r, :] = PGD(n, theta0, H, lr, tol, max_iter)
    rgd_hists[r, :] = np.array(RGD(n, theta0, lr, tol, max_iter))
    rrm_hists[r, :] = np.array(RRM(n, theta0, lr, tol, max_iter))
    flx_hists[r, :], flq_hists[r, :] = np.array(flaxman(n, theta0, lr, delta, max_iter))


# # Convert theta histories to loss histories
# rrm_loss = rrm_hists * (rrm_hists + alpha)
# rrm_lmean = np.mean(rrm_loss, axis = 0)
# rrm_lsem  = sem(rrm_loss, axis = 0)

rgd_loss  = rgd_hists * (g * (a1 * rgd_hists + a0) + (1 - g) * (b1 * rgd_hists + b0))
rgd_lmean = np.mean(rgd_loss, axis = 0)
rgd_lsem  = sem(rgd_loss, axis = 0)

pgd_loss  = pgd_hists * (g * (a1 * pgd_hists + a0) + (1 - g) * (b1 * pgd_hists + b0))
pgd_lmean = np.mean(pgd_loss, axis = 0)
pgd_lsem  = sem(pgd_loss, axis = 0)

flx_loss  = flx_hists * (g * (a1 * flx_hists + a0) + (1 - g) * (b1 * flx_hists + b0))
flx_lmean = np.mean(flx_loss, axis = 0)
flx_lsem  = sem(flx_loss, axis = 0)

flq_loss  = flq_hists * (g * (a1 * flq_hists + a0) + (1 - g) * (b1 * flq_hists + b0))
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
ratio = 1.0

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Plot parameter movement and loss
x = range(max_iter)
plt.figure()
ax = plt.subplot(1, 2, 1)
plt.plot(x, opt_theta * np.ones(max_iter), label = 'OPT')
plt.plot(x, fixed_theta * np.ones(max_iter), label = 'STAB')

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

ax.set_ylim(-1.25, -0.2)

xleft, xright = ax.get_xlim()
ybottom, ytop = ax.get_ylim()
ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)


ax = plt.subplot(1, 2, 2)
plt.plot(x, opt_loss * np.ones(max_iter), label = 'OPT')
plt.plot(x, fixed_loss * np.ones(max_iter), label = 'STAB')

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