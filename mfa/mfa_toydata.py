"""
Run MFA (VI) on toy data, compare parameter estimates to EM implementation
EM implementation cited from https://github.com/bobondemon/MixtureFA
"""
import pymc3 as pm
import numpy as np
import math
from numpy.linalg import *
import matplotlib.pyplot as plt
import seaborn as sns
from mfa_em import mfa_em
from mfa import MixtureFA


def gen_data(num_pts):
    """
    Generate toy data from a MFA: z ~ N(0,I), x|z ~ N(Wz+mu, sigma2*I)
    pi = [0.25, 0.75]
    mu1 = [-2, 1], mu2 = [1.5, -2]
    W1 = [-3, 1], W2 = [1.5, 2.2] (normalized by length: W1_normed = [-0.95, 0.32], W2_normed = [0.56, 0.83])
    sigma1 = sigma2 = 0.4    
    """
    pi1 = 0.25
    num_pts1 = int(pi1*num_pts)
    z = np.array([np.random.normal(loc=0, scale=1, size=num_pts1)])  # 1xnum_pts
    W1 = np.array([[-3],[1]])  # 2x1
    W1 = W1/np.sqrt(np.sum(W1*W1))
    mu1 = np.array([[-2],[1]])  # 2x1
    # multiply by -2 and 2 represent the range of z that covers +-2*sigma of the distribution 
    line_points1 = np.matmul(W1,[[-2, 2]])+mu1
    sigma1 = 0.4
    noise1 = np.array([np.random.normal(loc=0, scale=sigma1, size=num_pts1) for i in list(range(2))])
    out1 = np.matmul(W1,z) + mu1 + noise1  # 2xnum_pts

    pi2 = 1.0-pi1
    num_pts2 = int(pi2*num_pts)
    z = np.array([np.random.normal(loc=0, scale=1, size=num_pts2)])  # 1xnum_pts
    W2 = np.array([[1.5],[2.2]])  # 2x1
    W2 = W2/np.sqrt(np.sum(W2*W2))
    mu2 = np.array([[1.5],[-2]])  # 2x1
    line_points2 = np.matmul(W2,[[-2, 2]])+mu2
    sigma2 = 0.4
    noise2 = np.array([np.random.normal(loc=0, scale=sigma2, size=num_pts2) for i in list(range(2))])
    out2 = np.matmul(W2,z) + mu2 + noise2  # 2xnum_pts

    return out1, line_points1, out2, line_points2


def guassianPlot2D(mu,sigma):
    """
    Plot of 2D gaussian as an elongated circle based on mean and covariance
    """
    mu = np.array(mu).reshape((-1,1))
    sigma = np.array(sigma).reshape(2,2)
    assert(len(mu)==2)

    # first we generate the unit circle of (x,y) points
    def PointsInCircum(r,n=100):
        return [(math.cos(2*math.pi/n*x)*r,math.sin(2*math.pi/n*x)*r) for x in range(0,n+1)]
    pts = np.array(PointsInCircum(2)).T  # 2xN

    # we then calculate the sqrt of the sigma
    # the np.eig has output ( (N,), (N,N) )
    # where each col of eig_vec is eigen vector and is of norm=1
    # note that eig_val is not sorted
    # eig_vec * eig_val * eig_vec.T = sigma
    eig_val, eig_vec = eig(sigma)  # we assume sigma is positive definite, so eig_val > 0
    eig_val_sqrt = np.sqrt(eig_val).reshape((1,-1))  # 1x2
    sigma_sqrt = eig_vec*eig_val_sqrt  # 2x2

    # finally, transform pts based on sigma_sqrt
    # y = Ax, cov(y) = A*cov(x)*A.T
    # since cov(x) = I, so cov(y) = A*A.T
    # if we let A = sqrt(sigma), then cov(y) = sigma, which is the covariance matrix we need
    pts = np.matmul(sigma_sqrt,pts)  # 2xN
    pts += mu

    return pts


def plot_result(ax, W, mu, psi):
    """
    This function assume K is on the first column for all the parameters
    so outputs of the VI implementation needs to be transposed before fed in
    """
    def normalize(a):
        return a / np.sqrt(np.sum(a * a))
    ax.scatter(data1[0],data1[1], c='b', marker='o')
    ax.scatter(data2[0],data2[1], c='r', marker='o')

    # the principle axis for each mixture
    # normalizing
    W[0,...] = normalize(W[0, ...])
    W[1,...] = normalize(W[1, ...])
    # this is how the lines are generated
    line_points1 = np.matmul(W[0,...],[[-2, 2]]) + mu[0,:].reshape((2,1))
    line_points2 = np.matmul(W[1,...],[[-2, 2]]) + mu[1,:].reshape((2,1))
    ax.plot(line_points1[0,:],line_points1[1,:],'g--',linewidth=1, label='learned principle axis 1')
    ax.plot(line_points2[0,:],line_points2[1,:],'m--',linewidth=1, label='learned principle axis 2')
    ax.plot(line_points1_orig[0,:],line_points1_orig[1,:],'k-', linewidth=1, label='true principle axis 1')
    ax.plot(line_points2_orig[0,:],line_points2_orig[1,:],'k-',linewidth=1, label='true principle axis 2')

    # plot the gaussian for each mixture
    # represented by an elongated circle (mean + covariance)
    W1_cov = np.matmul(W[0, ...], W[0, ...].T)
    W2_cov = np.matmul(W[1, ...], W[1, ...].T)
    if len(psi.shape) > 1:
        C1 = W1_cov + np.diag(psi[0, 0, :])
        C2 = W2_cov + np.diag(psi[1, 0, :])
    else:
        C1 = W1_cov + np.diag(psi)
        C2 = W2_cov + np.diag(psi)
    pts1 = guassianPlot2D(mu[0,...],C1)
    pts2 = guassianPlot2D(mu[1,...],C2)
    ax.plot(pts1[0,:],pts1[1,:],'g-.',linewidth=2, label='covariance 1')
    ax.plot(pts2[0,:],pts2[1,:],'m-.',linewidth=2, label='covaraince 2')
    ax.legend()


if __name__ == '__main__':
    N = 600
    G = 2
    K = 2
    np.random.seed(0)
    data1, line_points1_orig, data2, line_points2_orig = gen_data(N)
    data = np.concatenate([data1,data2],axis=1).T

    # fit EM
    # note: initialization is random, so could end up in local optima
    print('Fitting EM')
    pi, mu, W, psi = mfa_em(data.T,K=1,M=2)

    # fit ADVI
    print('Fitting ADVI')
    mfa_vi = MixtureFA(Y=data, name='toydata', K=2, advi_iter=100000)
    mfa_vi.fit()
    mfa_vi.get_posterior()
    W_vi = np.transpose(mfa_vi.posterior['V'])
    mu_vi = np.transpose(mfa_vi.posterior['MU'])
    # sigma is sd, needs to square
    psi_vi = np.square(np.transpose(mfa_vi.posterior['SIGMA'], [2, 0, 1]))
    print('Finish fitting')

    # plot result
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    plot_result(axs[0], W, mu, psi)
    axs[0].set_title('EM')
    plot_result(axs[1], W_vi, mu_vi, psi_vi)
    axs[1].set_title('ADVI')
    plt.savefig('outputs/toydata_params.png')

    # plot posterior assignments
    true_labels = np.repeat([0, 1], [int(0.25 * N), int(0.75 * N)])
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=true_labels, ax=axs[0])
    axs[0].set_title('color by true label')
    sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=mfa_vi.posterior['R'].argmax(axis=1), ax=axs[1])
    axs[1].set_title('color by estimated label')
    plt.savefig('outputs/toydata_assignments.png')

    # print parameters
    print('True PI')
    print(np.array([0.25, 0.75]))
    print('EM')
    print(pi)
    print('ADVI')
    print(mfa_vi.posterior['PI'])

    print('True MU')
    print(np.array([[-2, 1], [1.5, -2]]))
    print('EM')
    print(mu)
    print('ADVI')
    print(mu_vi)

    print('True W (up to multiplication of orthogonal matrix)')
    print(np.array([[-0.95, 0.32],[0.56, 0.83]]))
    print('EM')
    print(W)
    print('ADVI')
    print(W_vi)

    print('True sigma')
    print(np.array([0.16, 0.16]))
    print('EM')
    print(psi)
    print('ADVI')
    print(psi_vi)


