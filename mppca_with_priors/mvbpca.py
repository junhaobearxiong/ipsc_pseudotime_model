import numpy as np
from sklearn.cluster import KMeans
from mppca_with_priors.vbpca import VBPCA
from scipy.special import digamma, gamma, gammaln


class MVBPCA(VBPCA):

    def __init__(self, t, m, C=None, priorX=None):
        # C: N-vector, C[i] = j means cell i is assigned to cell line j
        # priorX: (P, N) matrix, where P is the number of input features
        VBPCA.__init__(self, t, m, priorX)

        self.C = C
        self.q_s = S(self.N, self.M)
        if self.C is None:
            self.q_pi = Pi(self.M)
        else:
            self.lines = np.unique(C)
            if not np.array_equal(self.lines, np.arange(0, self.lines.size)):
                raise ValueError('C should contain consecutive integers starting from 0 and ends at # cell lines - 1') 
            # each cell line l has a unique pi_l
            self.q_pi = [Pi(self.M, l) for l in self.lines]


    def fit(self, max_iter=10000, min_iter=1000, eps=1e-3, verbose=False, init=True):

        if init:
            self.init()
        self.l_q = np.array([self.lower_bound()])
        '''
        self.e_theta = []
        self.e_x = []
        self.e_phi = []
        self.h_theta = []
        self.h_x = []
        self.h_phi = []
        self.e_w = []
        self.h_w = []
        self.e_s = []
        self.h_s = []
        '''
        # self.q_phi_e_phi = []
        # self.q_x_cov = []
        for iteration in range(max_iter):
            if verbose:
                print('=================== \n Iteration: %i\n Lower bound: %0.2f' % (iteration, self.l_q[-1]))

            if self.priorX is None:
                self.q_x.update(self.t, self.q_mu, self.q_w, self.q_tau)
            else:
                self.q_x.update(self.t, self.q_mu, self.q_w, self.q_tau, self.priorX, self.q_theta, self.q_phi)
                # self.q_theta.update(self.priorX, self.q_x, self.q_phi)
                # self.q_phi.update(self.priorX, self.q_x, self.q_theta)

            self.q_w.update(self.t, self.q_x, self.q_mu, self.q_tau, self.q_alpha, self.q_s)
            self.q_alpha.update(self.q_w)

            if self.C is None:
                self.q_pi.update(self.q_s)
            else:
                for l in self.lines:
                    self.q_pi[l].update(self.q_s, self.C)

            if self.priorX is None:
                self.q_s.update(self.t, self.q_x, self.q_tau, self.q_pi, self.q_mu, self.q_w, self.C)
            else:
                self.q_s.update(self.t, self.q_x, self.q_tau, self.q_pi, self.q_mu, self.q_w, self.C, self.priorX, self.q_theta, self.q_phi)

            self.q_mu.update(self.t, self.q_x, self.q_w, self.q_tau, self.q_s)
            self.q_tau.update(self.t, self.q_x, self.q_w, self.q_mu, self.q_s)
            self.l_q = np.append(self.l_q, self.lower_bound())

            if self.priorX is not None:
                '''
                self.e_theta.append(self._e_theta())
                self.e_x.append(self._e_x())
                self.e_phi.append(self._e_phi())
                self.h_theta.append(self._h_theta())
                self.h_x.append(self._h_x())
                self.h_phi.append(self._h_phi())
                self.e_w.append(self._e_w())
                self.h_w.append(self._h_w())
                self.e_s.append(self._e_s())
                self.h_s.append(self._h_s())
                '''
                # self.q_phi_e_phi.append(self.q_phi.e_phi)
                # self.q_x_cov.append(self.q_x.cov)

            if iteration > min_iter and np.abs(self.l_q[-1] - self.l_q[-2]) < eps:
                print('===> Final lower bound: %0.3f' % self.l_q[-1])
                break


    def init(self):

        km = KMeans(self.M)
        km.fit(self.t.T)
        means = km.cluster_centers_.T
        self.q_mu.mean = means

        dist = np.zeros((self.M, self.N))
        for n in range(self.N):
            for m in range(self.M):
                dist[m, n] = np.sqrt(np.sum((self.t[:, [n]] - means[:, [m]]) ** 2)) ** -1
        self.q_s.s_nm = dist / np.sum(dist, 0)

    def lower_bound(self):

        l_bound = self._e_t() + self._e_tau() + self._e_mu() + self._e_pi() + self._e_s() + self._e_alpha() + self._e_w() + self._e_x()
        l_bound -= self._h_tau() + self._h_mu() + self._h_pi() + self._h_s() + self._h_alpha() + self._h_w() + self._h_x()
        if self.priorX is not None:
            l_bound += self._e_theta() + self._e_phi() - self._h_theta() - self._h_phi()
        return l_bound

    def _e_pi(self):
        if self.C is None:
            s1 = 0
            u = self.q_pi.u_m[0]
            for m in range(self.M):
                s1 += self.q_pi.e_log_pi[m]
            s2 = gammaln(u)*self.M - gammaln(self.M*u)
            return (u - 1)*s1 - s2
        else:
            result = 0
            for l in self.lines:
                s1 = 0
                u = self.q_pi[l].u_m[0]
                for m in range(self.M):
                    s1 += self.q_pi[l].e_log_pi[m]
                s2 = gammaln(u)*self.M - gammaln(self.M*u)
                result += (u - 1)*s1 - s2
            return result

    def _e_s(self):
        if self.C is None:
            return np.sum(self.q_s.s_nm*self.q_pi.e_log_pi.reshape(self.M, 1))
        else:
            result = 0
            for l in self.lines:
                result += np.sum(self.q_s.s_nm[:, self.C==l]*self.q_pi[l].e_log_pi.reshape(self.M, 1))
            return result


    def _h_pi(self):
        if self.C is None:
            s1 = 0
            u = self.q_pi.u
            for m in range(self.M):
                s1 += (u[m] - 1) * self.q_pi.e_log_pi[m]
            s2 = 0
            for m in range(self.M):
                s2 += gammaln(u[m])
            s2 -= gammaln(np.sum(u))
            return s1 - s2
        else:
            result = 0
            for l in self.lines:
                s1 = 0
                u = self.q_pi[l].u
                for m in range(self.M):
                    s1 += (u[m] - 1) * self.q_pi[l].e_log_pi[m]
                s2 = 0
                for m in range(self.M):
                    s2 += gammaln(u[m])
                s2 -= gammaln(np.sum(u))
                result += s1 - s2
            return result

    def _h_s(self):
        return np.sum(self.q_s.s_nm*self.q_s.log_s_nm)


class Pi:

    def __init__(self, m, l=None):
        self.M = m
        self.l = l # which cell line this pi controls
        self.u_m = 1e-3 * np.ones((self.M,))
        self.u = self.u_m
        self.e_pi = self.u / np.sum(self.u)
        self.e_log_pi = np.empty((self.M,))
        self.calc_e_log_pi()

    def update(self, q_s, C=None):
        if self.l is None:
            self.u = self.u_m + np.sum(q_s.s_nm, axis=1)
        else:
            # only sum the snm where cn = l
            self.u = self.u_m + np.sum(q_s.s_nm[:, C==self.l], axis=1)
        self.e_pi = self.u / np.sum(self.u)
        self.calc_e_log_pi()

    def calc_e_log_pi(self):
        u0 = np.sum(self.u)
        for m in range(self.M):
            u_m = self.u[m]
            # this is how to compute E[log xi] when x1, ..., xK is Dirichlet
            self.e_log_pi[m] = digamma(u_m) - digamma(u0)


class S:

    def __init__(self, n, m):
        self.N = n
        self.M = m
        temp = np.random.normal(10.0, 1e-1, self.M * self.N).reshape(self.M, self.N)
        temp = np.maximum(0.0, temp)
        self.s_nm = temp / np.sum(temp, 0)
        self.s_nm0 = self.s_nm
        self.log_s_nm = self.s_nm

    def update(self, t, q_x, q_tau, q_pi, q_mu, q_w, C=None, priorX=None, q_theta=None, q_phi=None):
        s_n = np.zeros((self.M, self.N))
        for m in range(self.M):
            mu = q_mu.mean[:, [m]]
            e_w = q_w.mean[m]
            wTw = q_w.wtw[m]
            x = q_x.mean[m]

            # print('m: {}, cov: {}'.format(m, np.trace(q_x.cov[m])))

            s_n[m] = np.einsum('in,in->n', t, t).T + np.einsum('ij,ij->j', mu, mu) + np.einsum('ii', q_mu.cov[m])
            s_n[m] += 2 * np.einsum('l,ln->n', np.einsum('ij,ik->k', mu, e_w), x)
            s_n[m] -= 2 * np.einsum('ij,ji->i', np.einsum('ij,il->jl', t, e_w), x)
            s_n[m] -= 2 * np.einsum('ij,ik->j', t, mu)
            temp = np.einsum('ij,kj->jik', x, x) + q_x.cov[m]
            temp = np.einsum('ij,kjm->kim', wTw, temp)
            s_n[m] += np.einsum('kll->k', temp)

        # print(np.sum(s_n))

        self.square = s_n

        for m in range(self.M):
            x_m_cov = q_x.cov[m]
            x_m_mean = q_x.mean[m]
            if C is None:
                self.s_nm[m] = q_pi.e_log_pi[m]
            else:
                for l in np.unique(C):
                    self.s_nm[m, C==l] = q_pi[l].e_log_pi[m] 
            self.s_nm[m] += 0.5 * np.linalg.slogdet(x_m_cov)[1]
            self.s_nm[m] -= 0.5 * (q_tau.e_tau * self.square[m] + np.einsum('ij,ij->j', x_m_mean, x_m_mean) +
                                   np.tile(np.einsum('ii', x_m_cov), self.N))
            # if have priorX, the updates of s_nm should depend on theta, priorX, and phi
            if priorX is not None:
                # if q_phi is updated correctly, can just use q_phi.square
                s_xn = -2 * np.einsum('ij, ji -> i', np.einsum('ij, ik -> jk', x_m_mean, q_theta.mean[m]), priorX)
                temp = np.einsum('ij, kj -> jik', priorX, priorX)
                temp = np.einsum('ij, kjl -> kil', q_theta.theta_t_theta[m], temp)
                s_xn += np.einsum('ijj -> i', temp)
                self.s_nm[m] -= 0.5 * q_phi.e_phi * s_xn

        self.log_s_nm = self.s_nm
        # Normalising log likelihoods by using precision and discarding very negative logs
        precision = np.log(10**(-30)) - np.log(self.M)
        self.s_nm -= self.s_nm.max(0)
        idx_less = np.where(self.s_nm < precision)
        idx_greater = np.where(self.s_nm >= precision)
        self.s_nm[idx_less] = 0
        self.s_nm[idx_greater] = np.exp(self.s_nm[idx_greater])
        self.s_nm = self.s_nm / np.sum(self.s_nm, 0)
