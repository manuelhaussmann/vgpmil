from __future__ import print_function
import numpy as np
import cv2
from time import time
from helperfunctions import sigmoid, lambda_fun


class lm_vgpmil(object):
    def __init__(self, kernel, num_inducing=50, max_iter=10, C=1.0, V=1.0, normalize=True, verbose=False):
        """
        :param kernel: Specify the kernel to be used
        :param num_inducing: nr of inducing points
        :param max_iter: maximum number of iterations
        :param C: C value to be used
        :param V: V value to be used
        :param normalize: normalizes the data before training
        :param verbose: regulate verbosity
        """
        self.kernel = kernel
        self.num_ind = num_inducing
        self.max_iter = max_iter
        self.normalize = normalize
        self.verbose = verbose
        self.C, self.C2 = C, C**2
        self.V, self.V2 = V, V**2
        self.lH = np.log(1e12)  # log(H)

    def initialize(self, Xtrain, InstBagLabel, Bags, Z=None):
        """
        Initialize the model
        :param Xtrain: Nxd-dim array containing the training Instances
        :param InstBagLabel: N-dim vector containing the Bag label for each instance
        :param Bags: N-dim vector. Bags_i gives the corresponding bag for instance i
        :param Z: Mxd-dim array containing the inducing points
        """
        self.Ntot = len(Bags)
        self.B = len(np.unique(Bags))
        self.InstBagLabel = InstBagLabel
        self.Bags = Bags

        if self.normalize:
            self.data_mean, self.data_std = np.mean(Xtrain, 0), np.std(Xtrain, 0)
            self.data_std[self.data_std == 0] = 1.0
            Xtrain = (Xtrain - self.data_mean) / self.data_std

        # Compute Inducing points if not provided
        if Z is not None:
            assert self.num_ind == Z.shape[0]
            self.Z = Z
        else:
            Xzeros = Xtrain[InstBagLabel == 0].astype("float32")
            Xones = Xtrain[InstBagLabel == 1].astype("float32")
            num_ind_pos = np.uint32(np.floor(self.num_ind * 0.5))
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            nr_attempts = 10
            _, _, Z0 = cv2.kmeans(Xzeros, self.num_ind - num_ind_pos, None, criteria, attempts=nr_attempts, flags=cv2.KMEANS_RANDOM_CENTERS)
            _, _, Z1 = cv2.kmeans(Xones, num_ind_pos, None, criteria, attempts=nr_attempts, flags=cv2.KMEANS_RANDOM_CENTERS)
            self.Z = np.concatenate((Z0, Z1))
            if self.verbose:
                print("Inducing points are computed")

        self.Kzzinv = np.linalg.inv(self.kernel.compute(self.Z) + np.identity(self.num_ind) * 1e-6)

        Kzx = self.kernel.compute(self.Z, Xtrain)
        self.KzziKzx = np.dot(self.Kzzinv, Kzx)
        self.f_var = 1 - np.einsum("ji,ji->i", Kzx, self.KzziKzx)

        # The parameters for q(u)
        self.m = np.random.randn(self.num_ind)
        self.S = np.identity(self.num_ind) + np.random.randn(self.num_ind, self.num_ind) * 0.01

        # The parameters for the two Jaakkolas (\xi, \phi)
        self.xi = np.random.randn(self.Ntot)
        self.phi = np.random.randn(self.Ntot)

        # The parameters for q(g) and q(y)
        self.tau = np.random.uniform(0, 1, size=self.Ntot)
        self.pi = np.random.uniform(0, 0.1, size=self.Ntot)

    def train(self, Xtrain, InstBagLabel, Bags, Z=None, init=True):
        """
        Train the model
        :param Xtrain: nxd array of n instances with d features each
        :param InstBagLabel:  n-dim vector with the bag label of each instance
        :param Bags: n-dim vector with the bag index of each instance
        :param Z: (opt) set of precalculated inducing points to be used
        :param init: (opt) whether to initialize before training
        """

        if init:
            start = time()
            self.initialize(Xtrain, InstBagLabel, Bags, Z=Z)
            stop = time()
            if self.verbose:
                print("Initialized. \tMinutes needed:\t", (stop - start) / 60.)

        for it in range(self.max_iter):
            start = time()
            if self.verbose:
                print("Iter %i/%i" % (it + 1, self.max_iter))

            yy = 2 * (self.pi - 0.5)
            lam_xi = lambda_fun(self.xi)
            lam_phi = lambda_fun(self.phi)

            # Updating q(u)
            Lambda = 2 * (lam_xi * self.C2 + lam_phi * self.tau)
            Si = self.Kzzinv + np.dot(self.KzziKzx * Lambda, self.KzziKzx.T)
            self.S = np.linalg.inv(Si + np.identity(self.num_ind) * 1e-8)
            tmp = self.C * yy * (self.tau - 0.5) + 2 * self.C2 * lam_xi * yy * self.V + self.tau * (self.pi - 0.5)
            self.m = self.S.dot(self.KzziKzx).dot(tmp)

            # Updating q(g)
            Ef = self.KzziKzx.T.dot(self.m)
            mmTpS = np.outer(self.m, self.m) + self.S
            Eff = np.einsum("ij,ji->i", np.dot(self.KzziKzx.T, mmTpS), self.KzziKzx) + self.f_var

            self.tau =sigmoid(self.C * (yy * Ef - self.V) + Ef * (self.pi - 0.5) - lam_phi * Eff)

            # Update q(y)
            Emax = np.empty(len(self.pi))
            for b in np.unique(self.Bags):
                mask = self.Bags == b
                pisub = self.pi[mask]
                m1 = np.argmax(pisub)
                tmp = np.empty(len(pisub))
                tmp.fill(pisub[m1])
                pisub[m1] = -99
                m2 = np.argmax(pisub)
                tmp[m1] = pisub[m2]
                Emax[mask] = tmp
            Emax = np.clip(Emax, 0, 1)

            self.pi = sigmoid(2 * self.C * Ef * self.tau - self.C * Ef + 4 * lam_xi * self.C2 * Ef * self.V +
                              Ef * self.tau + self.lH * (2 * self.InstBagLabel - 2 * self.InstBagLabel * Emax - 1 + Emax))

            # Update the Jaakkola bounds
            self.xi = np.sqrt(self.C2 * (Eff - 2 * yy * Ef * self.V + self.V2))
            self.phi = np.sqrt(Eff * self.tau)

            stop = time()
            if self.verbose:
                print("Minutes needed:\t", (stop - start) / 60.)

    def predict(self, Xtest):
        """
        Predict instances
        :param Xtest: mxd matrix of n instances with d features
        :return: Instance Predictions
        """
        if self.normalize:
            Xtest = (Xtest - self.data_mean) / self.data_std

        Kzx = self.kernel.compute(self.Z, Xtest)
        KzziKzx = np.dot(self.Kzzinv, Kzx)

        return sigmoid(np.dot(KzziKzx.T, self.m))















