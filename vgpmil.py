from __future__ import print_function
import numpy as np
import cv2
from time import time

from helperfunctions import sigmoid, lambda_fun


class vgpmil(object):
    def __init__(self, kernel, num_inducing=50, max_iter=10, normalize=True, verbose=False):
        """
        :param kernel: Specify the kernel to be used
        :param num_inducing: nr of inducing points
        :param max_iter: maximum number of iterations
        :param normalize: normalizes the data before training
        :param verbose: regulate verbosity
        """
        self.kernel = kernel
        self.num_ind = num_inducing
        self.max_iter = max_iter
        self.normalize = normalize
        self.verbose = verbose
        self.lH = np.log(1e12)

    def initialize(self, Xtrain, InstBagLabel, Bags, Z=None, pi=None, mask=None):
        """
        Initialize the model
        :param Xtrain: nxd array of n instances with d features each
        :param InstBagLabel:  n-dim vector with the bag label of each instance
        :param Bags: n-dim vector with the bag index of each instance
        :param Z: (opt) set of precalculated inducing points to be used
        :param pi: (opt) n-dim vector to specify instance labels for semi-supervised learning
        :param mask: (opt) n-dim boolean vector to fix instance labels and prevent them from being updated
        """
        self.Ntot = len(Bags)               # Nr of Training Instances
        self.B = len(np.unique(Bags))       # Nr of Training Bags
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
        self.m = np.random.randn(self.num_ind, 1)
        self.S = np.identity(self.num_ind) + np.random.randn(self.num_ind, self.num_ind) * 0.01

        # The parameters for q(y)
        if pi is not None:
            assert mask is not None, "Don't forget to provide a mask"
            self.mask = mask.copy()
            self.pi = pi.copy()
        else:
            self.pi = np.random.uniform(0, 0.1, size=self.Ntot)
            self.mask = np.ones(self.Ntot) == 1

        # The parameters for Jaakkola
        self.xi = np.random.randn(self.Ntot)




    def train(self, Xtrain, InstBagLabel, Bags, Z=None, pi=None, mask=None, init=True):
        """
        Train the model
        :param Xtrain: nxd array of n instances with d features each
        :param InstBagLabel:  n-dim vector with the bag label of each instance
        :param Bags: n-dim vector with the bag index of each instance
        :param Z: (opt) set of precalculated inducing points to be used
        :param pi: (opt) n-dim vector to specify instance labels for semi-supervised learning
        :param mask: (opt) n-dim boolean vector to fix instance labels and prevent them from being updated
        :param init: (opt) whether to initialize before training
        """
        if init:
            start = time()
            self.initialize(Xtrain, InstBagLabel, Bags, Z=Z, pi=pi, mask=mask)
            stop = time()
            if self.verbose:
                print("Initialized. \tMinutes needed:\t", (stop - start) / 60.)

        for it in range(self.max_iter):
            start = time()
            if self.verbose:
                print("Iter %i/%i" % (it + 1, self.max_iter))

            # Updating q(u)
            Lambda = 2 * lambda_fun(self.xi)
            Si = self.Kzzinv + np.dot(self.KzziKzx * Lambda, self.KzziKzx.T)
            self.S = np.linalg.inv(Si + np.identity(self.num_ind) * 1e-8)
            self.m = self.S.dot(self.KzziKzx).dot(self.pi - 0.5)

            # Updating q(y)
            Ef = self.KzziKzx.T.dot(self.m)
            mmTpS = np.outer(self.m, self.m) + self.S
            Eff = np.einsum("ij,ji->i", np.dot(self.KzziKzx.T, mmTpS), self.KzziKzx) + self.f_var

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

            mask = self.mask
            self.pi[mask] = sigmoid(Ef[mask] + self.lH * (2 * self.InstBagLabel + Emax[mask] -
                                                          2 * self.InstBagLabel[mask] * Emax - 1))

            # Update Jaakkola
            self.xi = np.sqrt(Eff)

            stop = time()
            if self.verbose:
                print("Minutes needed: ", (stop - start) / 60.)

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


















