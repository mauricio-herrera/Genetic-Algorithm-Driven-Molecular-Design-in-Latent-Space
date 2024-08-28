
import pytensor
import pytensor.tensor as pt

import numpy as np

from .gauss import *

from pytensor.tensor.slinalg import Cholesky as MatrixChol

import math

def n_pdf(x):
    return 1.0 / pt.sqrt(2 * math.pi) * pt.exp(-0.5 * x**2)

def log_n_pdf(x):
    return -0.5 * pt.log(2 * math.pi) - 0.5 * x**2

def n_cdf(x):
    return 0.5 * (1.0 + pt.erf(x / pt.sqrt(2.0)))

def log_n_cdf_approx(x):
    return log_n_pdf(x) - pt.log(-x - 1/x + 2 / x**3)

def log_n_cdf(x):
    x = pt.switch(pt.lt(x, -10), log_n_cdf_approx(x), pt.log(n_cdf(x)))
    return x

def ratio(x):
    x = pt.switch(pt.lt(x, -10), -(1.0/x - 1.0/x**3 + 3.0/x**5 - 15.0/x**7), n_cdf(x) / n_pdf(x))
    return x

def LogSumExp(x, axis=None):
    x_max = pt.max(x, axis=axis, keepdims=True)
    return pt.log(pt.sum(pt.exp(x - x_max), axis=axis, keepdims=True)) + x_max

class Sparse_GP:
    def __init__(self, n_inducing_points, n_points, input_d, input_means, input_vars, training_targets):
        self.ignore_variances = True
        self.n_inducing_points = n_inducing_points
        self.n_points = n_points
        self.input_d = input_d
        self.training_targets = training_targets
        self.input_means = input_means
        self.input_vars = input_vars

        initial_value = np.zeros((n_inducing_points, n_inducing_points))
        self.LParamPost = pytensor.shared(value=initial_value.astype(pytensor.config.floatX), name='LParamPost', borrow=True)
        self.mParamPost = pytensor.shared(value=initial_value[:, 0:1].astype(pytensor.config.floatX), name='mParamPost', borrow=True)
        self.lls = pytensor.shared(value=np.zeros(input_d).astype(pytensor.config.floatX), name='lls', borrow=True)
        self.lsf = pytensor.shared(value=np.zeros(1).astype(pytensor.config.floatX)[0], name='lsf', borrow=True)
        self.z = pytensor.shared(value=np.zeros((n_inducing_points, input_d)).astype(pytensor.config.floatX), name='z', borrow=True)
        self.lvar_noise = pytensor.shared(value=0.0 * np.ones(1).astype(pytensor.config.floatX)[0], name='lvar_noise', borrow=True)

        self.set_for_training = 1.0

        self.jitter = 1e-3

    def compute_output(self):
        self.Kzz = compute_kernel(self.lls, self.lsf, self.z, self.z) + pt.eye(self.z.shape[0]) * self.jitter * pt.exp(self.lsf)
        self.KzzInv = pt.nlinalg.MatrixInversePSD()(self.Kzz)
        LLt = pt.dot(self.LParamPost, pt.transpose(self.LParamPost))
        self.covCavityInv = self.KzzInv + LLt * (self.n_points - self.set_for_training) / self.n_points
        self.covCavity = pt.nlinalg.MatrixInversePSD()(self.covCavityInv)
        self.meanCavity = pt.dot(self.covCavity, (self.n_points - self.set_for_training) / self.n_points * self.mParamPost)
        self.KzzInvcovCavity = pt.dot(self.KzzInv, self.covCavity)
        self.KzzInvmeanCavity = pt.dot(self.KzzInv, self.meanCavity)
        self.covPosteriorInv = self.KzzInv + LLt
        self.covPosterior = pt.nlinalg.MatrixInversePSD()(self.covPosteriorInv)
        self.meanPosterior = pt.dot(self.covPosterior, self.mParamPost)
        self.Kxz = compute_kernel(self.lls, self.lsf, self.input_means, self.z)
        self.B = pt.dot(self.KzzInvcovCavity, self.KzzInv) - self.KzzInv
        v_out = pt.exp(self.lsf) + pt.dot(self.Kxz * pt.dot(self.Kxz, self.B), pt.ones_like(self.z[:, 0:1]))

        if self.ignore_variances:
            self.output_means = pt.dot(self.Kxz, self.KzzInvmeanCavity)
            self.output_vars = abs(v_out) + 0 * pt.sum(self.input_vars)
        else:
            self.EKxz = compute_psi1(self.lls, self.lsf, self.input_means, self.input_vars, self.z)
            self.output_means = pt.dot(self.EKxz, self.KzzInvmeanCavity)
            self.B2 = pt.outer(pt.dot(self.KzzInv, self.meanCavity), pt.dot(self.KzzInv, self.meanCavity))

            exact_output_vars = True

            if exact_output_vars:
                self.psi2 = compute_psi2(self.lls, self.lsf, self.z, self.input_means, self.input_vars)
                ll = pt.transpose(self.EKxz[:, None, :] * self.EKxz[:, :, None], [1, 2, 0])
                kk = pt.transpose(self.Kxz[:, None, :] * self.Kxz[:, :, None], [1, 2, 0])
                v1 = pt.transpose(pt.sum(pt.sum(pt.shape_padaxis(self.B2, 2) * (self.psi2 - ll), 0), 0, keepdims=True))
                v2 = pt.transpose(pt.sum(pt.sum(pt.shape_padaxis(self.B, 2) * (self.psi2 - kk), 0), 0, keepdims=True))
            else:
                v1 = 0
                v2 = 0
                n = self.input_d
                for j in range(1, n + 1):
                    mask = pt.zeros_like(self.input_vars)
                    mask = pt.set_subtensor(mask[:, j - 1], 1)
                    inc = mask * pt.sqrt(n * self.input_vars)
                    self.kplus = pt.sqrt(1.0 / (2 * n)) * compute_kernel(self.lls, self.lsf, self.input_means + inc, self.z)
                    self.kminus = pt.sqrt(1.0 / (2 * n)) * compute_kernel(self.lls, self.lsf, self.input_means - inc, self.z)

                    v1 += pt.dot(self.kplus * pt.dot(self.kplus, self.B2), pt.ones_like(self.z[:, 0:1]))
                    v1 += pt.dot(self.kminus * pt.dot(self.kminus, self.B2), pt.ones_like(self.z[:, 0:1]))
                    v2 += pt.dot(self.kplus * pt.dot(self.kplus, self.B), pt.ones_like(self.z[:, 0:1]))
                    v2 += pt.dot(self.kminus * pt.dot(self.kminus, self.B), pt.ones_like(self.z[:, 0:1]))

                v1 -= pt.dot(self.EKxz * pt.dot(self.EKxz, self.B2), pt.ones_like(self.z[:, 0:1]))
                v2 -= pt.dot(self.Kxz * pt.dot(self.Kxz, self.B), pt.ones_like(self.z[:, 0:1]))

            self.output_vars = abs(v_out) + abs(v2) + abs(v1)

        self.output_vars = self.output_vars + pt.exp(self.lvar_noise)

        return

    def get_params(self):
        return [self.lls, self.lsf, self.z, self.mParamPost, self.LParamPost, self.lvar_noise]

    def set_params(self, params):
        self.lls.set_value(params[0])
        self.lsf.set_value(params[1])
        self.z.set_value(params[2])
        self.mParamPost.set_value(params[3])
        self.LParamPost.set_value(params[4])
        self.lvar_noise.set_value(params[5])
        
    def getLogNormalizerCavity(self):
        assert self.covCavity is not None and self.meanCavity is not None and self.covCavityInv is not None
        return 0.5 * self.n_inducing_points * np.log(2 * np.pi) + 0.5 * pt.nlinalg.LogDetPSD()(self.covCavity) + 0.5 * pt.dot(pt.dot(pt.transpose(self.meanCavity), self.covCavityInv), self.meanCavity)

    def getLogNormalizerPrior(self):
        assert self.KzzInv is not None
        return 0.5 * self.n_inducing_points * np.log(2 * np.pi) - 0.5 * pt.nlinalg.LogDetPSD()(self.KzzInv)

    def getLogNormalizerPosterior(self):
        assert self.covPosterior is not None and self.meanPosterior is not None and self.covPosteriorInv is not None
        return 0.5 * self.n_inducing_points * np.log(2 * np.pi) + 0.5 * pt.nlinalg.LogDetPSD()(self.covPosterior) + 0.5 * pt.dot(pt.dot(pt.transpose(self.meanPosterior), self.covPosteriorInv), self.meanPosterior)

    def elbo(self):
        logZprior = self.getLogNormalizerPrior()
        logZcavity = self.getLogNormalizerCavity()
        logZposterior = self.getLogNormalizerPosterior()
        return logZposterior - logZcavity + logZprior

    def sample(self):
        L = pt.slinalg.Cholesky()(self.covPosterior)
        epsilon = pt.random.normal(size=(self.n_inducing_points, 1))
        return self.meanPosterior + pt.dot(L, epsilon)

# Funciones adicionales utilizadas en la clase Sparse_GP

def compute_kernel(lls, lsf, x1, x2):
    sqdist = pt.sum(pt.square(x1[:, None, :] - x2[None, :, :]), axis=2)
    return pt.exp(lsf) * pt.exp(-0.5 * pt.exp(-lls) * sqdist)

def compute_psi1(lls, lsf, input_means, input_vars, z):
    lsf = pt.exp(lsf)
    input_vars = pt.exp(input_vars)
    psi1 = lsf * pt.exp(-0.5 * pt.sum(pt.square(z[:, None, :] - input_means[None, :, :]) / (input_vars[None, :, :] + pt.exp(lls)), axis=2))
    return psi1

def compute_psi2(lls, lsf, z, input_means, input_vars):
    lsf = pt.exp(lsf)
    input_vars = pt.exp(input_vars)
    psi2 = pt.zeros((z.shape[0], z.shape[0], input_means.shape[0]))
    for i in range(input_means.shape[0]):
        psi2 = pt.set_subtensor(psi2[:, :, i], lsf * pt.exp(-0.5 * pt.sum(pt.square(z[:, None, :] - z[None, :, :]) / (2 * input_vars[None, None, i, :] + pt.exp(lls)), axis=2)))
    return psi2
