# This class represents a node within the network
import pytensor as pt
import pytensor.tensor as T
from .sparse_gp_theano_internal import *
import scipy.stats as sps
import scipy.optimize as spo
import numpy as np
import sys
import time
from tqdm import tqdm

def casting(x):
    return np.array(x).astype(pt.config.floatX)

def global_optimization(grid, lower, upper, function_grid, function_scalar, function_scalar_gradient):

    grid_values = function_grid(grid)
    best = grid_values.argmin()
    # We solve the optimization problem

    X_initial = grid[ best : (best + 1), : ]
    def objective(X):
        X = casting(X)
        X = X.reshape((1, grid.shape[ 1 ]))
        value = function_scalar(X)
        gradient_value = function_scalar_gradient(X).flatten()
        return np.float(value), gradient_value.astype(np.float)

    lbfgs_bounds = list(zip(lower.tolist(), upper.tolist()))
    x_optimal, y_opt, opt_info = spo.fmin_l_bfgs_b(objective, X_initial, bounds = lbfgs_bounds, iprint = 0, maxiter = 150)
    x_optimal = x_optimal.reshape((1, grid.shape[ 1 ]))

    return x_optimal, y_opt

def adam_theano(loss, all_params, learning_rate = 0.001):
    b1 = 0.9
    b2 = 0.999
    e = 1e-8
    gamma = 1 - 1e-8
    updates = []
    all_grads = pt.grad(loss, all_params)
    alpha = learning_rate
    t = pt.shared(casting(1.0))
    for theta_previous, g in zip(all_params, all_grads):
        m_previous = pt.shared(np.zeros(theta_previous.get_value().shape, dtype=pt.config.floatX))
        v_previous = pt.shared(np.zeros(theta_previous.get_value().shape, dtype=pt.config.floatX))
        m = b1 * m_previous + (1 - b1) * g                           # (Update biased first moment estimate)
        v = b2 * v_previous + (1 - b2) * g**2                            # (Update biased second raw moment estimate)
        m_hat = m / (1 - b1**t)                                          # (Compute bias-corrected first moment estimate)
        v_hat = v / (1 - b2**t)                                          # (Compute bias-corrected second raw moment estimate)
        theta = theta_previous - (alpha * m_hat) / (T.sqrt(v_hat) + e) #(Update parameters)
        updates.append((m_previous, m))
        updates.append((v_previous, v))
        updates.append((theta_previous, theta) )
    updates.append((t, t + 1.))
    return updates

class SparseGP:

    def __init__(self, input_means, input_vars, training_targets, n_inducing_points):

        self.input_means = pt.shared(value = input_means.astype(pt.config.floatX), borrow = True, name = 'X')
        self.input_vars = pt.shared(value = input_vars.astype(pt.config.floatX), borrow = True, name = 'X')
        self.original_training_targets = pt.shared(value = training_targets.astype(pt.config.floatX), borrow = True, name = 'y')
        self.training_targets = self.original_training_targets

        self.n_points = input_means.shape[ 0 ]
        self.d_input = input_means.shape[ 1 ]

        self.sparse_gp = Sparse_GP(n_inducing_points, self.n_points, self.d_input, self.input_means, self.input_vars, self.training_targets)

        self.set_for_prediction = False
        self.predict_function = None

    def initialize(self):
        self.sparse_gp.initialize()

    def setForTraining(self):
        self.sparse_gp.setForTraining()

    def setForPrediction(self):
        self.sparse_gp.setForPrediction()

    def get_params(self):
        return self.sparse_gp.get_params()

    def set_params(self, params):
        self.sparse_gp.set_params(params)

    def getEnergy(self):
        self.sparse_gp.compute_output()
        return self.sparse_gp.getContributionToEnergy()[ 0, 0 ]

    def predict(self, means_test, vars_test):

        self.setForPrediction()

        means_test = means_test.astype(pt.config.floatX)
        vars_test = vars_test.astype(pt.config.floatX)

        if self.predict_function is None:

            self.sparse_gp.compute_output()
            predictions = self.sparse_gp.getPredictedValues()

            X = T.matrix('X', dtype = pt.config.floatX)
            Z = T.matrix('Z', dtype = pt.config.floatX)

            self.predict_function = pt.function([ X, Z ], predictions, givens = { self.input_means: X, self.input_vars: Z  })

        predicted_values = self.predict_function(means_test, vars_test)

        self.setForTraining()

        return predicted_values

    def train_via_LBFGS(self, input_means, input_vars, training_targets, max_iterations = 500):

        input_means = input_means.astype(pt.config.floatX)
        input_vars = input_vars.astype(pt.config.floatX)
        training_targets = training_targets.astype(pt.config.floatX)
        self.input_means.set_value(input_means)
        self.input_vars.set_value(input_vars)
        self.original_training_targets.set_value(training_targets)

        self.initialize()
        self.setForTraining()

        X = T.matrix('X', dtype = pt.config.floatX)
        Z = T.matrix('Z', dtype = pt.config.floatX)
        y = T.matrix('y', dtype = pt.config.floatX)
        e = self.getEnergy()
        energy = pt.function([ X, Z, y ], e, givens = { self.input_means: X, self.input_vars: Z, self.training_targets: y })
        all_params = self.get_params()
        energy_grad = pt.function([ X, Z, y ], T.grad(e, all_params), \
            givens = { self.input_means: X, self.input_vars: Z, self.training_targets: y })

        initial_params = pt.function([ ], all_params)()

        params_shapes = [ s.shape for s in initial_params ]

        def de_vectorize_params(params):
            ret = []
            for shape in params_shapes:
                if len(shape) == 2:
                    ret.append(params[ : np.prod(shape) ].reshape(shape))
                    params = params[ np.prod(shape) : ]
                elif len(shape) == 1:
                    ret.append(params[ : np.prod(shape) ])
                    params = params[ np.prod(shape) : ]
                else:
                    ret.append(params[ 0 ])
                    params = params[ 1 : ]
            return ret

        def vectorize_params(params):
            return np.concatenate([ s.flatten() for s in params ])

        def objective(params):
                
            params = de_vectorize_params(params)
            self.set_params(params)
            energy_value = energy(input_means, input_vars, training_targets)
            gradient_value = energy_grad(input_means, input_vars, training_targets)

            return -energy_value, -vectorize_params(gradient_value)

        initial_params = vectorize_params(initial_params)
        x_opt, y_opt, opt_info = spo.fmin_l_bfgs_b(objective, initial_params, bounds = None, iprint = 1, maxiter = max_iterations)

        self.set_params(de_vectorize_params(x_opt))

        return y_opt

    def train_via_ADAM(self, input_means, input_vars, training_targets, input_means_test, input_vars_test, test_targets, \
        max_iterations = 500, minibatch_size = 4000, learning_rate = 1e-3, ignoroe_variances = True):

        input_means = input_means.astype(pt.config.floatX)
        input_vars = input_vars.astype(pt.config.floatX)
        training_targets = training_targets.astype(pt.config.floatX)
        n_data_points = input_means.shape[ 0 ]
        selected_points = np.random.choice(n_data_points, n_data_points, replace = False)[ 0 : min(n_data_points, minibatch_size) ]
        self.input_means.set_value(input_means[ selected_points, : ])
        self.input_vars.set_value(input_vars[ selected_points, : ])
        self.original_training_targets.set_value(training_targets[ selected_points, : ])

        print('Initializing network')
        sys.stdout.flush()
        self.setForTraining()
        self.initialize()

        X = T.matrix('X', dtype = pt.config.floatX)
        Z = T.matrix('Z', dtype = pt.config.floatX)
        y = T.matrix('y', dtype = pt.config.floatX)

        e = self.getEnergy()

        all_params = self.get_params()

        print('Compiling adam updates')
        sys.stdout.flush()

        process_minibatch_adam = pt.function([ X, Z, y ], -e, updates = adam_theano(-e, all_params, learning_rate), \
            givens = { self.input_means: X, self.input_vars: Z, self.original_training_targets: y })

        print('Training via ADAM')
        sys.stdout.flush()
        min_test_error = np.inf

        for iteration in range(max_iterations):

            selected_points = np.random.choice(n_data_points, n_data_points, replace = False)[ 0 : min(n_data_points, minibatch_size) ]
            batch_means = input_means[ selected_points, : ]
            batch_vars = input_vars[ selected_points, : ]
            batch_targets = training_targets[ selected_points, : ]
            train_error = process_minibatch_adam(batch_means, batch_vars, batch_targets)

            if ignoroe_variances:
                predicted_test_means = self.predict(input_means_test, np.zeros(input_means_test.shape, dtype = pt.config.floatX))
            else:
                predicted_test_means = self.predict(input_means_test, input_vars_test)

            test_error = np.mean((predicted_test_means - test_targets)**2)
            if test_error < min_test_error:
                min_test_error = test_error
                self.best_params = self.get_params()

            print(f'Iteration {iteration + 1}, train error: {train_error}, test error: {test_error}')
            sys.stdout.flush()

        self.set_params(self.best_params)

        return min_test_error

