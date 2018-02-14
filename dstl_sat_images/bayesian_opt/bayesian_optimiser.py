########################################################################
################### BAYESIAN GP hyper optimiser ########################
########################################################################
# orignal source (+ lots of my own alterations) 
# https://thuijskens.github.io/2016/12/29/bayesian-optimisation/
# https://github.com/thuijskens/bayesian-optimization/blob/master/ipython-notebooks/svm-optimization.ipynb

import numpy as np
import sklearn.gaussian_process as gp
import matplotlib.pyplot as plt

import scipy.stats
import scipy.optimize

##############################################################
#################### acquisition function ####################
##############################################################

def expected_improvement(model, x, sampled_losses):
    """ expected_improvement
    Expected improvement acquisition function.
    Arguments:
    ----------
        model:             GaussianProcessRegressor object trained on previously evaluated hyperparameters.
        x:                 [n_samples, n_hyperparams] The point for which the expected improvement needs to be computed.
        sampled_losses:    Numpy array that contains the values off the loss function for the previously evaluated hyperparameters.
    """

    mu, sigma = model.predict(x, return_std=True)
    sigma = sigma.reshape(mu.shape) # fix crazy return shape in sigma 
    
    best_loss = np.min(sampled_losses)
    
    # In case sigma equals zero
    with np.errstate(divide='ignore'):
        delta = -1.0 * (mu - best_loss) 
        Z =  delta / sigma
        improvement = delta * scipy.stats.norm.cdf(Z) + sigma * scipy.stats.norm.pdf(Z)
        improvement[sigma == 0.0] == 0.0

    return improvement

##############################################################
################ Best selection choice method ################
##############################################################

class MontoCarloProposer:
    """ 
    Proposes the next hyperparameter to sample the loss function for.
    Arguments:
    ----------
        acquisition_func:    Acquisition function to propose new optimise point.
        n_restarts:          Number of times to run the minimiser with different starting points.
        model:               Gaussian process trained on previously evaluated hyperparameters.
        losses:       Numpy array that contains the values off the loss function for the previously evaluated hyperparameters.
    """
    def __init__(self,
                 acquisition_func=None,
                 n_params=None,
                 samples=10000):
        self.acquisition_func=acquisition_func
        self.n_params = n_params
        self.samples = samples

    def __call__(self,
                 model,
                 hypers,
                 losses):
        x = np.random.uniform(0.0, 1.0, size=(self.samples, self.n_params))
        ei = self.acquisition_func(model, x, losses)
        idx = np.argmax(ei)
        next_sample = x[idx, :]

        return next_sample

class MontoCarloDescentProposer:
    """ 
    Proposes the next hyperparameter to sample the loss function for.
    Tries to sample an item around the best so as to decsend from it
    Arguments:
    ----------
        acquisition_func:    Acquisition function to propose new optimise point.
        n_restarts:          Number of times to run the minimiser with different starting points.
        model:               Gaussian process trained on previously evaluated hyperparameters.
        losses:       Numpy array that contains the values off the loss function for the previously evaluated hyperparameters.
    """
    def __init__(self,
                 acquisition_func=None,
                 n_params=None,
                 samples=1000,
                 stddev=0.1):
        self.acquisition_func=acquisition_func
        self.n_params = n_params
        self.samples = samples
        self.stddev = stddev
        
    def __call__(self,
                 model,
                 hypers,
                 losses):
        idx = np.argmin(losses)
        x_base = hypers[idx, :]
        
        x = np.random.normal(scale=self.stddev,
                             size=(self.samples, self.n_params)) + x_base

        x = np.clip(x, 0.0, 1.0)
        ei = self.acquisition_func(model, x, losses)
        idx = np.argmax(ei)
        next_sample = x[idx, :]

        return next_sample
    
class BFGSProposer:
    """ 
    Proposes the next hyperparameter to sample the loss function for.
    Arguments:
    ----------
        acquisition_func:    Acquisition function to propose new optimise point.
        n_restarts:          Number of times to run the minimiser with different starting points.
        model:               Gaussian process trained on previously evaluated hyperparameters.
        losses:              Numpy array that contains the values off the loss function for the previously evaluated hyperparameters.
    """
    def __init__(self, 
                 acquisition_func=None,
                 n_params=None,
                 n_restarts=25):
        self.n_restarts     = n_restarts
        self.n_params = n_params
        self.acquisition_func=acquisition_func
        
    def __call__(self,
                 model, 
                 hypers,
                 losses):
        best_x = None
        best_acquisition_value = 1

        for starting_point in np.random.uniform(0.0, 1.0, size=(n_restarts, n_params)):
            res = scipy.optimize.minimize(fun    = acquisition_func,
                                          x0     = starting_point.reshape(1, -1),
                                          bounds = np.repeat([[0.0,1.0]], n_params, axis=0),
                                          method = 'L-BFGS-B',
                                          args   = (gaussian_process, losses, self.n_params))

            if res.fun < best_acquisition_value:
                best_acquisition_value = res.fun
                best_x = res.x

        return best_x

##############################################################
################ Overall hyper plane modeller ################
##############################################################

class RandomModel:
    def __init__(self, n_params):
        self.n_params = n_params
    
    def __call__(self, hypers, losses):
        next_sample = np.random.uniform(0.0, 1.0, (self.n_params))
        method = "random"
        
        return next_sample, method

class GuassianProcessModel:
    """
    alpha:           Variance of the error term of the GP.
    """

    def __init__(self,
                 proposer=None,
                 callback=None,
                 alpha=1e-5,
                 tag="gp(norm)"):
        self.alpha    = alpha
        self.proposer = proposer
        self.callback = callback
        self.tag      = tag
        
    def __call__(self, hypers, losses):
        # np.inf in the losses means broken hypers, failure to train or whatever else is bad
        # convert the inf to the worst known sample values
    
        clean_losses = losses.copy()
        if np.sum(losses == np.inf) > 0:
            max_loss = np.max(losses[losses != np.inf])
            clean_losses[losses == np.inf] = max_loss

        # Create the GP
        kernel = gp.kernels.Matern()
        model  = gp.GaussianProcessRegressor(kernel              = kernel,
                                             alpha               = self.alpha,
                                             n_restarts_optimizer= 10,
                                             normalize_y         = True)

        model.fit(hypers, clean_losses)

        # Sample next hyperparameter
        next_sample = self.proposer(model,
                                    hypers,
                                    clean_losses)
        
        self.callback(model, next_sample, hypers, losses)
        
        return next_sample, self.tag

class GuassianProcess25Model:
    """
    alpha:           Variance of the error term of the GP.
    """

    def __init__(self,
                 proposer=None,
                 callback=None,
                 alpha=1e-5,
                 epislon=1e-5,
                 tag="gp(25%)"):
        self.alpha    = alpha
        self.epislon  = epislon
        self.proposer = proposer
        self.callback = callback
        self.tag      = tag
        
    def __call__(self, hypers, losses):
        # np.inf in the losses means broken hypers, failure to train or whatever else is bad
        # convert the inf to the worst known sample values
    
        clean_losses = losses.copy()
        if np.sum(losses == np.inf) > 0:
            max_loss = np.max(losses[losses != np.inf])
            clean_losses[losses == np.inf] = max_loss
            
        # Create the GP -- 50% mean
        kernel = gp.kernels.Matern(nu=1.5)
        model_mean  = gp.GaussianProcessRegressor(kernel              = kernel,
                                                  alpha               = self.alpha,
                                                  n_restarts_optimizer= 10,
                                                  normalize_y         = True)

        model_mean.fit(hypers, clean_losses)

        # predict for each point
        mean = model_mean.predict(hypers)

        # now select the items less than or equal to the mean
        idx2 = (losses <= (mean + self.epislon))[:,0]
    
        # now remodel on lower sub set of data
        kernel             = gp.kernels.Matern(nu=1.5)
        model_25percential = gp.GaussianProcessRegressor(kernel              = kernel,
                                                         #alpha               = alpha[idx2,0],
                                                         alpha               = 1e-5,
                                                         n_restarts_optimizer= 10,
                                                         normalize_y         = True)
        
        model_25percential.fit(hypers[idx2,:], clean_losses[idx2,:])
    
        # Sample next hyperparameter
        next_sample = self.proposer(model_25percential,
                                    hypers,
                                    clean_losses)
        
        self.callback(model_25percential,
                      next_sample, hypers, losses)
        
        return next_sample, self.tag
    
class StackedDecisionModel:
    def __init__(self):
        self.model_stack = []
        self.total = 0
        
    def add(self, weight, model):        
        self.total = self.total + weight        
        self.model_stack.append((self.total,model))
        
    def __call__(self,hypers,losses):
        choice = self.total * np.random.random()

        for w,m in self.model_stack:
            if choice <= w:
                return m(hypers,losses)

##############################################################
#################### hyper plane rendering ###################
##############################################################

class KeyPointCA:
    # key point component analysis 
    # This basically slices the 4d+ world of hyperparams, along a best/worst axis (x) 
    # with a point of interast forming the other a axis(sy) 
    
    # the plane formula is currently 
    #  r(s,t) = OP + main_axis.s + other_axis.t
    #
    #  let W = [main_axis; other_axis], x = [s,t]^T
    #  y    = OP + x * W 
    #  x    = (y - OP) * W^T  

    def __init__(self):
        pass
    
    def transform(self,points):
        # project samples into 2d space
        return np.matmul(points - self.offset, np.transpose(self.W))
            
    def inverse_transform(self,points):
        return np.matmul(points, self.W) + self.offset

    def fit(self,
            next_sample,
            sampled_params,
            sampled_loss):
        # best and worst before the last sample point!
        best_idx  = np.argmin(sampled_loss[:-1])
        worst_idx = np.argmax(sampled_loss[:-1])
    
        shape = (1,sampled_params.shape[1])
        
        p1 = sampled_params[best_idx].reshape(shape)
        p2 = sampled_params[worst_idx].reshape(shape)
        #p3 = sampled_params[-1].reshape(shape)           # which is the selected best in the EI plane
        p3 = next_sample.reshape(shape)
        
        v21 = p2 - p1
        main_axis = v21 / np.sqrt(np.sum(v21 * v21))
        
        #Now my second axis should include p3 (the best point in the EI plane that was selected) 
        v31 = p3 - p1
        v31_unit = v31 / np.sqrt(np.sum(v31 * v31))

        # project the v31 against the main axis... 
        # https://www.youtube.com/watch?v=fqPiDICPkj8
        v31_projected = np.sum(main_axis * v31_unit)*main_axis        
    
        # then subtract the projected from the main axis to find the other axis
        other_axis = v31_unit - v31_projected
        other_axis = other_axis / np.sqrt(np.sum(other_axis * other_axis))
        
        # now formulate the PCA like translation matrix
        W = np.concatenate((main_axis,other_axis), axis=0)

        # ok now the center (0,0) should match my mutli dim centers (which are all [0,1] wide )
        pc = np.ones(shape) * 0.5
        
        # project center into 2d space
        stc = np.matmul(pc - p1, np.transpose(W))
        
        # return center to plane in Nd space
        ppc = np.matmul(stc, W) +  p1

        # and store the config
        self.W = W
        self.offset = ppc

def graphing_callback_keypoint(model,
                               next_sample,
                               sampled_params, 
                               sampled_loss):
    """
    Plots a heatmap (2D) of the estimated loss function and expected
    improvement function this iteration of the search algorithm.

    This makes a 2d plane slice into the hyper space using 3 key points,
    the best, the worst and the next point of intereast
    """
    if model is None:
        return
    
    n_params = sampled_params.shape[1]
    iteration = sampled_params.shape[0]
    
    # compute a keypoint slice through the data
    kca = KeyPointCA()
    kca.fit(next_sample,sampled_params,sampled_loss)
    
    X = np.concatenate((np.zeros((1,n_params)),
                        np.ones((1,n_params)),
                        np.eye(n_params),
                        sampled_params,
                        next_sample.reshape((1,) + next_sample.shape)),axis=0)
    X_kca = kca.transform(X)    

    # get kca'ed version of params
    kca_sampled_params = X_kca[(2+n_params):]
    
    # compute the translated data range and step for kca graphing
    #steps = 11
    steps = 21
    X_kca_min = np.amin(X_kca, axis=0).reshape((2,1))
    X_kca_max = np.amax(X_kca, axis=0).reshape((2,1))
    X_kca_step = (X_kca_max - X_kca_min) / (steps - 1)
    kca_param_axis  = X_kca_step * np.repeat(np.arange(steps+1).reshape((1,steps+1)),2, axis=0) + (X_kca_min - 0.5*X_kca_step)   
    # note the steps+1 and 0.5 X_kca_steps are for a boarder edge
    
    # Transform grids into vectors for EI evaluation
    kca_param_grid = np.array([[a1, a2] for a1 in kca_param_axis[0] for a2 in kca_param_axis[1]])

    param_grid = kca.inverse_transform(kca_param_grid)

    # latest sample
    # kca_last_sample = kca_sampled_params[-1]    
    kca_last_sample = kca.transform(next_sample.reshape((1,) + next_sample.shape))[0]
    
    mu, std = model.predict(param_grid, return_std=True)
    ei = expected_improvement(model,
                              param_grid,                               
                              sampled_loss)

    # clip out of zone data
    bounds_idx = (np.sum(param_grid > 1.05,axis=1) + np.sum(param_grid < -0.05,axis=1)) > 0    
    mu [bounds_idx] = np.nan
    std[bounds_idx] = np.nan
    ei [bounds_idx] = np.nan
    
    fig, ax = plt.subplots(1, 3, figsize=(18,5), sharex=True, sharey=True)

    X, Y = np.meshgrid(kca_param_axis[0], kca_param_axis[1], indexing='ij')
    
    # Loss contour plot
    cp2 = ax[0].contourf(X, Y, mu.reshape(X.shape))
    plt.colorbar(cp2, ax=ax[0])
    ax[0].autoscale(False)
    ax[0].scatter(kca_sampled_params[:, 0], kca_sampled_params[:, 1], zorder=1)
    ax[0].axvline(kca_last_sample[0], color='k')
    ax[0].axhline(kca_last_sample[1], color='k')
    #ax[0].scatter(kca_last_sample[0], kca_last_sample[1])
    ax[0].set_title("Mean estimate of loss surface for iteration %d" % (iteration))

    best_idx  = np.argmin(sampled_loss[:-1])
    worst_idx = np.argmax(sampled_loss[:-1])

    ax[0].scatter(kca_sampled_params[best_idx,0],  kca_sampled_params[best_idx,1],  marker='*', c='gold', s=150)
    ax[0].scatter(kca_sampled_params[worst_idx,0], kca_sampled_params[worst_idx,1], marker='*', c='red',  s=150)

    # stdev contour plot
    cp2 = ax[1].contourf(X, Y, std.reshape(X.shape))
    plt.colorbar(cp2, ax=ax[1])
    ax[1].autoscale(False)
    ax[1].scatter(kca_sampled_params[:, 0], kca_sampled_params[:, 1], zorder=1)
    ax[1].axvline(kca_last_sample[0], color='k')
    ax[1].axhline(kca_last_sample[1], color='k')
    #ax[1].scatter(kca_last_sample[0], kca_last_sample[1])
    ax[1].set_title("Std dev of loss surface for iteration %d" % (iteration))

    ax[1].scatter(kca_sampled_params[best_idx,0],  kca_sampled_params[best_idx,1],  marker='*', c='gold', s=150)
    ax[1].scatter(kca_sampled_params[worst_idx,0], kca_sampled_params[worst_idx,1], marker='*', c='red',  s=150)

    
    # EI contour plot
    cp = ax[2].contourf(X, Y, ei.reshape(X.shape))
    plt.colorbar(cp, ax=ax[2])
    ax[2].set_title("Expected Improvement. Last keypoint at (%.2f, %.2f) " % (kca_last_sample[0], kca_last_sample[1]))
    ax[2].autoscale(False)
    ax[2].axvline(kca_last_sample[0], color='k')
    ax[2].axhline(kca_last_sample[1], color='k')
    ax[2].scatter(kca_last_sample[0], kca_last_sample[1])
    
    return fig

class BayesianOptimizer:
    # Models the hyper parameter planes *lower bounds* to optimise the loss value returned from samples
    # the hyper planes are *all* between real values in the range [0,1]
    # use the sample_loss function to translate that real value into the actual config values required
    
    # Arguments:
    # ----------
    #     n_iters:         Number of iterations to run the search algorithm.
    #     sample_loss:     Function to be optimised.
    #     n_params:        The number of hyperparams to search into.
    #     initial_samples: Number of starting samples to gather before modeling hyper-space        
    #     gp_params:       Dictionary of parameters to pass on to the underlying Gaussian Process.
    #     random_search:   the ratio of how how many samples are drawn by pure chance

    def __init__(self,
                 n_params,
                 initial_samples=3,
                 initial_modeller=None,
                 callback        =None,
                 acquisition_func=None,
                 proposer        =None,
                 modeller        =None):

        self.n_params            = n_params
        self.initial_samples     = initial_samples

        if initial_modeller is None:
            initial_modeller = RandomModel(n_params=n_params)

        # default setup - 100% GP model with monto carlo choice of best point computed from EI
        if acquisition_func is None:
            acquisition_func = expected_improvement
            
        if proposer is None:
            proposer = MontoCarloProposer(acquisition_func=acquisition_func,
                                          n_params=n_params)
                        
        if modeller is None:
            modeller = GuassianProcessModel(proposer=proposer,
                                            callback=callback)

        self.initial_modeller = initial_modeller 
        self.modeller         = modeller
    
    def make_decision(self,hypers,losses):
        if np.sum(losses != np.inf) < self.initial_samples:
            next_sample, method = self.initial_modeller(hypers,losses)
        else:
            next_sample, method = self.modeller(hypers,losses)
                        
        return {"method": method, "sample": next_sample }
            
    def run(self,
            sampler_func,
            n_iters=1, 
            hypers=None,
            losses=None):
        # warning not tested as im really running it via main.py
        if hypers is None:
            hypers = np.zeros((0,self.n_params))
            
        if losses is None:
            losses = np.zeros((0,1))
                
        for n in range(n_iters):
            job = self.make_decision(hypers,losses)
            
            # Sample loss for new set of parameters
            loss = sampler_func(job)            

            sample = job["sample"]
            
            hypers = np.append(hypers, sample.reshape((1,sample.shape[0])), axis=0)
            losses = np.append(losses, np.array([[loss]]), axis=0) 
            
        return hypers, losses
