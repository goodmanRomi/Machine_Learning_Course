import numpy as np
from typing import Union
import math #to use factorial function

def poisson_log_pmf(k: Union[int, np.ndarray], rate: float) -> Union[float, np.ndarray]:
    """
    k: A discrete instance or an array of discrete instances
    rate: poisson rate parameter (lambda)

    return the log pmf value for instances k given the rate
    """

    log_p = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    
    #logP(K=k)=log(λ^k)+log(e^−λ)−log(k!)=klog(λ)−λ−log(k!)
    #from piazza discussions: allowed to import math 
    #calculation log(k!) can be very inefficient for large k, so we use loggamma
    
    k_array=np.asarray(k)
    #if k was an int, it will become a 0-dimentional array
    #if k was already an np array, it wont do anything to it
    #once we have our k_array, we can behave the same way for both cases
    
    #ensure all values are non negative and valid ints. 
    #k represents "how many events occured" so it can't be negative or partial
    if np.any(k_array<0):
        raise ValueError("All entries of k must be non negative ints for the Poisson PMF")
    
    if not np.all(np.mod(k_array,1)==0):
        raise ValueError("All entires of k must be integers")
    
    k_array=k_array.astype(int) #saftey

    #now we can compute log(k!) using math.lgamma(k+1) identity
    # For each integer k[i], log(k[i]!) = math.lgamma(k[i] + 1)

    lgamma_vector=np.vectorize(math.lgamma)
    log_factorial=lgamma_vector(k_array+1)

    #proceed to calculate the entire formula:k * log(rate) - rate - log(k!)

    log_rate=math.log(rate) #scalar
    log_p=k_array * log_rate - rate - log_factorial

    # if original k was int- we return a float, otherwise we return an array
    if np.isscalar(k) or (isinstance(k, np.ndarray) and k.ndim==0):
        return float(log_p)
    else:
        return log_p

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

def possion_analytic_mle(samples):
    """
    samples: set of univariate discrete observations

    return: the rate that maximizes the likelihood
    """
    mean = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    samples_vec=np.asarray(samples)
    
    #validate dataset- non empty, valid entries, non negative and whole ints

    if samples_vec.size==0:
        raise ValueError("cannot compute Poisson MLE on empty set")
    
    if np.any(samples_vec<0):
        raise ValueError("All counts must be nonnegatice for Poisson disterbution")
    
    if not np.all(np.mod(samples_vec,1)==0):
        raise ValueError("All couts must be integers")

    samples_vec=samples_vec.astype(int)


    #computing sample mean in vectorized way:
    #in exercise 2 in theoretical part we got lambda_hat = (1/n) * sum_i_to_n of x_i
    lambda_hat=np.mean(samples_vec)

    mean=lambda_hat
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return mean

def possion_confidence_interval(lambda_mle, n, alpha=0.05):
    """
    lambda_mle: an MLE for the rate parameter (lambda) in a Poisson distribution
    n: the number of samples used to estimate lambda_mle
    alpha: the significance level for the confidence interval (typically small value like 0.05)
 
    return: a tuple (lower_bound, upper_bound) representing the confidence interval
    """
    # Use norm.ppf to compute the inverse of the normal CDF
    from scipy.stats import norm
    lower_bound = None
    upper_bound = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    
    # computing z_{1 - α/2} = Φ⁻¹(1 - α/2) via scipy.stats.norm.ppf
    z = norm.ppf(1-alpha/2) 

    #estimating the standard error (SE) of λ̂ as sqrt(λ̂ / n)
    se = (lambda_mle/n)**0.5 

    lower_bound=lambda_mle - z * se
    if lower_bound<0:
        lower_bound=0.0
    upper_bound = lambda_mle + z * se

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return lower_bound, upper_bound

def get_poisson_log_likelihoods(samples, rates):
    """
    samples: set of univariate discrete observations
    rates: an iterable of rates to calculate log-likelihood by.

    return: 1d numpy array, where each value represent that log-likelihood value of rates[i]
    """
    likelihoods = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    samples_arr=np.asarray(samples)
    rates_arr=np.asarray(rates)

    #validating the sample 
    if samples_arr.size==0:
        raise ValueError("Data set must have at least one sample")
    if np.any(samples_arr < 0):
        raise ValueError("all values must be nonnegative")
    if not np.all(np.mod(samples_arr,1)==0):
        raise ValueError("all values ,ust be whole integers")
    
    samples_arr=samples_arr.astype(int)

    if np.any(rates_arr<=0):
        raise ValueError("all rate values must be positive")

    #precomputing data-dependent constants 
    #log L(λ; X) = (∑ x_i)·log(λ) – n·λ – ∑ log(x_i!)
    n=samples_arr.shape[0]  # n = number of observations
    total_count=np.sum(samples_arr)
    lgamma_vec=np.vectorize(math.lgamma)
    sum_log_factorials=np.sum(lgamma_vec(samples_arr+1))

    #computing log-likylhood for each λ
    # log L(λ; X) = total_count·log(λ) – n·λ – sum_log_factorials.
    log_rates=np.log(rates_arr)
    num1=total_count*log_rates
    num2=n*rates_arr

    likelihoods=num1-num2-sum_log_factorials 

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return likelihoods

class conditional_independence():

    def __init__(self):

        # You need to fill the None value with *valid* probabilities
        self.X = {0: 0.3, 1: 0.7}  # P(X=x)
        self.Y = {0: 0.3, 1: 0.7}  # P(Y=y)
        self.C = {0: 0.5, 1: 0.5}  # P(C=c)

        self.X_Y = {
            (0, 0): 0.18,   # P(X=0, Y=0) = P(X=0,Y=0,C=0) + P(X=0,Y=0,C=1) = 0.18 + 0 = 0.18
            (0, 1): 0.12,   # P(X=0, Y=1) = 0.12 + 0 = 0.12
            (1, 0): 0.12,   # P(X=1, Y=0) = 0.12 + 0 = 0.12
            (1, 1): 0.58    # P(X=1, Y=1) = 0.08 + 0.50 = 0.58
        }  # P(X=x, Y=y)

        self.X_C = {
            (0, 0): 0.30, # For C = 0:  P(X=0,C=0) = 0.18 + 0.12 = 0.30
            (0, 1): 0.00, # For C = 1:  P(X=0,C=1) = 0 + 0 = 0
            (1, 0): 0.20, # P(X=1,C=0) = 0.12 + 0.08 = 0.20
            (1, 1): 0.50  # P(X=1,C=1) = 0 + 0.50 = 0.50
        }  # P(X=x, C=c)

        self.Y_C = {
            (0, 0): 0.30, # For C = 0:  P(Y=0,C=0) = 0.18 + 0.12 = 0.30
            (0, 1): 0.00, # For C = 1:  P(Y=0,C=1) = 0 + 0 = 0
            (1, 0): 0.20, # P(Y=1,C=0) = 0.12 + 0.08 = 0.20
            (1, 1): 0.50  # P(Y=1,C=1) = 0 + 0.50 = 0.50
        }  # P(Y=y, C=c)

        self.X_Y_C = {
            (0, 0, 0): 0.18, # P(X=0,Y=0,C=0) = 0.5 * (0.6 * 0.6) = 0.5·0.36 = 0.18
            (0, 0, 1): 0.00, # P(X=0,Y=0,C=1) = 0.5 * (0.0 * 0.0) = 0.5·0.0 = 0.0
            (0, 1, 0): 0.12, # P(X=0,Y=1,C=0) = 0.5 * (0.6 * 0.4) = 0.5·0.24 = 0.12
            (0, 1, 1): 0.00, # P(X=0,Y=1,C=1) = 0.5 * (0.0 * 1.0) = 0.5·0.0 = 0.0
            (1, 0, 0): 0.12, # P(X=1,Y=0,C=0) = 0.5 * (0.4 * 0.6) = 0.5·0.24 = 0.12
            (1, 0, 1): 0.00, # P(X=1,Y=0,C=1) = 0.5 * (1.0 * 0.0) = 0.5·0.0 = 0.0
            (1, 1, 0): 0.08, # P(X=1,Y=1,C=0) = 0.5 * (0.4 * 0.4) = 0.5·0.16 = 0.08
            (1, 1, 1): 0.50, # P(X=1,Y10,C=1) = 0.5 * (1.0 * 1.0) = 0.5·1 = 0.5
        }  # P(X=x, Y=y, C=c)

    def is_X_Y_dependent(self):
        """
        return True iff X and Y are depndendent
        """
        X = self.X
        Y = self.Y
        X_Y = self.X_Y
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################    
        
        # check for any (x,y) if P(X=x, Y=y) != P(X=x)*P(Y=y):
        for x in [0, 1]:
            for y in [0, 1]:
                if not np.isclose(X_Y[(x, y)], X[x] * Y[y]):
                    return True  # we found marginal dependence
                
        # If we never returned True, then X and Y would be marginally independent.
        return False        

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def is_X_Y_given_C_independent(self):
        """
        return True iff X_given_C and Y_given_C are indepndendent
        """
        X = self.X
        Y = self.Y
        C = self.C
        X_C = self.X_C
        Y_C = self.Y_C
        X_Y_C = self.X_Y_C
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        for c in [0, 1]:
            for x in [0, 1]:
                for y in [0, 1]:
                    lhs = X_Y_C[(x, y, c)]/C[c]
                    # as p(x|c) = P(X=x, C=c)/P(C=c), we can calc it as:
                    p_x_given_c = X_C[(x, c)] / C[c]
                    p_y_given_c = Y_C[(y, c)] / C[c]
                    rhs = p_x_given_c * p_y_given_c
                    if not np.isclose(lhs, rhs):
                        return False  # violates the conditional independence
        return True
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################


def normal_pdf(x, mean, std):
    """
    Calculate normal desnity function for a given x, mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - std:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mean and std for the given x.    
    """
    p = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    
    #converting x into a np arr to handle both scalars and arrays
    x_arr=np.asarray(x,dtype=float)

    # normalization constant: 1/sqrt(2*pi*std^2)
    denomenator=std*np.sqrt(2.0*np.pi)

    #exponent term- exp(-(x-mean)^2 / (2*std^2))
    exponent=-0.5 * ((x_arr-mean)/std)**2

    p=(1/denomenator)*np.exp(exponent)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return p

class NaiveNormalClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulates information on the feature-specific
        class conditional distributions for a given class label.
        Each of these distributions is a univariate normal distribution with
        separate parameters (mean and std).
        These distributions are fit to specified training data.
        
        Input
        - dataset: The training dataset as a 2d numpy array, assuming the class label is the last column
        - class_value : The class label to calculate the class conditionals for.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        # Split features vs. labels
        X = dataset[:, :-1]   # shape = (N, d)
        Y = dataset[:, -1]    # shape = (N,)

        self.class_value = class_value

        mask = (Y == class_value)
        X_given_y = X[mask]      # shape = (N_y, d)
        N_y = X_given_y.shape[0]
        N_total = dataset.shape[0]

        # PRIOR: P(Y = class_label) = N_y / N_total
        self._prior = float(N_y) / float(N_total)

        # num of features (d)
        self._num_features = X.shape[1]

        # If no training point has that label, then N_y = 0, so neither mean nor sigma is defined.
        if N_y == 0:
           raise ValueError(f"No training examples with value {class_value!r}.")
        
        # For each feature i = 0,..., d-1, we compute MLE mean and (population) standard deviation:
        self._means = X_given_y.mean(axis=0)       # shape = (d,)
        self._stds = X_given_y.std(axis=0, ddof=0) # shape = (d,)

        # If theres a sigma_i that happens to be zero (all training values in that feature are identical),
        # we assign a small positive num to avoide deviding by 0

        zero_std_mask = (self._stds == 0)
        if np.any(zero_std_mask):
            self._stds[zero_std_mask] = 1e-8


        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    
    def get_prior(self):
        """
        Returns the prior porbability of the class, as computed from the training data.
        """
        prior = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        prior=self._prior
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance given the class label according to
        the feature-specific classc conditionals fitted to the training data.
        """
        likelihood = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        x_arr=np.asarray(x,dtype=float)
        if x_arr.shape[0] != self._num_features:
            raise ValueError(f"Expected a {self._num_features}-dimensional input, got shape {x_arr.shape}.")

        # calc the univariate Gaussian PDF for each feature:
        per_feature_pdf = normal_pdf(x_arr, self._means, self._stds)  # shape = (d,)
        likelihood = np.prod(per_feature_pdf)  # scalar

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return likelihood
    
    def get_instance_joint_prob(self, x):
        """
        Returns the joint probability of the input instance (x) and the class label.
        """
        joint_prob = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        prior=self.get_prior()
        liklyhood=self.get_instance_likelihood(x)
        joint_prob=prior*liklyhood 
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return joint_prob

class MAPClassifier():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum a posteriori classifier. 
        This class holds a ClassDistribution object (either NaiveNormal or MultiNormal)
        for each of the two class labels (0 and 1). 
        Using these objects it predicts class labels for input instances using the MAP rule.
    
        Input
            - ccd0 : A ClassDistribution object for class label 0.
            - ccd1 : A ClassDistribution object for class label 1.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.ccd0=ccd0
        self.ccd1=ccd1
        
        #validating integrity 
        if not (hasattr(self.ccd0, "class_label") and hasattr(self.ccd1, "class_label")):
            raise ValueError("Each input must have a .class_label attribute.")
        labels={self.ccd0.class_label, self.ccd1.class_label}
        if labels != {0, 1}:
            raise ValueError(f"Expected class_label 0 and 1, got {labels}.")

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        # calc joint P(Y=0, X=x) and P(Y=1, X=x)
        joint0 = self.ccd0.get_instance_joint_prob(x)
        joint1 = self.ccd1.get_instance_joint_prob(x)

        pred=0 if joint0>joint1 else 1
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return pred

    
def multi_normal_pdf(x, mean, cov):
    """
    Calculate multivariate normal desnity function under specified mean vector
    and covariance matrix for a given x.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean vector of the distribution.
    - cov:  The covariance matrix of the distribution.
 
    Returns the normal distribution pdf according to the given mean and var for the given x.    
    """
    pdf = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    x_arr = np.asarray(x, dtype=float)
    mu_arr = np.asarray(mean, dtype=float)
    cov_mat = np.asarray(cov, dtype=float)

    #validate dimentions
    d = mu_arr.shape[0]
    if x_arr.shape[0] != d:
        raise ValueError(f"Dimension mismatch: x has length {x_arr.shape[0]}, mean has length {d}.")
    if cov_mat.shape != (d, d):
        raise ValueError(f"Covariance matrix must be {d}x{d}, but got {cov_mat.shape}.")

    #calc determinant
    det_cov = np.linalg.det(cov_mat)
    if det_cov <= 0:
        raise ValueError("Covariance matrix must be positive definite (determinant > 0).")

    #calc inverse of covar matrix
    inv_cov = np.linalg.inv(cov_mat)

    # normalization constant: (2π)^(-d/2) * |cov|^(-1/2)
    norm_const = (2.0 * np.pi) ** (-d / 2) * det_cov ** (-0.5)

    #exponent: −½ (x−mean)^T cov^(-1) (x−mean)
    diff = x_arr - mu_arr
    exponent = -0.5 * diff.dot(inv_cov).dot(diff)

    pdf = norm_const * np.exp(exponent)
    pdf=float(pdf)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pdf

class MultiNormalClassDistribution():

    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the multivariate normal distribution
        representing the class conditional distribution for a given class label.
        The mean and cov matrix should be computed from a given training data set
        (You can use the numpy function np.cov to compute the sample covarianve matrix).
        
        Input
        - dataset: The dataset as a numpy array
        - class_value : The class label to calculate the parameters for.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        pass
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        
    def get_prior(self):
        """
        Returns the prior porbability of the class, as computed from the training data.
        """
        prior = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        pass
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance given the class label according to
        the multivariate classc conditionals fitted to the training data.
        """
        likelihood = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        pass
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return likelihood
    
    def get_instance_joint_prob(self, x):
        """
        Returns the joint probability of the input instance (x) and the class label.
        """
        joint_prob = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        pass
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return joint_prob



def compute_accuracy(test_set, map_classifier):
    """
    Compute the accuracy of a given MAP classifier on a given test set.
    
    Input
        - test_set: The test data (Numpy array) on which to compute the accuracy. The class label is the last column
        - map_classifier : A MAPClassifier object that predicits the class label from a feature vector.
        
    Ouput
        - Accuracy = #Correctly Classified / number of test samples
    """
    acc = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return acc

class DiscreteNBClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the probabilites for a discrete naive bayes
        class conditional distribution for a given class label.
        The probabilites of each feature-specific class conditional
        are computed with laplace smoothing.
        
        Input
        - dataset: The dataset as a numpy array
        - class_value : The class label to calculate the probabilities for.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        pass
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    
    def get_prior(self):
        """
        Returns the prior porbability of the class, as computed from the training data.
        """
        prior = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        pass
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance given the class label according to
        the product of feature-specific discrete class conidtionals fitted to the training data.
        """
        likelihood = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        pass
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return likelihood
    
    def get_instance_joint_prob(self, x):
        """
        Returns the joint probability of the input instance (x) and the class label.
        """
        joint_prob = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        pass
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return joint_prob
