###### Your ID ######
# ID1: 207002544
# ID2: 206860967
#####################

# imports 
import numpy as np
import pandas as pd

def preprocess(X,y):
    """
    Perform Standardization on the features and true labels.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - X: The Standardized input data.
    - y: The Standardized true labels.
    """
    ###########################################################################
    # TODO: Implement the normalization function.                             #
    ###########################################################################

    # Edge cases
    validateInput(X,y)

    X = arg_reshape(X)[0] # make sure it is 2D

    #check if bias was already applied. If so - normalize only from second column
    has_bias = X.shape[1] > 0 and np.all(X[:, 0] == 1)

    #normalize X
    if has_bias:
        bias = X[:, [0]]
        features = X[:, 1:]

        means, stds = features.mean(axis=0), features.std(axis=0)
        features = (features - means)
        features = np.divide(features, stds, out=np.zeros_like(features), where=stds != 0)

        X = np.hstack((bias, features))

    else:
        means, stds = X.mean(axis=0), X.std(axis=0)
        X = X - means
        X = np.divide(X, stds, out=np.zeros_like(X), where=stds != 0)

    #Normalize y
    mean_y, std_y = np.mean(y), np.std(y)

    if np.all(y == y[0]):  # all elements are equal
        y = np.zeros_like(y)
    else:
        y = (y - mean_y) / std_y

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return X, y

def apply_bias_trick(X):
    """
    Applies the bias trick to the input data.

    Input:
    - X: Input data (n instances over p features).

    Returns:
    - X: Input data with an additional column of ones in the
        zeroth position (n instances over p+1).x
    """
    ###########################################################################
    # TODO: Implement the bias trick by adding a column of ones to the data.                             #
    ###########################################################################
    if X is None or X.size == 0:
        raise ValueError('X cannot be None or empty.')

    #make it 2D in case it is 1D
    X = arg_reshape(X)[0]

    #check if bias was already applied
    if np.all(X[:, 0] == 1):
        return X

    ones = np.ones((X.shape[0], 1))
    X = np.hstack((ones, X))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return X

def compute_loss(X, y, theta):
    """
    Computes the average squared difference between an observation's actual and
    predicted values for linear regression.  

    Input:
    - X: Input data (n instances over p features).
    - y: True labels (n instances).
    - theta: the parameters (weights) of the model being learned.

    Returns:
    - J: the loss associated with the current set of parameters (single number).
    """
    
    J = 0  # We use J for the loss.
    ###########################################################################
    # TODO: Implement the MSE loss function.                                  #
    ###########################################################################
    validateInput(X, y, theta)
    n = X.shape[0]

    #compute y_hats
    prediction_vector = np.dot(X, theta) 

    #compute loss per instance
    prediction_vector = prediction_vector - y 
    prediction_vector = prediction_vector ** 2

    #compute mean
    J = (1 / (2 * n)) * np.sum(prediction_vector) 

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return J

# TODO: understand if preproccess is needed
def gradient_descent(X, y, theta, eta, num_iters):
    """
    Learn the parameters of the model using gradient descent using 
    the training set. Gradient descent is an optimization algorithm 
    used to minimize some (loss) function by iteratively moving in 
    the direction of steepest descent as defined by the negative of 
    the gradient. We use gradient descent to update the parameters
    (weights) of our model.

    Input:
    - X: Input data (n instances over p features).
    - y: True labels (n instances).
    - theta: The parameters (weights) of the model being learned.
    - eta: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    theta = theta.copy() # optional: theta outside the function will not change
    J_history = [] # Use a python list to save the loss value in every iteration
    ###########################################################################
    # TODO: Implement the gradient descent optimization algorithm.            #
    ###########################################################################
    validateInput(X, y, theta)

    if num_iters <= 0 or eta <= 0:
        raise ValueError('eta and num_iters must be a positive numbers.')
    n = X.shape[0]

    for iter in range(num_iters):
        # compute gradient
        predictions = np.dot(X, theta)
        errors = predictions - y
        gradient = (1 / n) * X.T.dot(errors)
        # gradient = (1 / n) * X.T.dot(errors).sum(axis=0)

        if np.any(np.isinf(gradient)) or np.any(np.isnan(gradient)):
            print(f"Warning: Unstable gradient encountered at iteration {iter}")
            J_history.append(float('inf'))  # Record that gradient diverged
            break

        #update in the counter direction of the gradient
        theta =  theta - eta * gradient

        max_value = 1e10
        # Clip theta values if they become too extreme
        theta = np.clip(theta, -max_value, max_value)

        loss = compute_loss(X, y, theta)

        if np.isinf(loss) or np.isnan(loss):
            J_history.append(float('inf'))
            break

        #log losses
        J_history.append(loss)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return theta, J_history

def compute_pinv(X, y):
    """
    Compute the optimal values of the parameters using the pseudoinverse
    approach as you saw in class using the training set.

    #########################################
    #### Note: DO NOT USE np.linalg.pinv ####
    #########################################

    Input:
    - X: Input data (n instances over p features).
    - y: True labels (n instances).

    Returns:
    - pinv_theta: The optimal parameters of your model.
    """
    
    pinv_theta = []
    ###########################################################################
    # TODO: Implement the pseudoinverse algorithm.                            #
    ###########################################################################
    validateInput(X, y)
    X, y = preprocess(X, y)

    try:
        pseudo_inverse = np.linalg.inv(np.dot(X.T, X))
    except:
        raise np.linalg.LinAlgError('X.T * X is not invertible and therefore the pseudoinverse is not available.')

    pinv_theta = np.dot(pseudo_inverse, (np.dot(X.T, y)))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pinv_theta

def gradient_descent_stop_condition(X, y, theta, eta, max_iter, epsilon=1e-8):
    """
    Learn the parameters of your model using the training set, but stop 
    the learning process once the improvement of the loss value is smaller 
    than epsilon. This function is very similar to the gradient descent 
    function you already implemented.

    Input:
    - X: Input data (n instances over p features).
    - y: True labels (n instances).
    - theta: The parameters (weights) of the model being learned.
    - eta: The learning rate of your model.
    - max_iter: The maximum number of iterations.
    - epsilon: The threshold for the improvement of the loss value.
    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    theta = theta.copy() # optional: theta outside the function will not change
    J_history = [] # Use a python list to save the loss value in every iteration
    ###########################################################################
    # TODO: Implement the gradient descent with stop condition optimization algorithm.  #
    ###########################################################################
    validateInput(X, y, theta)
    if max_iter <= 0 or eta <= 0 or epsilon <= 0:
        raise ValueError('eta, max_iter and epsilon must be a positive numbers.')

    n = X.shape[0]

    prev_J = compute_loss(X, y, theta)
    J_history.append(prev_J)

    for iter in range(max_iter):
        # compute gradient
        predictions = np.dot(X, theta)
        errors = predictions - y
        gradient = (1 / n) * X.T.dot(errors)

        # Check if gradient contains extreme values
        if np.any(np.isinf(gradient)) or np.any(np.isnan(gradient)):
            print(f"Warning: Unstable gradient encountered at iteration {iter}")
            J_history.append(float('inf'))  # Record that gradient diverged
            break

        # update in the counter direction of the gradient
        theta = theta - eta * gradient

        max_value = 1e10
        # Clip theta values if they become too extreme
        theta = np.clip(theta, -max_value, max_value)

        cur_J = compute_loss(X, y, theta)

        # avoid performing operations on 'inf' or 'nan'
        if np.isinf(cur_J) or np.isnan(cur_J):
            J_history.append(float('inf'))
            break

        J_history.append(cur_J)

        # early exit if improvement is insignificant
        if abs(prev_J - cur_J) < epsilon:
            break
        # log losses
        prev_J = cur_J

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return theta, J_history

def find_best_learning_rate(X_train, y_train, X_val, y_val, iterations):
    """
    Iterate over the provided values of eta and train a model using 
    the training dataset. Maintain a python dictionary with eta as the 
    key and the loss on the validation set as the value.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the training and validation data
    - iterations: maximum number of iterations

    Returns:
    - eta_dict: A python dictionary - {eta_value : validation_loss}
    """
    
    etas = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3]
    eta_dict = {} # {eta_value: validation_loss}
    ###########################################################################
    # TODO: Implement the function and find the best eta value.             #
    ###########################################################################
    validateInput(X_train, y_train)
    validateInput(X_val, y_val)
    if iterations <= 0:
        raise ValueError('iterations must be positive.')

    for eta in etas:
        #gradient descent parameters: X, y, theta, eta, num_iters, default epsilon = 1e-8
        theta, J_history = gradient_descent(X_train, y_train, np.random.randn(X_train.shape[1]), eta, iterations)
        eta_dict[eta] = compute_loss(X_val, y_val, theta)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return eta_dict

def forward_feature_selection(X_train, y_train, X_val, y_val, best_eta, iterations):
    """
    Forward j selection is a greedy, iterative algorithm used to
    select the most relevant features for a predictive model. The objective 
    of this algorithm is to improve the model's performance by identifying 
    and using only the most relevant features, potentially reducing overfitting, 
    improving accuracy, and reducing computational cost.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the input data without bias trick
    - best_eta: the best learning rate previously obtained
    - iterations: maximum number of iterations for gradient descent

    Returns:
    - selected_features: A list of selected top 5 j indices
    """
    selected_features = []
    #####c######################################################################
    # TODO: Implement the function and find the best eta value.             #
    ###########################################################################
    X_train, y_train = preprocess(X_train, y_train)
    X_val, y_val = preprocess(X_val, y_val)
    X_train = apply_bias_trick(X_train)
    X_val = apply_bias_trick(X_val)
    M_complement = set(range(1, X_train.shape[1]))
    j_star = None

    while len(selected_features) < 5 and M_complement:
        min_loss = float('inf')

        for j in M_complement:
            #train on M ∪ j (make sure not to forget to include the bias column!)
            slice = [0] + selected_features + [j]
            cur_train = X_train[:, slice]
            theta, _ = gradient_descent_stop_condition(cur_train, y_train, (np.random.rand(cur_train.shape[1]) * 0.01), best_eta, iterations)
            cur_loss = compute_loss(X_val[:, slice], y_val, theta)

            if cur_loss < min_loss:
                min_loss = cur_loss
                j_star = j


        #M = M ∪ j_star
        if j_star is not None:
            selected_features.append(j_star)
            M_complement = M_complement.difference({j_star})

        else:
            # No feature was successfully computed.
            print("Warning: No feature could be successfully evaluated. Stopping selection.")
            break

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return selected_features

def create_square_features(df):
    """
    Create square features for the input data.

    Input:
    - df: Input data (n instances over p features) as a dataframe.

    Returns:
    - df_poly: The input data with polynomial features added as a dataframe
               with appropriate feature names
    """

    df_poly = df.copy()
    ###########################################################################
    # TODO: Implement the function to add polynomial features                 #
    ###########################################################################
    if df is None or len(df_poly.columns) == 0:
        return df_poly

    df_poly = df.copy()
    features = list(df.columns)

    # Dictionary to hold new features
    new_features = {}

    for i in range(len(features)):
        featureI = features[i]
        new_features[featureI + "^2"] = df[featureI] ** 2

        for j in range(i + 1, len(features)):
            featureJ = features[j]
            new_features[featureI + "*" + featureJ] = df[featureI] * df[featureJ]

    # Convert dictionary to DataFrame
    new_features_df = pd.DataFrame(new_features)

    # Concatenate all at once
    df_poly = pd.concat([df_poly, new_features_df], axis=1)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return df_poly

def arg_reshape(*args):
    """
    Handle cases where arrays are given as 1D arrays by reshaping them to 2D.

    Args:
        *args: Any number of numpy arrays to reshape if needed.

    Returns:
        reshaped_args: A tuple of reshaped arrays, each as a 2D array
                      (each row represents one instance).
    """
    reshaped_args = []

    for arg in args:
        if arg.ndim == 1:
            # Reshape 1D array to a column vector
            reshaped_args.append(arg.reshape(-1, 1))
        else:
            # Keep array as is if it's already 2D or higher
            reshaped_args.append(arg)

    # Return as tuple (to match the original function's return style)
    return tuple(reshaped_args)


def validateInput(X, y, theta=None):
    """
    Validate the input data for the functions.
    Input:
    - X: Input data (n instances over p features).
    - y: True labels (n instances).
    - theta: The parameters (weights) of the model being learned.
    """

    # Check if X and y are numpy arrays
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise ValueError("X and y must be numpy arrays.")

    # Check if X and y are empty
    if X.size == 0 or y.size == 0:
        raise ValueError("X and y cannot be empty.")

    # Check if X and y have the same number of rows
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows.")

    if theta is not None:
        # Check if theta is a numpy array
        if not isinstance(theta, np.ndarray):
            raise ValueError("Theta must be a numpy array. Theta type is ", type(theta))

        # Check if theta is empty
        if theta.size == 0:
            raise ValueError("Theta cannot be empty.")

        # if X already has bias trick applied, theta must have the same number of elements as features in X
        if X.shape[1] > 0 and np.all(X[:, 0] == 1):
            if theta.shape[0] != X.shape[1]:
                raise ValueError("Theta must have the same number of elements as features in X. X.shape, theta.shape", X.shape, theta.shape)

                # if X does not have bias trick applied, theta must have the same number of elements as features in X + 1
        elif X.shape[1] + 1 != theta.shape[0]:
            raise ValueError("X and theta must have compatible dimensions. X.shape, theta.shape", X.shape, theta.shape)