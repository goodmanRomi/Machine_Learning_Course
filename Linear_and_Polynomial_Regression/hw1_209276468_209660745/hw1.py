###### Your ID ######
# ID1: 209276468
# ID2: 209660745
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
    
    X_standardized = (X-np.mean(X,axis=0))/np.std(X,axis=0) #X.shape=(n_samples,n_features) so axis=0 means to compute the means and std of feature coloums
    y_standardized = (y-np.mean(y))/np.std(y) #y.shape=(n-samples,) 
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    
    return X_standardized, y_standardized

def apply_bias_trick(X):
    """
    Applies the bias trick to the input data.

    Input:
    - X: Input data (n instances over p features).

    Returns:
    - X: Input data with an additional column of ones in the
        zeroth position (n instances over p+1).
    """
    ###########################################################################
    # TODO: Implement the bias trick by adding a column of ones to the data.                             #
    ###########################################################################
    if X.ndim == 1: # cases where the array is given as 1D arrays - so we reshape to 2D
        X = X.reshape(-1,1)
    bias_trick = np.ones((X.shape[0],1))
    X = np.hstack((bias_trick,X))

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
    n = X.shape[0]
    y_hat = np.dot(X,theta) # equivalent X @ theta
    
    J = (1/(2*n))*np.sum((y_hat-y)**2)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return J

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
    n = X.shape[0]
    t = 0
    for t in range(num_iters):
        y_hat = np.dot(X,theta) #matrix mult for each xi
        gradient = (1/n)*np.dot(X.T,(y_hat-y)) #calculating the gradient according to formula
        J_history.append(compute_loss(X,y,theta)) # save the cost of the value in every iteration
        theta -= eta * gradient

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
    X_T_X = X.T @ X #(X^T dot product X)
    pseudoinverse = np.linalg.inv(X_T_X) #inverse of X^T dot product X
    X_T_Y = X.T @ y #X^T*y
    pinv_theta= pseudoinverse @ X_T_Y  #computing theta star, pinv theta = X^T*y dot product inverse of X^T dot product X

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
    n = X.shape[0]
    for t in range(max_iter):
        y_hat = np.dot(X,theta)
        gradient = (1/n)*np.dot(X.T,(y_hat-y))
        theta-=eta*gradient
        J=compute_loss(X,y,theta)
        
        if J>1e10: # if the cost is too high, overflow gaurd
            break

        J_history.append(J)

        if t>0 and abs(J_history[-2]-J_history[-1]) < epsilon: # if the improvment is smaller then the threshold '1e-8'
            break
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
    
    for eta in etas:
        np.random.seed(42)
        theta = np.random.random(size=X_train.shape[1])*0.01
        theta = gradient_descent_stop_condition(X_train,y_train,theta,eta,iterations)[0]
        validation_loss = compute_loss(X_val,y_val, theta)
        eta_dict[eta] = validation_loss
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return eta_dict

def forward_feature_selection(X_train, y_train, X_val, y_val, best_eta, iterations):
    """
    Forward feature selection is a greedy, iterative algorithm used to 
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
    - selected_features: A list of selected top 5 feature indices
    """
    selected_features = []
    #####c######################################################################
    # TODO: Implement the function and find the best eta value.             #
    ###########################################################################
    
    n_features = X_train.shape[1]
    feature_remain_to_check = list(range(X_train.shape[1]))

    #X_train,y_train=preprocess(X_train,y_train)
    #X_val,y_val=preprocess(X_val,y_val)

    X_train=apply_bias_trick(X_train)
    X_val=apply_bias_trick(X_val)
    

    while len(selected_features)<5 and feature_remain_to_check: 
        best_mse = float("inf")
        best_feature = None

        for feature in feature_remain_to_check:
            curr_coloums=[0]+ selected_features + [feature]
            curr_feature_train = X_train[:, curr_coloums] # bias + selected + candidate 'feature'
            theta = gradient_descent_stop_condition(curr_feature_train,y_train, np.random.random(size=curr_feature_train.shape[1])*0.01 ,best_eta, iterations, epsilon=1e-8)[0]    
            x_val_sub=X_val[:, curr_coloums]
            mse = compute_loss(x_val_sub,y_val,theta)
            
            if mse < best_mse:
                best_mse = mse  #feature that gave me the minimal mse until this point
                best_feature = feature

        if best_feature is not None:
            selected_features.append(best_feature)
            feature_remain_to_check.remove(best_feature)
        else:
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
    if df.empty or df is None: # if the dataframe is empty
        return df 
    
    df_poly = df.copy()
    original_columns = list(df.columns)

    # Dictionary to hold new features
    new_features = {}

    for i in range(len(original_columns)):
        feature_i = original_columns[i]
        new_features[f"{feature_i}^2"] = df[feature_i] ** 2
        
        for j in range(i + 1, len(original_columns)):
            feature_j = original_columns[j]
            new_features[feature_i + "*" + feature_j] = df[feature_i] * df[feature_j]

    # Convert dictionary to DataFrame
    new_features_df = pd.DataFrame(new_features)

    # Concatenate all at once
    df_poly = pd.concat([df_poly, new_features_df], axis=1)
    ###########################################################################
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return df_poly