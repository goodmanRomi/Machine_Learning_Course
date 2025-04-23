###### Your ID ######
# ID: 318416237
#####################

# imports 
import numpy as np
import pandas as pd

def preprocess(X,y):
    """
    Perform mean normalization on the features and true labels.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - X: The mean normalized inputs.
    - y: The mean normalized labels.
    """
    ###########################################################################
    # TODO: Implement the normalization function.   
    
    # Normalize the features
    # Calculate the average of each feature vector as a avrage 
    X_normalized = (X - X.mean(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    y_normalized = (y - np.mean(y)) / (np.max(y) - np.min(y))

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return X_normalized, y_normalized

def apply_bias_trick(X):
    """
    Applies the bias trick to the input data.

    Input:
    - X: Input data (m instances over n features).

    Returns:
    - X: Input data with an additional column of ones in the
        zeroth position (m instances over n+1 features).
    """
    ###########################################################################
    # TODO: Implement the bias trick by adding a column of ones to the data.  
    one = np.ones(len(X)) # create a column of ones
    X = np.column_stack((one, X)) # add the column of ones to the data

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return X

def compute_cost(X, y, theta):
    """
    Computes the average squared difference between an observation's actual and
    predicted values for linear regression.  

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: the parameters (weights) of the model being learned.

    Returns:
    - J: the cost associated with the current set of parameters (single number).
    """
    
    J = 0  # We use J for the cost.
    ###########################################################################
    # TODO: Implement the MSE cost function.                                  #
    m = len(y) 
    h = np.dot(X,theta) # hypothesis
    e = h - y # error
    squered_errors = e**2
    J = (1/ (2*m) * np.sum(squered_errors)) # cost function
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return J

def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of the model using gradient descent using 
    the training set. Gradient descent is an optimization algorithm 
    used to minimize some (loss) function by iteratively moving in 
    the direction of steepest descent as defined by the negative of 
    the gradient. We use gradient descent to update the parameters
    (weights) of our model.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    theta = theta.copy() # optional: theta outside the function will not change
    J_history = [] # Use a python list to save the cost value in every iteration
    ###########################################################################
    # TODO: Implement the gradient descent optimization algorithm.            #
    m = len(y)
    for i in range(num_iters):
        h = np.dot(X,theta) # hypothesis
        e = h - y # error
        gradient = np.dot(X.T, e) / m
        theta -= alpha * gradient
        J_history.append(compute_cost(X,y,theta)) # save the cost of the value in every iteration

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
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - pinv_theta: The optimal parameters of your model.
    """
    
    pinv_theta = []
    ###########################################################################
    # TODO: Implement the pseudoinverse algorithm.                            #

    pinv_theta = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y) # pinv theta = (X^TX)^-1*X^T*y
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pinv_theta

def efficient_gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of your model using the training set, but stop 
    the learning process once the improvement of the loss value is smaller 
    than 1e-8. This function is very similar to the gradient descent 
    function you already implemented.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    theta = theta.copy() # optional: theta outside the function will not change
    J_history = [] # Use a python list to save the cost value in every iteration
    ###########################################################################
    # TODO: Implement the efficient gradient descent optimization algorithm.  #
    ###########################################################################
    m = len(y)
    for i in range(num_iters): 
        h = np.dot(X,theta) # hypothesis
        e = h - y # error
        gradient = np.dot(X.T, e) / m # calculate the gradient
        theta -= alpha * gradient # update the theta
        J = compute_cost(X,y,theta) # calculate the cost

        if J > 1e10: # if the cost is too high
            break

        J_history.append(compute_cost(X,y,theta)) # save the cost of the value in every iteration

        if i > 0 and abs(J_history[-2] - J_history[-1]) < 1e-8: # if the improvment is smaller then the threshold '1e-8'
            break

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return theta, J_history

def find_best_alpha(X_train, y_train, X_val, y_val, iterations):
    """
    Iterate over the provided values of alpha and train a model using 
    the training dataset. maintain a python dictionary with alpha as the 
    key and the loss on the validation set as the value.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the training and validation data
    - iterations: maximum number of iterations

    Returns:
    - alpha_dict: A python dictionary - {alpha_value : validation_loss}
    """
    
    alphas = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3]
    alpha_dict = {} # {alpha_value: validation_loss}
    ###########################################################################
    # TODO: Implement the function and find the best alpha value.             #
    ###########################################################################

    for i in alphas: # iterate over the alphas
        
        theta = np.ones(X_train.shape[1]) # initialize the theta
        theta = efficient_gradient_descent(X_train, y_train, theta, i, iterations)[0] # extract only the theta
        loss_of_validation = compute_cost(X_val,y_val,theta) # calculate the loss of the validation

        alpha_dict[i]=loss_of_validation # add the alpha and the loss to the dictionary


    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return alpha_dict

def forward_feature_selection(X_train, y_train, X_val, y_val, best_alpha, iterations):
    """
    Forward feature selection is a greedy, iterative algorithm used to 
    select the most relevant features for a predictive model. The objective 
    of this algorithm is to improve the model's performance by identifying 
    and using only the most relevant features, potentially reducing overfitting, 
    improving accuracy, and reducing computational cost.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the input data without bias trick
    - best_alpha: the best learning rate previously obtained
    - iterations: maximum number of iterations for gradient descent

    Returns:
    - selected_features: A list of selected top 5 feature indices
    """
    selected_features = []
    #####c######################################################################
    # TODO: Implement the function and find the best alpha value.             #
    ###########################################################################
    np.random.seed(42) 
    feature_remain_to_check = list(range(X_train.shape[1]))
    previous_loss = float("inf")

    # select the best 5 features
    for i in range (5):
        current_best_feature = None
        best_validation_loss = float("inf")
        theta = np.zeros(i + 2)

        # check all the features
        for feature in feature_remain_to_check:
            current_feature_to_check = selected_features + [feature] 
            selected_X_train = X_train[:, current_feature_to_check]
            selected_X_val = X_val[:, current_feature_to_check] 

            # apply bias trick
            selected_X_train_bias = apply_bias_trick(selected_X_train) 
            selected_X_val_bias = apply_bias_trick(selected_X_val)

            # train the model
            theta, _ = efficient_gradient_descent(selected_X_train_bias, y_train, theta, best_alpha, iterations)
            current_validation_loss = compute_cost(selected_X_val_bias, y_val, theta)

            # check if the current feature is the best
            if current_validation_loss < best_validation_loss:
                best_validation_loss = current_validation_loss
                current_best_feature = feature
        
        # update the selected features
        selected_features.append(current_best_feature)
        feature_remain_to_check.remove(current_best_feature)
  

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return selected_features

def create_square_features(df):
    """
    Create square features for the input data.

    Input:
    - df: Input data (m instances over n features) as a dataframe.

    Returns:
    - df_poly: The input data with polynomial features added as a dataframe
               with appropriate feature names
    """

    df_poly = df.copy()
    ###########################################################################
    # TODO: Implement the function to add polynomial features                 #
    ###########################################################################
    if df.empty: # if the dataframe is empty
        return df 

  
    names_of_features = df.columns
    number_of_features = len(names_of_features)

    for i in range(number_of_features): # iterate over the features

        feature_i = names_of_features[i] 
        df_poly[f'{feature_i}^2'] = df[feature_i] ** 2 # add the square of the feature
    
        for j in range(i + 1, number_of_features): # add the multiplication of the features

            feature_j = names_of_features[j]
            new_column_name = f'{feature_i} * {feature_j}'
            df_poly[new_column_name] = df[feature_i] * df[feature_j]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return df_poly