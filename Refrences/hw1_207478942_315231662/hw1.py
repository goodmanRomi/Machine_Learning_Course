###### Your ID ######
# ID1: 207478942
# ID2: 315231662
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

    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    y = (y - np.mean(y)) / np.std(y)
    
    return X, y

def apply_bias_trick(X):
    """
    Applies the bias trick to the input data.

    Input:
    - X: Input data (n instances over p features).

    Returns:
    - X: Input data with an additional column of ones in the
        zeroth position (n instances over p+1).
    """
    if(X.ndim == 1):
        X = X.reshape(-1, 1) #n, p
    ones_c = np.ones((X.shape[0], 1))
    X = np.hstack((ones_c, X))

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
    mult = np.dot(X, theta)
    error = mult - y
    pow = error ** 2

    n = X.shape[0]

    J = (1 / (2 * n)) * np.sum(pow)
    
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
    gradient = 0  # We use J for the loss.
    n = X.shape[0]

    for i in range(num_iters):

        mult = np.dot(X, theta) #matrix multiplication for each xi
        error = mult - y #error calculation

        gradient = (1 / n) * np.dot(X.T, error) #gradient calculation  
        
        theta = theta - (eta * gradient) #update theta

        loss = (1 / (2 * n)) * np.sum(error ** 2) #loss calculation
        J_history.append(loss)    
        
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
        
    x_transpose_x = np.dot(X.T, X) #computing (X^T X)
    
    res_inv = np.linalg.inv(x_transpose_x) #computing the inverse of (X^T X)

    mult = np.dot(res_inv, X.T)

    pinv_theta = np.dot(mult, y) #computing the final result (X^T X)^(-1) X^T y
    
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

    n = X.shape[0]
    #first loss
    mult = np.dot(X, theta) #matrix multiplication for each xi
    error = mult - y #error calculation
    gradient = (1 / n) * np.dot(X.T, error) #gradient calculation         
    theta = theta - (eta * gradient) #update theta
    loss = (1 / (2 * n)) * np.sum(error ** 2) #loss calculation
    J_history.append(loss)
    improvement = loss  
    
    while improvement > epsilon and max_iter > 0:
        mult = np.dot(X, theta) #matrix multiplication for each xi
        error = mult - y #error calculation
        gradient = (1 / n) * np.dot(X.T, error) #gradient calculation         
        theta = theta - (eta * gradient) #update theta
        loss = (1 / (2 * n)) * np.sum(error ** 2) #loss calculation
        J_history.append(loss)  
        improvement = abs(loss - J_history[-2]) #improvment calculation
        max_iter-= 1 #decrease max_iter
    
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

    for eta in etas:
        theta = np.zeros(X_train.shape[1]) #initialize the theta
        theta, _ = gradient_descent_stop_condition(X_train, y_train, theta, eta, iterations) 
        val_loss = compute_loss(X_val, y_val, theta)
        eta_dict[eta] = val_loss

    return eta_dict

def forward_feature_selection(X_train, y_train, X_val, y_val, best_eta, iterations):
    """
    Forward feature selectiont is a greedy, iterative algorithm used to 
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
    temp_features = []
    features = range(X_train.shape[1])

    while len(selected_features) < 5:
        best_feature = None
        max_val_loss = float('inf')

        for feature in features:
            if feature not in selected_features:
                temp_features = selected_features + [feature]

                #apply bias to new features and slice the data
                X_train_temp = X_train[:,temp_features]
                X_train_temp = apply_bias_trick(X_train_temp)
                X_val_temp = X_val[:,temp_features]
                X_val_temp = apply_bias_trick(X_val_temp)

                # computte loss for the current feature
                theta = np.zeros(X_train_temp.shape[1])
                theta, _ = gradient_descent(X_train_temp, y_train, theta, best_eta, iterations) 
                val_loss = compute_loss(X_val_temp, y_val, theta)

                if val_loss < max_val_loss:
                    max_val_loss = val_loss
                    best_feature = feature
                    
        # append the best feature to the selected features
        selected_features.append(best_feature)
        
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
    original_cols = list(df.columns)  # original columns

    #create square features + append to the original dataframe
    for col in original_cols:
        #square of each feature
        df_poly[col + '^2'] = df_poly[col] ** 2 

        #product of all features
    for i, col1 in enumerate(original_cols):
        for j in range(i + 1, len(original_cols)):
            col2 = original_cols[j]
            df_poly[col1 + '*' + col2] = df[col1] * df[col2]      

    return df_poly