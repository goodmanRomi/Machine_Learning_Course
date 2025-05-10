import numpy as np
import matplotlib.pyplot as plt

### Chi square table values ###
# The first key is the degree of freedom 
# The second key is the p-value cut-off
# The values are the chi-statistic that you need to use in the pruning

chi_table = {1: {0.5 : 0.45,
             0.25 : 1.32,
             0.1 : 2.71,
             0.05 : 3.84,
             0.0001 : 100000},
         2: {0.5 : 1.39,
             0.25 : 2.77,
             0.1 : 4.60,
             0.05 : 5.99,
             0.0001 : 100000},
         3: {0.5 : 2.37,
             0.25 : 4.11,
             0.1 : 6.25,
             0.05 : 7.82,
             0.0001 : 100000},
         4: {0.5 : 3.36,
             0.25 : 5.38,
             0.1 : 7.78,
             0.05 : 9.49,
             0.0001 : 100000},
         5: {0.5 : 4.35,
             0.25 : 6.63,
             0.1 : 9.24,
             0.05 : 11.07,
             0.0001 : 100000},
         6: {0.5 : 5.35,
             0.25 : 7.84,
             0.1 : 10.64,
             0.05 : 12.59,
             0.0001 : 100000},
         7: {0.5 : 6.35,
             0.25 : 9.04,
             0.1 : 12.01,
             0.05 : 14.07,
             0.0001 : 100000},
         8: {0.5 : 7.34,
             0.25 : 10.22,
             0.1 : 13.36,
             0.05 : 15.51,
             0.0001 : 100000},
         9: {0.5 : 8.34,
             0.25 : 11.39,
             0.1 : 14.68,
             0.05 : 16.92,
             0.0001 : 100000},
         10: {0.5 : 9.34,
              0.25 : 12.55,
              0.1 : 15.99,
              0.05 : 18.31,
              0.0001 : 100000},
         11: {0.5 : 10.34,
              0.25 : 13.7,
              0.1 : 17.27,
              0.05 : 19.68,
              0.0001 : 100000}}

def calc_proportion(data, column_index=0):
    """
    Calculate the proportions of each value in a column.
    
    Parameters:
    -----------
    data : numpy.ndarray
        The dataset as a NumPy array
    column_index : int or None
        The index of the column to analyze
        If None, defaults to the last column (assumed to be the target/class)
    
    Returns:
    --------
    dict
        A dictionary with values as keys and their proportions as values
    """
    # Default to the last column if no column_index is provided
    if column_index is None:
        column_index = -1  # -1 refers to the last column in NumPy indexing

    column_data = data[:, column_index]

    # Get unique values and their counts
    unique_values, counts = np.unique(column_data, return_counts=True)

    # Calculate proportions
    total_samples = len(column_data)
    proportions = {value: count/total_samples for value, count in zip(unique_values, counts)}

    return proportions

def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.
 
    Input:
    - data: any dataset where the last column holds the labels.
 
    Returns:
    - gini: The gini impurity value.
    """
    gini = 0.0
    ###########################################################################
    # TODO: Implement the function.  
    # 1 - sum of all sqaured proportions
    ###########################################################################
    proportions=np.array(list(calc_proportion(data,-1).values()))
    gini=1-np.sum(np.square(proportions))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return gini


def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - entropy: The entropy value.
    """
    entropy = 0.0
    ###########################################################################
    # TODO: Implement the function.
    ###########################################################################
    class_proportions=np.array(list(calc_proportion(data,-1).values()))
    p = class_proportions[class_proportions > 0]
    entropy=-np.sum(p*np.log(p))

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return entropy

class DecisionNode:

    
    def __init__(self, data, impurity_func, feature=-1,depth=0, chi=1, max_depth=1000, gain_ratio=False):
        
        self.data = data # the data instances associated with the node
        self.terminal = False # True iff node is a leaf
        self.feature = feature # column index of feature/attribute used for splitting the node
        self.pred = self.calc_node_pred() # the class prediction associated with the node
        self.depth = depth # the depth of the node
        self.children = [] # the children of the node (array of DecisionNode objects)
        self.children_values = [] # the value associated with each child for the feature used for splitting the node
        self.max_depth = max_depth # the maximum allowed depth of the tree
        self.chi = chi # the P-value cutoff used for chi square pruning
        self.impurity_func = impurity_func # the impurity function to use for measuring goodness of a split
        self.gain_ratio = gain_ratio # True iff GainRatio is used to score features
        self.feature_importance = 0
    
    def calc_node_pred(self):
        """
        Calculate the node's prediction.

        Returns:
        - pred: the prediction of the node
        """
        pred = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        labels=self.data[:,-1]
        values, counts = np.unique(labels, return_counts=True)
        pred=str(values[np.argmax(counts)])

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return pred
        
    def add_child(self, node, val):
        """
        Adds a child node to self.children and updates self.children_values

        This function has no return value
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.children.append(node)
        self.children_values.append(val)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    
    def goodness_of_split(self, feature):
        """
        Calculate the goodness of split of a dataset given a feature and impurity function.

        Input:
        - feature: the feature index the split is being evaluated according to.

        Returns:
        - goodness: the goodness of split
        - groups: a dictionary holding the data after splitting 
                  according to the feature values.
        """
        goodness = 0
        groups = {} # groups[feature_value] = data_subset
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        
        feature_values = np.unique(self.data[:, feature]) # Get all unique values in the feature column
         
        parent_impurity = self.impurity_func(self.data) # Calculate the impurity of the parent node

        total_samples = len(self.data) #total number of samples in this node(count of all data points that have reached this node)
        
        # Calculate weighted sum of children impurities - will be calculating for each subgroup and add all to get the sum
        weighted_child_impurity = 0

        # Split the data based on feature values
        for val in feature_values: 
            subset = self.data[self.data[:, feature] == val] # Create a subset of data where the feature has this value
            groups[val] = subset 

            if len(subset)==0: #skip when the subset is empty
                continue

            weight = len(subset) / total_samples # Calculate weight (proportion of samples in this subset)

            subset_impurity=self.impurity_func(subset) #calculate the impurity of this specific subgroup 
            
            weighted_child_impurity+=weight*subset_impurity #Add weighted impurity to the sum
            
        impurity_reduction=parent_impurity-weighted_child_impurity #formula to calculate infogain
        """
        Information Gain tends to favor features with many unique values. 
        Gain Ratio corrects this bias by normalizing the Information Gain with a term called Split Information.
        The Gain Ratio penalizes features that split the data into many subsets, 
        thus correcting the bias of Information Gain toward such features
        
        """
        #if we are using gain ratio and impurity func is entropy:
        if self.gain_ratio and self.impurity_func==calc_entropy: 
            split_info=0 #calculate the split info
            for val in feature_values:
                subset=groups[val]
                if len(subset)==0:
                    continue
                
                weight=len(subset)/total_samples 
                
                split_info-=weight*np.log(weight) #using log base 2 for entropy
                
                if split_info > 0: #to avoid deviding by 0 (undefined)
                    goodness=impurity_reduction/split_info
                else:
                    goodness=0
        #if we are not using gain ratio and impurity func is simply the impurity reduction:
        else: 
            goodness = impurity_reduction
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return goodness, groups
        
    def calc_feature_importance(self, n_total_sample):
        """
        Calculate the selected feature importance.
        
        Input:
        - n_total_sample: the number of samples in the dataset.

        This function has no return value - it stores the feature importance in 
        self.feature_importance
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        if self.terminal or self.feature == -1:
            self.feature_importance = 0
            return
    
        # Calculate the node's relative weight
        node_weight = len(self.data) / n_total_sample
        
        # Calculate goodness of split for the selected feature
        goodness, _ = self.goodness_of_split(self.feature)

        self.feature_importance=goodness*node_weight       
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    
    def calculate_chi_square(self,feature,groups):
        """
        Calculate the chi-square statistic for a given feature split:
        X²(S,A) = Σ_(v∈Values(A)) Σ_(j=1)^k ((|S_v,j| - |S_v|p_j)²) / (|S_v|p_j)
        
        Input:
        - feature: the feature index being evaluated
        - groups: dictionary mapping feature values to data subsets
        
        Returns:
        - chi_square: the chi-square statistic
        """
        
        # get unique class labels (j values from 1 to k)(here i only have 2)
        label_values = np.unique(self.data[:, -1])
            
        # calculate proportions
        parent_class_proportions = {}
        total_samples = len(self.data)
        for label_val in label_values:
            total = np.sum(self.data[:, -1] == label_val) #
            parent_class_proportions[label_val] = total / total_samples # 
    
        # Calculate chi-square statistic
        chi_square = 0.0

        # First sigma: sum over all feature values (v∈Values(A))
        for value, subset in groups.items():
            subset_size = len(subset)  # |S_v|
            
            if subset_size == 0: # Skip empty subsets
                continue       
                            
            # Second sigma: sum over all class labels (j=1 to k)
            for label_val in label_values:
                # Calculate observed frequency |S_v,j|
                actual = np.sum(subset[:, -1] == label_val)
                expected = subset_size * (parent_class_proportions[label_val])
                # Skip if expected is very close to 0
                if expected < 1e-10:
                    continue
                
                # Add chi-square term: ((|S_v,j| - |S_v|p_j)²) / (|S_v|p_j)            
                chi_square += ((actual - expected) ** 2) / expected
    
        return chi_square
        
    def split(self):
        """
        Splits the current node according to the self.impurity_func. This function finds
        the best feature to split according to and create the corresponding children.
        This function should support pruning according to self.chi and self.max_depth.

        This function has no return value
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        #halting conditions and supports pruning by depth:
        if self.depth>=self.max_depth: #we wont split if we reached the maximum allowed depth
            self.terminal = True
            return

        unique_labels=np.unique(self.data[:,-1]) #if all samples are the same lable than the node is pure and therefore there is nothing to split
        if len(unique_labels)==1:
            self.terminal=True
            return
        
        #checking best feature to split upon:
        best_feature=-1
        best_goodness=-float('inf')
        best_groups={}

        #loop over each feather (all coloums excpet the last col which is the label's col)
        for feature in range(self.data.shape[1] -1):
            #calculate goodness of split of that feature
            goodness,groups=self.goodness_of_split(feature)

            #check each time if value is better, otherwise discard
            if goodness>best_goodness:
                best_goodness=goodness
                best_feature=feature
                best_groups=groups
        #if none of the feature give us a good split, we mark this as termial ie leaf

        if best_feature==-1 or best_goodness<=0:
            self.terminal=True
            return
        
        # Apply chi-square pruning
        if self.chi<1: #only halt according to chi test if chi < 1. 
            
            # Calculate degrees of freedom 
            # deg_freedom = (k - 1)(|Values(A)| - 1)
            k = len(unique_labels)  # Number of classes in this node
            values_A = len(best_groups)  # Number of unique values for best feature
            deg_freedom = (k - 1) * (values_A - 1)  # Degrees of freedom

            deg_freedom = min(deg_freedom, max(chi_table.keys())) #deg_freedom is within the bounds of our chi_table

            # Calculate chi-square statistic for this spcific split
            #X²(S,A) := Σ_(v∈Values(A)) Σ_(j=1)^k ((|S_v,j| - |S_v|p_j)²) / (|S_v|p_j)
            chi_statistic = self.calculate_chi_square(best_feature, best_groups)

            # Get the critical chi-square value from the table
            critical_value = chi_table[deg_freedom][self.chi]
        
            # Check if chi-square is significant
            if chi_statistic <= critical_value:
                # Split is NOT significant, make this a leaf node
                self.terminal = True
                return

        # If we get here, either chi test passed or chi=1 (no pruning)
        self.feature=best_feature
        
        #create child nodes for each value of the best feature
        for value,subset in best_groups.items():
            if len(subset)==0: #skip if subset is empty
                continue
        
            #create a new child node with the subset data
            child=DecisionNode(
                data=subset, 
                impurity_func=self.impurity_func, 
                depth=self.depth+1, 
                chi=self.chi, 
                max_depth=self.max_depth, 
                gain_ratio=self.gain_ratio
            )
        
            #now we add this child to the node's children's list:
            self.add_child(child,value)
            
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

                    
class DecisionTree:
    def __init__(self, data, impurity_func, feature=-1, chi=1, max_depth=1000, gain_ratio=False):
        self.data = data # the training data used to construct the tree
        self.root = None # the root node of the tree
        self.max_depth = max_depth # the maximum allowed depth of the tree
        self.chi = chi # the P-value cutoff used for chi square pruning
        self.impurity_func = impurity_func # the impurity function to be used in the tree
        self.gain_ratio = gain_ratio #
        
    def depth(self):
        return self.root.depth

    def build_tree(self):
        """
        Build a tree using the given impurity measure and training dataset. 
        You are required to fully grow the tree until all leaves are pure 
        or the goodness of split is 0.

        This function has no return value
        """
        self.root = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        """
        we first create the root,
        recursivly split the tree unitl leaves are pure or no info_gain
        
        """
        #initiallizing the root node using the entire trainig data
        self.root=DecisionNode(data=self.data, impurity_func=self.impurity_func, chi=self.chi,
                               max_depth=self.max_depth, gain_ratio=self.gain_ratio)

        #helper function to recursivley split nodes. recives node as input which is the curr node to process. 
        def _build_subtree(node):
            #call split func we build prior on curr node. in split it checks halting
            node.split()
            
            # If not a leaf node, recursively split children
            if not node.terminal:
                for child in node.children:
                    _build_subtree(child)

            # Calculate feature importance for this node
            # (This happens during the recursive "unwinding")
            node.calc_feature_importance(len(self.data))

        _build_subtree(self.root)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def predict(self, instance):
        """
        Predict a given instance
     
        Input:
        - instance: an row vector from the dataset. Note that the last element 
                    of this vector is the label of the instance.
     
        Output: the prediction of the instance.
         
        self.data = data # the data instances associated with the node
        self.terminal = False # True iff node is a leaf
        self.feature = feature # column index of feature/attribute used for splitting the node
        self.pred = self.calc_node_pred() # the class prediction associated with the node
        self.depth = depth # the depth of the node
        self.children = [] # the children of the node (array of DecisionNode objects)
        self.children_values = [] # the value associated with each child for the feature used for splitting the node
        self.max_depth = max_depth # the maximum allowed depth of the tree
        self.chi = chi # the P-value cutoff used for chi square pruning
        self.impurity_func = impurity_func # the impurity function to use for measuring goodness of a split
        self.gain_ratio = gain_ratio # True iff GainRatio is used to score features
        self.feature_importance = 0
        
        """
        pred = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        curr_node=self.root
        
        # Traverse the tree until reaching a leaf node
        while not curr_node.terminal:
            feature_index=curr_node.feature #feature according to which we split the node at this stage
            feature_val=instance[feature_index]
            
             # If no matching branch is found (unseen feature value)
            if feature_val not in curr_node.children_values:
                break 
            
            for i, val in enumerate(curr_node.children_values):
                if feature_val==val:
                    #we move to this child
                    curr_node=curr_node.children[i]
                    break 

        node=curr_node 
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return node.pred

    def calc_accuracy(self, dataset):
        """
        Predict a given dataset 
     
        Input:
        - dataset: the dataset on which the accuracy is evaluated
     
        Output: the accuracy of the decision tree on the given dataset (%).
       
        """
        accuracy = 0
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        correct_prediction=0
        total_instances=len(dataset)

        for instance in dataset:
            predicted_val=self.predict(instance)
        
            actual_val=str(instance[-1]) # convert to str for consistent comparison

            if predicted_val==actual_val:
                correct_prediction+=1
        
        # Calculate accuracy as a percentage
        accuracy=(correct_prediction/total_instances)*100

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return accuracy
        

def depth_pruning(X_train, X_validation):
    """
    Calculate the training and validation accuracies for different depths
    using the best impurity function and the gain_ratio flag you got
    previously. (Impurity Function: calc_entropy, gain_ratio Flag: True)

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels
 
    Output: the training and validation accuracies per max depth
    """
    training = []
    validation  = []
    root = None
    # our best configuration: Entropy with Gain Ratio
    impurity_func = calc_entropy
    use_gain_ratio = True
    for max_depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        # Using calc_entropy as the best impurity function and gain_ratio=True
        tree=DecisionTree(data=X_train,impurity_func=impurity_func,max_depth=max_depth,gain_ratio=use_gain_ratio )
        tree.build_tree()
        training_accuracy=tree.calc_accuracy(X_train)
        validation_accuracy=tree.calc_accuracy(X_validation)

        training.append(training_accuracy)
        validation.append(validation_accuracy)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    return training, validation


def chi_pruning(X_train, X_test):

    """
    Calculate the training and validation accuracies for different chi values
    using the best impurity function and the gain_ratio flag you got
    previously. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels
 
    Output:
    - chi_training_acc: the training accuracy per chi value
    - chi_validation_acc: the validation accuracy per chi value
    - depth: the tree depth for each chi value
    """
    chi_training_acc = []
    chi_validation_acc  = []
    depth = []

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    # best configuration from previous experiments calc_entropy with gain_ratio=True
    impurity_func = calc_entropy
    use_gain_ratio = True

    chi_vals=[1, 0.5, 0.25, 0.1, 0.05, 0.0001]
    for chi_val in chi_vals:
        tree=DecisionTree(data=X_train,impurity_func=impurity_func,chi=chi_val,gain_ratio=use_gain_ratio)
        tree.build_tree()

        training_accuracy=tree.calc_accuracy(X_train)
        chi_training_acc.append(training_accuracy)

        validation_accuracy=tree.calc_accuracy(X_test)
        chi_validation_acc.append(validation_accuracy)
        
        depth.append(find_max_depth(tree.root))
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
        
    return chi_training_acc, chi_validation_acc, depth

def find_max_depth(node):
    if node is None:
        return 0

    if node.terminal or len(node.children)==0:
        return node.depth
        
    max_child_depth=0
    for child in node.children:
        child_depth=find_max_depth(child)
        if child_depth > max_child_depth:
            max_child_depth=child_depth

    return max_child_depth

def count_nodes(node):
    """
    Count the number of node in a given tree
 
    Input:
    - node: a node in the decision tree.
 
    Output: the number of node in the tree.
    """
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    if node is None:
        return 0
    n_nodes=0
    stack=[node]
    while stack:
        n_nodes+=1
        curr_node=stack.pop()
        if hasattr(curr_node, 'children') and curr_node.children:
            for child in curr_node.children:
                if child is not None:
                    stack.append(child)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return n_nodes






