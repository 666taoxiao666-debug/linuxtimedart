# -*- coding: utf-8 -*-
"""
Finite Adaptability Functions

@author: a.stratigakos@imperial.ac.uk
"""

#Import Libraries
import numpy as np
from sklearn.linear_model import LinearRegression
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from clean_torch_custom_layers import *

class Learn_FiniteAdapt_Robust_Reg(object):
  '''Finite Adaptability - **LEARN** Partitions: Tree-based method to learn uncertainty set partitions (Algorithm 4).
    Each leaf stores a nominal model and a robust/ adaptive robust with linear decision rules (LDR).
    All the underlying models are based on the class Adaptive_LDR_Regression     
  
  Paremeters:
      D: Maximum depth of the tree
      target_col: Columns with features that could go missing
      fix_col: Columns with features that are always available
      type_split: regular or random splits for the ExtraTree algorithm (perhaps passed as hyperparameter in the forest)
      Max_splits: Maximum number of node splits (each split generates two partitions)
      red_threshold: Minimum loss reduction to split a node
      error_metric: Loss function
      **kwargs: Keyword arguments for Adaptive_LDR_Regression
      '''
  def __init__(self, target_col, fix_col, D = 10, Max_splits = 5, red_threshold = .01, error_metric = 'mse', **kwargs):
      
    self.D = D
    self.Max_splits = Max_splits
    self.red_threshold = red_threshold
    self.error_metric = error_metric
    
    # initialize target and fixed features
    self.target_col = np.array(target_col).copy().astype(int)
    self.fix_col = np.array(fix_col).copy().astype(int)
        
    # arguments for base learner
    self.adapt_regr_param = kwargs
    if self.adapt_regr_param['hidden_sizes'] == []:
        self.base_model = 'LR'
    else:
        self.base_model = 'NN'

    # Set-up some parameters used multiple times
    # *** I only consider inequality-based uncertainty sets ***
    self.attack_type = self.adapt_regr_param['attack_type']
    self.budget_constraint = 'inequality'
        
    # Check if model is declared properly
    if 'apply_LDR' not in self.adapt_regr_param:
        print('Warning: Did not select LDR policy, set to TRUE by default')
        self.adapt_regr_param['apply_LDR'] = True


  def fit(self, X, Y, tree_grow_algo = 'leaf-wise', max_gap_threshold = 0.001, val_split = 0.05, **kwargs):
    ''' Learn tree-based partitions
    
    Paremeters:
        X: Features
        Y: Target data
        tree_grow_algo: If leaf-wise / depth-first, then we minimize the loss gap; else, the tree is grown horizontally
        max_gap: Maximum loss gap (UB-LB). If the max loss gap across all leafs is smaller than max_gap, then stop growing the tree
        error_metric: Loss function
        **kwargs: Keyword arguments for gradient-based algorithm/ standard hyperparameters for torch layers/ will be parsed onto Adaptive_LDR_Regression
    '''

    # self.MLP_train_dict = kwargs
    # Gradient-descent and training hyperparameters
    self.gd_train_parameters = kwargs
    
    # Check hyperparameters for gradient descent// set default values if they are missing
    if 'weight_decay' not in self.gd_train_parameters:
        self.gd_train_parameters['weight_decay'] = 0

    if 'batch_size' not in self.gd_train_parameters:
        self.gd_train_parameters['batch_size'] = 512

    if 'lr' not in self.gd_train_parameters:
        self.gd_train_parameters['lr'] = 1e-2

        
    # store robust and adaptive robust models
    self.Robust_models = []
    self.max_gap_threshold = max_gap_threshold

    # keyword arguments for standard class object resilient_MLP
    num_features = X.shape[1]    #Number of features
    n_obs = len(Y)
    
    # Each node has a combination
    #Initialize tree structure
    self.Node_id = [0]
    self.Depth_id = [0]
    self.parent_node  = [None]
    self.children_left = [-1]
    self.children_right = [-1]
    self.children_left_dict = {}
    self.children_right_dict = {}
    
    node_id_counter = 0
    
    #### Initialize root node
    print('Initializing root node...')
    # at root node, no missing data (1 = missing, 0 = not missing)
    self.missing_pattern = [np.zeros(num_features)]
    # number of missing features per tree node
    self.total_missing_feat = [self.missing_pattern[0].sum()]
    # features that CANNOT change in the current node (fixed features and features that changed in parent nodes)
    self.fixed_features = [self.fix_col]
    # features that CAN change missing pattern within current node
    self.target_features = [self.target_col]
    # node split parameters: feature to split on and its value
    self.feature = [-1]
    self.threshold = [-1]

    ####### Create train/validation data loaders for torch modules
    temp_miss_X = (1-self.missing_pattern[0])*X
    ### Train Nominal model (root node: no missing data)
    n_valid_obs = int(val_split*len(Y))
    
    if val_split == 0:    
        trainY = Y
        validY = Y
        
        train_temp_miss_X = temp_miss_X
        valid_temp_miss_X = temp_miss_X   
        
    else:
        trainY = Y[:-n_valid_obs]
        validY = Y[-n_valid_obs:]

        train_temp_miss_X = temp_miss_X[:-n_valid_obs]
        valid_temp_miss_X = temp_miss_X[-n_valid_obs:]        

    tensor_trainY = torch.FloatTensor(trainY)
    tensor_validY = torch.FloatTensor(validY)
        
    tensor_train_missX = torch.FloatTensor(train_temp_miss_X)
    tensor_valid_missX = torch.FloatTensor(valid_temp_miss_X)

    train_data_loader = create_data_loader([tensor_train_missX, tensor_trainY], batch_size = self.gd_train_parameters['batch_size'], shuffle = False)        
    valid_data_loader = create_data_loader([tensor_valid_missX, tensor_validY], batch_size = self.gd_train_parameters['batch_size'], shuffle = False)
        
    ####### Train nominal model
    ###########################################
    nominal_model = Adaptive_LDR_Regression(input_size = num_features, hidden_sizes = self.adapt_regr_param['hidden_sizes'], 
                                             output_size = self.adapt_regr_param['output_size'], target_col = self.target_features[0], 
                                             fix_col = self.fixed_features[0], projection = False, 
                                             apply_LDR = self.adapt_regr_param['apply_LDR'], Gamma = 0, budget_constraint = self.budget_constraint)
    
    optimizer = torch.optim.Adam(nominal_model.parameters(), 
                                 lr = self.gd_train_parameters['lr'], weight_decay = self.gd_train_parameters['weight_decay'])
    
    # If base model is linear, then warm-start with analytical solution// skip gradient descent
    if self.base_model == 'LR':
        lr_model = LinearRegression(fit_intercept = True)
        lr_model.fit(train_temp_miss_X, trainY)
            
        nominal_model.model[0].weight.data = torch.FloatTensor(lr_model.coef_[0].reshape(1,-1))
        nominal_model.model[0].bias.data = torch.FloatTensor(lr_model.intercept_)
    else:                
        nominal_model.train_model(train_data_loader, valid_data_loader, optimizer, 
                                   epochs = self.gd_train_parameters['epochs'], patience = self.gd_train_parameters['patience'], verbose = self.gd_train_parameters['verbose'])
        
    ######## Train Adaptive Robust Model
    # Set current budget of uncertainty
    # Nominal loss: is estimated/ WC loss: set to negative value
    # if self.adapt_regr_param['budget_constraint'] == 'equality':                
    #     gamma_temp = 1
    # elif self.adapt_regr_param['budget_constraint'] == 'inequality':
    #     gamma_temp = len(self.target_features[0])
    
    # Set current budget of uncertainty (*** I assume inequality uncertainty sets ***)
    gamma_temp = len(self.target_features[0])
        
    robust_model = Adaptive_LDR_Regression(input_size = num_features, hidden_sizes = self.adapt_regr_param['hidden_sizes'], 
                                             output_size = self.adapt_regr_param['output_size'], target_col = self.target_col, fix_col = self.fix_col, projection = False, 
                                             apply_LDR = self.adapt_regr_param['apply_LDR'], Gamma = gamma_temp, budget_constraint = self.budget_constraint)
    
    optimizer = torch.optim.Adam(robust_model.parameters(), 
                                 lr = self.gd_train_parameters['lr'], weight_decay = self.gd_train_parameters['weight_decay'])

    # Warm start robust model with nominal model parameters
    robust_model.load_state_dict(nominal_model.state_dict(), strict=False)

    robust_model.adversarial_train_model(train_data_loader, valid_data_loader, optimizer, epochs = self.gd_train_parameters['epochs'], 
                                 patience = self.gd_train_parameters['patience'], verbose = self.gd_train_parameters['verbose'],
                                 attack_type = self.attack_type, warm_start_nominal = False, freeze_weights = False)
    
    # Nominal and WC loss
    insample_loss = eval_predictions(nominal_model.predict(train_temp_miss_X, self.missing_pattern[0]), trainY, self.error_metric)
    insample_wc_loss = eval_predictions(robust_model.predict(train_temp_miss_X, self.missing_pattern[0]), trainY, self.error_metric)
    
    print(f'Lower loss bound:{np.round(insample_loss, 4)}')
    print(f'Upper loss bound:{np.round(robust_model.best_val_loss, 4)}')
    
    # Nominal, WC loss, and loss gap
    self.LB_Loss = [insample_loss]
    self.UB_Loss = [robust_model.best_val_loss]
    
    self.Loss_gap = [self.UB_Loss[0] - self.LB_Loss[0]]
    self.Loss_gap_perc = [(self.UB_Loss[0] - self.LB_Loss[0])/self.LB_Loss[0]]
    
    # store nominal and WC model parameters
    # !!!!! Potentially store the weights of a torch model
    self.node_model_ = [nominal_model]
    self.wc_node_model_ = [robust_model]
    self.total_models = 1
    
    # Keep list with nodes that we have not checked yet (are neither parent nodes or set as leaf nodes), updated dynamically at each iteration
    # list of nodes that we could split on
    list_nodes_candidates = [0]
    # same list ordered by loss gap (best-first)
    list_nodes_candidates_ordered = [0]    

    # while the (ordered) list of candidate nodes is not empty
    if tree_grow_algo == 'leaf-wise':
        # **suggested approach**, grow best-first, leafwise 
        list_to_check = list_nodes_candidates_ordered
    else:
        # grow depth-wise
        list_to_check = list_nodes_candidates
        
    ##### While loop to grow the tree
    while list_to_check:     
        # node to check in iteration
        node = list_to_check[0]
        
        # Depth-first: grow the leaf with the highest WC loss        
        print(f'Node = {node}')
        if (self.Depth_id[node] >= self.D) or (self.total_models >= self.Max_splits) or (self.Loss_gap_perc[node] <= self.max_gap_threshold):
            # remove node from list to check (only used for best-first growth)
            
            list_nodes_candidates.remove(node)
            list_nodes_candidates_ordered.remove(node)

            # update checking list
            if tree_grow_algo == 'leaf-wise':
                list_to_check = list_nodes_candidates_ordered
            else:
                list_to_check = list_nodes_candidates
            
            # Fix as leaf node and go back to loop
            self.children_left.append(-1)
            self.children_right.append(-1)
            
            self.children_left_dict[node] = -1
            self.children_right_dict[node] = -1

            continue
        
        # candidate features that CAN go missing in current node        
        cand_features = self.target_features[node]
        
            
        # Initialize placeholder for subtree error
        Best_loss = self.LB_Loss[node]
        # Indicators to check for splitting node
        solution_count = 0
        apply_split = False
            
        ### Loop over features, find the worst-case loss when a feature goes missing (i.e., feature value is set to 0)
        for j, cand_feat in enumerate(cand_features):
            #print('Col: ', j)
            
            # temporary missing patterns            
            temp_missing_pattern = self.missing_pattern[node].copy()
            temp_missing_pattern[cand_feat] = 1
            temp_miss_X = (1-temp_missing_pattern)*X                
            
            
            ####### Create train/validation data loaders for torch modules
            if val_split == 0:                    
                train_temp_miss_X = temp_miss_X
                valid_temp_miss_X = temp_miss_X   
            else:
                train_temp_miss_X = temp_miss_X[:-n_valid_obs]
                valid_temp_miss_X = temp_miss_X[-n_valid_obs:]        
                
            tensor_train_missX = torch.FloatTensor(train_temp_miss_X)
            tensor_valid_missX = torch.FloatTensor(valid_temp_miss_X)

            train_data_loader = create_data_loader([tensor_train_missX, tensor_trainY], batch_size = self.gd_train_parameters['batch_size'], shuffle = False)        
            valid_data_loader = create_data_loader([tensor_valid_missX, tensor_validY], batch_size = self.gd_train_parameters['batch_size'], shuffle = False)
                
            # Find performance degradation for the NOMINAL model when data are missing
            current_node_loss = eval_predictions(self.node_model_[node].predict(train_temp_miss_X, temp_missing_pattern), trainY, self.error_metric)
        
            # Check if nominal model **degrades** enough (loss increase)
            nominal_loss_worse_ind = ((current_node_loss-self.LB_Loss[node])/self.LB_Loss[node] > self.red_threshold)   
            
            if (current_node_loss > Best_loss)*(nominal_loss_worse_ind):    
                # Further check if a new model **improves** over the WC model enough (decrease loss)

                ####### Train nominal model with missing data

                new_nominal_model = Adaptive_LDR_Regression(input_size = num_features, hidden_sizes = self.adapt_regr_param['hidden_sizes'], 
                                                         output_size = self.adapt_regr_param['output_size'], target_col = self.target_features[node], 
                                                         fix_col = self.fixed_features[node], projection = False, 
                                                         apply_LDR = self.adapt_regr_param['apply_LDR'])
                
                optimizer = torch.optim.Adam(new_nominal_model.parameters(), 
                                             lr = self.gd_train_parameters['lr'], weight_decay = self.gd_train_parameters['weight_decay'])
    
                # If base model is linear, then warm-start with analytical solution// skip gradient descent
                if self.base_model == 'LR':
                    new_lr_model = LinearRegression(fit_intercept = True)
                    new_lr_model.fit(train_temp_miss_X, trainY)
                        
                    new_nominal_model.model[0].weight.data = torch.FloatTensor(new_lr_model.coef_[0].reshape(1,-1))
                    new_nominal_model.model[0].bias.data = torch.FloatTensor(new_lr_model.intercept_)
                else:                
                    new_nominal_model.train_model(train_data_loader, valid_data_loader, optimizer, 
                                               epochs = self.gd_train_parameters['epochs'], patience = self.gd_train_parameters['patience'], verbose = self.gd_train_parameters['verbose'])

                # Check model degradation
                retrain_loss = eval_predictions(new_nominal_model.predict(tensor_train_missX, temp_missing_pattern), trainY, self.error_metric)
                wc_node_loss = eval_predictions(self.wc_node_model_[node].predict(tensor_train_missX, temp_missing_pattern), trainY, self.error_metric)
                
                if ((wc_node_loss-retrain_loss)/wc_node_loss > self.red_threshold):
                                
                    solution_count = solution_count + 1
                    apply_split = True
                                        
                    # placeholders for node split
                    best_new_model = new_nominal_model
                    best_split_feature = cand_feat
                    
                    Best_loss = current_node_loss
                    # update running loss function at current node/ nominal loss at right child node
                    Best_insample_loss = retrain_loss

        #If split is applied, then update tree structure (missing feature goes to right child, left child copies current node)
        # Right child: new model with missing feature
        # Left child: worst-case scenario of the remaining features
        
        if apply_split == True:
            print('Solution found, learning WC model and updating tree structure...')

            self.total_models += 1
            
            self.parent_node.extend(2*[node])    
            self.Node_id.extend([node_id_counter + 1, node_id_counter + 2])
            self.Depth_id.extend(2*[self.Depth_id[node]+1])
            
            self.feature[node] = best_split_feature
            self.feature.extend(2*[-1])
            
            #### Left child node: learn worst-case model/ !!! previous missing patterns need to be respected 
            # update left child/ same parameters/ update fixed_cols/same missingness pattern
            left_fix_cols = np.append(self.fixed_features[node].copy(), best_split_feature)
            left_target_cols = self.target_features[node].copy()
            left_target_cols = np.delete(left_target_cols, np.where(left_target_cols==best_split_feature))
            left_missing_pattern = self.missing_pattern[node].copy()

            # if self.gd_FDRR_params['budget_constraint'] == 'equality':                
            #     gamma_temp = 1
            # elif self.gd_FDRR_params['budget_constraint'] == 'inequality':
            #     gamma_temp = len(left_target_cols)
            
            # Update budget of uncertainty
            temp_left_gamma = len(left_target_cols)                
            temp_miss_X = (1-left_missing_pattern)*X
            
            ####### Create train/validation data loaders for torch modules
            if val_split == 0:                    
                train_temp_miss_X = temp_miss_X
                valid_temp_miss_X = temp_miss_X   
            else:
                train_temp_miss_X = temp_miss_X[:-n_valid_obs]
                valid_temp_miss_X = temp_miss_X[-n_valid_obs:]        
                
            tensor_train_missX = torch.FloatTensor(train_temp_miss_X)
            tensor_valid_missX = torch.FloatTensor(valid_temp_miss_X)

            left_train_data_loader = create_data_loader([tensor_train_missX, tensor_trainY], batch_size = self.gd_train_parameters['batch_size'], shuffle = False)        
            left_valid_data_loader = create_data_loader([tensor_valid_missX, tensor_validY], batch_size = self.gd_train_parameters['batch_size'], shuffle = False)

            
            ############ Left leaf: Robust model
                
            left_robust_model = Adaptive_LDR_Regression(input_size = num_features, hidden_sizes = self.adapt_regr_param['hidden_sizes'], 
                                                     output_size = self.adapt_regr_param['output_size'], target_col = left_target_cols, 
                                                     fix_col = left_fix_cols, projection = False, 
                                                     apply_LDR = self.adapt_regr_param['apply_LDR'], Gamma = temp_left_gamma, budget_constraint = self.budget_constraint)
            
            optimizer = torch.optim.Adam(left_robust_model.parameters(), 
                                         lr = self.gd_train_parameters['lr'], weight_decay = self.gd_train_parameters['weight_decay'])

            # Warm start robust model with nominal model parameters
            left_robust_model.load_state_dict(self.node_model_[node].state_dict(), strict=False)

            left_robust_model.adversarial_train_model(left_train_data_loader, left_valid_data_loader, optimizer, epochs = self.gd_train_parameters['epochs'], 
                                         patience = self.gd_train_parameters['patience'], verbose = self.gd_train_parameters['verbose'], attack_type = self.attack_type, 
                                         warm_start_nominal = False, freeze_weights = False)
            
            # Estimate WC loss and nominal loss
            left_insample_wcloss = left_robust_model.best_val_loss

            # Nominal loss: inherits the nominal loss of the parent node/ WC loss: the estimated
            self.LB_Loss.append(self.LB_Loss[node])
            self.UB_Loss.append(left_insample_wcloss)
            self.Loss_gap.append( self.UB_Loss[-1] -  self.LB_Loss[-1])
            self.Loss_gap_perc.append( (self.UB_Loss[-1] -  self.LB_Loss[-1])/self.LB_Loss[-1] )
            
            self.missing_pattern.append(left_missing_pattern)
            self.total_missing_feat.append(left_missing_pattern.sum())

            self.node_model_.append(self.node_model_[node])
            self.wc_node_model_.append(left_robust_model)
            
            # update missing patterns for downstream robust problem
            self.fixed_features.append(left_fix_cols)
            self.target_features.append(left_target_cols)
            
            #### Right child node: update with new model
            # update right child/ update parameters/ update fixed_cols/same missingness pattern
            right_fix_cols = left_fix_cols
            right_target_cols = left_target_cols
            right_missing_pattern = self.missing_pattern[node].copy()
            right_missing_pattern[best_split_feature] = 1
            
            # Set new budget of uncertainty            
            temp_right_gamma = len(right_target_cols)
            temp_miss_X = (1-right_missing_pattern)*X
            
            ####### Create train/validation data loaders for torch modules
            if val_split == 0:                    
                train_temp_miss_X = temp_miss_X
                valid_temp_miss_X = temp_miss_X   
            else:
                train_temp_miss_X = temp_miss_X[:-n_valid_obs]
                valid_temp_miss_X = temp_miss_X[-n_valid_obs:]        
                
            tensor_train_missX = torch.FloatTensor(train_temp_miss_X)
            tensor_valid_missX = torch.FloatTensor(valid_temp_miss_X)
                
            right_train_data_loader = create_data_loader([tensor_train_missX, tensor_trainY], batch_size = self.gd_train_parameters['batch_size'], shuffle=False)        
            right_valid_data_loader = create_data_loader([tensor_valid_missX, tensor_validY], batch_size = self.gd_train_parameters['batch_size'], shuffle=False)


            ############ Right leaf: Robust model
                
            right_robust_model = Adaptive_LDR_Regression(input_size = num_features, hidden_sizes = self.adapt_regr_param['hidden_sizes'], 
                                                     output_size = self.adapt_regr_param['output_size'], target_col = right_target_cols, 
                                                     fix_col = right_fix_cols, projection = False, 
                                                     apply_LDR = self.adapt_regr_param['apply_LDR'], Gamma = temp_right_gamma, budget_constraint = self.budget_constraint)
            
            optimizer = torch.optim.Adam(right_robust_model.parameters(), 
                                         lr = self.gd_train_parameters['lr'], weight_decay = self.gd_train_parameters['weight_decay'])

            # Warm start robust model with **new** nominal model
            right_robust_model.load_state_dict(best_new_model.state_dict(), strict=False)

            right_robust_model.adversarial_train_model(right_train_data_loader, right_valid_data_loader, optimizer, epochs = self.gd_train_parameters['epochs'], 
                                         patience = self.gd_train_parameters['patience'], verbose = self.gd_train_parameters['verbose'], attack_type = self.attack_type, 
                                         warm_start_nominal = False, freeze_weights = False)

            
            # Estimate WC loss and nominal loss
            right_insample_wcloss = right_robust_model.best_val_loss
            
            self.LB_Loss.append(Best_insample_loss)
            self.UB_Loss.append(right_insample_wcloss)
            self.Loss_gap.append( self.UB_Loss[-1] -  self.LB_Loss[-1])
            self.Loss_gap_perc.append( (self.UB_Loss[-1] -  self.LB_Loss[-1])/self.LB_Loss[-1] )

            self.missing_pattern.append(right_missing_pattern)
            self.total_missing_feat.append(right_missing_pattern.sum())


            self.node_model_.append(best_new_model)            
            self.wc_node_model_.append(right_robust_model)
                        
            # update missing patterns for downstream robust problem            
            self.fixed_features.append(right_fix_cols)
            self.target_features.append(right_target_cols)
                        
            if node==0:
                self.children_left[node] = node_id_counter+1
                self.children_right[node] = node_id_counter+2
            else:
                self.children_left.append(node_id_counter+1)
                self.children_right.append(node_id_counter+2)

            self.children_left_dict[node] = node_id_counter+1
            self.children_right_dict[node] = node_id_counter+2

            # Update lists of candidate nodes (node is removed in both cases)
            list_nodes_candidates = list_nodes_candidates + [ node_id_counter + 1, node_id_counter + 2]
            
            list_nodes_candidates_ordered = [list_nodes_candidates[i] for i in np.argsort(np.array(self.Loss_gap_perc)[list_nodes_candidates])[::-1]]
                    
            list_nodes_candidates.remove(node)            
            list_nodes_candidates_ordered.remove(node)            
                        
            # Update checking list for while loop
            if tree_grow_algo == 'leaf-wise':
                list_to_check = list_nodes_candidates_ordered
                print(list_to_check)
            else:
                list_to_check = list_nodes_candidates

            node_id_counter = node_id_counter + 2
            
        else:
            #Fix as leaf node
            self.children_left.append(-1)
            self.children_right.append(-1)
            
            self.children_left_dict[node] = -1
            self.children_right_dict[node] = -1

            # Update lists of candidate nodes (remove node)
            list_nodes_candidates.remove(node)            
            list_nodes_candidates_ordered.remove(node)            
        
            # Update checking list for while loop
            if tree_grow_algo == 'leaf-wise':
                list_to_check = list_nodes_candidates_ordered
            else:
                list_to_check = list_nodes_candidates

                            
  def apply(self, X, missing_mask):
     ''' Function that returns the Leaf id for each point. Similar to sklearn's implementation.
     '''
     node_id = np.zeros((X.shape[0],1))
     for i in range(X.shape[0]): 
         #New query point
         x0 = X[i:i+1,:]
         m0 = missing_mask[i:i+1,:]

         #Start from root node
         node = 0
         #Go downwards until reach a Leaf Node
         while ((self.children_left_dict[node] != -1) and (self.children_right_dict[node] != -1)):
             if (m0==self.missing_pattern[node]).all():
                 break
             if m0[:, self.feature[node]] == 0:
                 # if feature is not missing, go left
                 node = self.children_left_dict[node]
             elif m0[:,self.feature[node]] == 1:
                 # if feature is missing, go right
                node = self.children_right_dict[node]
             #print('New Node: ', node)
         node_id[i] = self.Node_id[node]
     return node_id
  
  def predict(self, X, missing_mask):
     ''' Function to predict using a trained tree. 
     '''
     Predictions = []
     for i in range(X.shape[0]): 
         #New query point
         x0 = X[i:i+1,:]
         m0 = missing_mask[i:i+1,:]
         #Start from root node
         node = 0

         # !!!! If you go to leaf, might be overly conservative
         #Go down the tree until you match the missing pattern OR a Leaf node is reached
         while ((self.children_left_dict[node] != -1) and (self.children_right_dict[node] != -1)):
             if (m0==self.missing_pattern[node]).all():
                 break
             
             elif m0[:, self.feature[node]] == 0:
                 # if feature is not missing, go left
                 node = self.children_left_dict[node]
             elif m0[:,self.feature[node]] == 1:
                 # if feature is missing, go right
                node = self.children_right_dict[node]
             # print('New Node: ', node)
             # print(m0)
         
         if (m0==self.missing_pattern[node]).all():
             # nominal model
             Predictions.append( self.node_model_[node].predict(x0, m0).reshape(-1))
         else:
             # WC model
             Predictions.append( self.wc_node_model_[node].predict(x0, m0).reshape(-1))

     return np.array(Predictions).reshape(-1,1)
 
class Fixed_FiniteAdapt_Robust_Reg(object):
  '''Finite Adaptability - **FIXED** Partitions: Partition at integers in range [0, gamma] where gamma is the budget of uncertainty.
     Each ``leaf'' stores a nominal model and a robust/ adaptive robust model with linear decision rules (LDR).
     All the underlying models are based on the class Adaptive_LDR_Regression object
            
  Paremeters:
      target_col: Columns with features that could go missing
      fix_col: Columns with features that are always available
      **kwargs: Keyword arguments for Adaptive_LDR_Regression object
      '''
  def __init__(self, target_col, fix_col, error_metric = 'mse', **kwargs):
      
    self.Gamma_max = len(target_col)
    self.error_metric = error_metric
    
    # initialize target and fixed features
    self.target_col = np.array(target_col).copy().astype(int)
    self.fix_col = np.array(fix_col).copy().astype(int)

    # arguments for base learner
    self.adapt_regr_param = kwargs
    
    if self.adapt_regr_param['hidden_sizes'] == []:
        self.base_model = 'LR'
    else:
        self.base_model = 'NN'
        
    # Check if model is declared properly
    if 'apply_LDR' not in self.adapt_regr_param:
        print('Warning: Did not select LDR policy, set to TRUE by default')
        self.adapt_regr_param['apply_LDR'] = True
        
  def fit(self, X, Y, val_split = 0.05, **kwargs):
    ''' Fitting models.
        Paremeters:
            X: features
            Y: Y target
            val_split: percentage validation split (will be used for NNs)
            **kwargs: keyword arguments for gradient-based training (epochs, batch size, etc.) to be passed on Adaptive_LDR_Regression object

    '''
    # self.MLP_train_dict = kwargs
    # Gradient-descent and training hyperparameters
    self.gd_train_parameters = kwargs
    
    # Check hyperparameters for gradient descent// set default values if they are missing
    if 'weight_decay' not in self.gd_train_parameters:
        self.gd_train_parameters['weight_decay'] = 0

    if 'batch_size' not in self.gd_train_parameters:
        self.gd_train_parameters['batch_size'] = 512

    if 'lr' not in self.gd_train_parameters:
        self.gd_train_parameters['lr'] = 1e-2

    # store robust and adaptive robust models
    self.Robust_models = []
    self.missing_feat_leaf = []
    # keyword arguments for standard class object resilient_MLP
    num_features = X.shape[1]    #Number of features
    n_obs = len(Y)
    
    # Prepare supervised learning sets
    ####### Create train/validation data loaders for torch modules
    ### Train Nominal model (no missing data here)
    n_valid_obs = int(val_split*len(Y))
    if val_split == 0:    
        trainY = Y
        validY = Y        
        train_temp_X = X
        valid_temp_X = X   
    else:
        trainY = Y[:-n_valid_obs]
        validY = Y[-n_valid_obs:]

        train_temp_X = X[:-n_valid_obs]
        valid_temp_X = X[-n_valid_obs:]        

    tensor_trainY = torch.FloatTensor(trainY)
    tensor_validY = torch.FloatTensor(validY)
        
    tensor_train_X = torch.FloatTensor(train_temp_X)
    tensor_valid_X = torch.FloatTensor(valid_temp_X)

    train_data_loader = create_data_loader([tensor_train_X, tensor_trainY], batch_size = self.gd_train_parameters['batch_size'], shuffle = False)        
    valid_data_loader = create_data_loader([tensor_valid_X, tensor_validY], batch_size = self.gd_train_parameters['batch_size'], shuffle = False)
    
    print('Start training robust models...')
    
    for gamma in np.arange(self.Gamma_max + 1):
        self.missing_feat_leaf.append(gamma)
        print(f'Budget: {gamma}')
        
        # Declare robust model
        temp_fdr_model = Adaptive_LDR_Regression(input_size = num_features, hidden_sizes = self.adapt_regr_param['hidden_sizes'], 
                                                 output_size = self.adapt_regr_param['output_size'], target_col = self.target_col, fix_col = self.fix_col, projection = False, 
                                                 apply_LDR = self.adapt_regr_param['apply_LDR'], Gamma = gamma, budget_constraint = 'equality')
        
        optimizer = torch.optim.Adam(temp_fdr_model.parameters(), lr = self.gd_train_parameters['lr'], weight_decay = self.gd_train_parameters['weight_decay'])
        
        # If base model is linear, then warm-start with analytical solution// skip gradient descent
        if (gamma == 0) and (self.base_model == 'LR'):
            # Train Linear Regression model
            lr_model = LinearRegression(fit_intercept = True)
            lr_model.fit(train_temp_X, trainY)
                
            temp_fdr_model.model[0].weight.data = torch.FloatTensor(lr_model.coef_[0].reshape(1,-1))
            temp_fdr_model.model[0].bias.data = torch.FloatTensor(lr_model.intercept_)
            self.Robust_models.append(temp_fdr_model)
            continue
                    
        # Warm-start: Use nominal model or previous iteration
        if gamma > 0:            
            temp_fdr_model.load_state_dict(self.Robust_models[-1].state_dict(), strict=False)

        temp_fdr_model.adversarial_train_model(train_data_loader, valid_data_loader, optimizer, 
                                   epochs = self.gd_train_parameters['epochs'], patience = self.gd_train_parameters['patience'],
                                   verbose = self.gd_train_parameters['verbose'], attack_type = 'random_sample', 
                                   freeze_weights = False, warm_start_nominal = False)
    
        self.Robust_models.append(temp_fdr_model)
                              
  def predict(self, X, missing_mask):
     ''' Function to predict using a trained tree. Trees are fully compiled, i.e., 
     leaves correspond to predictive prescriptions 
     '''
     Predictions = []
     for i in range(X.shape[0]): 
         #New query point
         x0 = X[i:i+1,:]
         m0 = missing_mask[i:i+1,:]
         # Total missing features in leaf
         temp_total_missing_feat = m0.sum()

         Predictions.append( self.Robust_models[temp_total_missing_feat].predict(x0, m0).reshape(-1))
     return np.array(Predictions).reshape(-1,1)

###### Auxiliary functions

def eval_predictions(pred, target, metric = 'mae'):
    ''' Evaluates determinstic forecasts'''
    if metric == 'mae':
        return np.mean(np.abs(pred.reshape(-1)-target.reshape(-1)))
    elif metric == 'rmse':
        return np.sqrt(np.square(pred.reshape(-1)-target.reshape(-1)).mean())
    elif metric == 'mape':
        return np.mean(np.abs(pred.reshape(-1)-target.reshape(-1))/target)
    elif metric == 'mse':
        return np.square(pred.reshape(-1)-target.reshape(-1)).mean()
    
