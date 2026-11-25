# -*- coding: utf-8 -*-
"""
Torch custom layers and objects

@author: a.stratigakos
"""

# [wjb vendoring] removed unused import: import cvxpy as cp
import torch
from torch import nn
# [wjb vendoring] removed unused import: from cvxpylayers.torch import CvxpyLayer
from torch.utils.data import Dataset, DataLoader
import numpy as np
import copy
import time

def to_np(x):
    return x.detach().numpy()

# Define a custom dataset
class MyDataset(Dataset):
    def __init__(self, *inputs):
        self.inputs = inputs

        # Check that all input tensors have the same length (number of samples)
        self.length = len(inputs[0])
        if not all(len(input_tensor) == self.length for input_tensor in inputs):
            raise ValueError("Input tensors must have the same length.")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return tuple(input_tensor[idx] for input_tensor in self.inputs)

# Define a custom data loader
def create_data_loader(inputs, batch_size, num_workers=0, shuffle=True):
    dataset = MyDataset(*inputs)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle
    )
    return data_loader

def epoch_train(torch_model, loader, opt=None):
    """Standard training/evaluation epoch over the dataset"""
    total_loss = 0.
    
    for X,y in loader:        
        
        y_hat = torch_model.forward(X, torch.zeros_like(X, requires_grad = False))                    
        loss_i = torch_model.estimate_loss(y_hat, y)                    
        loss = torch.mean(loss_i)
        
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
        total_loss += loss.item() * X.shape[0]
        
    return total_loss / len(loader.dataset)

def adversarial_epoch_train(torch_model, loader, opt=None, attack_type = 'greedy'):
    """Adversarial training/evaluation epoch over the dataset"""
    total_loss = 0.
    
    for X,y in loader:            
        
        #### Find adversarial example of missing features
        if attack_type == 'greedy':
            # Greedy top-down heuristic (Algorithm 1), works best when budget_constraint == 'inequality'
            alpha = greedy_missing_data_attack(torch_model, X, y, attack_budget_gamma = torch_model.gamma)            

        elif attack_type == 'random_sample':
            # Uniform sampling: approximates well uncertainty set with budget equality constraint
            feat_col = np.random.choice(np.arange(len(torch_model.target_col)), replace = False, size = (torch_model.gamma))
            alpha = torch.zeros_like(X)
            for c in feat_col: 
                alpha[:,c] = 1
                
        elif attack_type == 'l1_norm':                
            # L1 attack with projected gradient descent            
            alpha = l1_norm_attack(torch_model, X, y) 
                
        ###!!!! forward pass plus correction
        y_hat = torch_model.forward(X, alpha)                    
        loss_i = torch_model.estimate_loss(y_hat, y)    
        loss = torch.mean(loss_i)

        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
        total_loss += loss.item() * X.shape[0]
                
    return total_loss / len(loader.dataset)

def greedy_missing_data_attack(torch_model, X, y, attack_budget_gamma = 1, perc = 0.1):
    """ Finds adversarial example, applies greedy missing data attack (Algorithm 1), 
        returns a vector of x*(1-a), if a_j == 1: x_j is missing"""
    
    # estimate nominal loss (no missing data)
    y_hat = torch_model.forward(X, torch.zeros_like(X, requires_grad=False))        
    current_loss = torch_model.estimate_loss(y_hat, y).mean()
    # initialize wc loss and adversarial example
    wc_loss = current_loss.data
    best_alpha =  torch.zeros_like(X)
    
    # Features that could go missing
    # !!!! When placing a model within a tree-based partition, we have features that are already missing once we reach this point
    # !!!! These features have already been multiplied by (1-a), hence are set to 0; the respective attack learned here does apply any correction
    #       as we assume these features were never 'observed' by this model to begin with
    current_target_col = torch_model.target_col
    
    # Iterate over gamma, greedily add one feature per iteration
    for g in range(attack_budget_gamma):    
        # store losses for all features
        local_loss = []
        # placeholders for splitting a column
        best_col = None
        apply_split = False  
        # !!!!! Important to use clone/copy here, else we update both
        alpha_init = torch.clone(best_alpha)   
        y_hat = torch_model.forward(X, alpha_init)

        # Nominal loss for this iteration using previous alpha values            
        temp_nominal_loss = torch_model.estimate_loss(y_hat, y).mean().data
        wc_loss = temp_nominal_loss
        
        # loop over target columns (for current node), find worst-case loss:
        for col in current_target_col:
            # create adversarial example
            alpha_temp = torch.clone(alpha_init)
            
            # set feature to missing
            alpha_temp[:,col] = 1
                            
            # predict using adversarial example
            with torch.no_grad():
                y_adv_hat = torch_model.forward(X, alpha_temp)
                
            temp_loss = torch_model.estimate_loss(y_adv_hat, y).mean().data
            local_loss.append(temp_loss.data)
                    
        best_col_ind = np.argsort(local_loss)[-1]                
        wc_loss = np.max(local_loss)
        best_col = current_target_col[best_col_ind]
        
        # !!!!! This approximates an equality constraint on the total budget
        if torch_model.budget_constraint == 'equality':
            best_alpha[:,best_col] = 1
            apply_split = True
        elif torch_model.budget_constraint == 'inequality':
            # check if performance degrades enough, apply split
            if wc_loss > temp_nominal_loss:
                # update
                best_alpha[:,best_col] = 1
                apply_split = True
        #update list of eligible columns
        if apply_split:
            current_target_col = torch.cat([current_target_col[0:best_col_ind], current_target_col[best_col_ind+1:]])   
        else:
            break
    
    return best_alpha

def l1_norm_attack(torch_model, X, y, num_iter = 10, randomize=False):
    
        # initialize with greedy heuristic search 
        # alpha_init = torch_model.missing_data_attack(X, y, gamma = self.gamma)            
        alpha = torch.zeros_like(X[0:1], requires_grad=True)           
        # alpha.data = alpha_init
        # proj_simplex = nn.Softmax()
        optimizer = torch.optim.SGD([alpha], lr=1e-2)
        # print(alpha)
        for t in range(num_iter):

            pred = torch_model.forward(X, alpha)
            
            # Maximize MSE loss == minimize negative loss
            negative_loss = -nn.MSELoss()(pred, y)
            optimizer.zero_grad()
            negative_loss.backward()
            optimizer.step()                

        alpha_proj = torch_model.projection_layer(alpha)[0]
        alpha.data = alpha_proj
        return alpha.detach()

class LDR_Layer(nn.Module):
    """ Linear layer with Linear Decision Rules
        Applies the same correction across all nodes of the hidden layer"""

    def __init__(self, size_in, size_out, dimension_alpha):
        super().__init__()
        # size_in, size_out: size of layer input & output, standard definitions
        # dimension_alpha: size of the masking vector, equals size_in of first layer

        self.size_in, self.size_out = size_in, size_out
        self.dimansion_alpha = dimension_alpha
        
        # Nominal weights
        weight = torch.Tensor(size_out, size_in)
        self.weight = torch.nn.Parameter(weight)
        bias = torch.Tensor(size_out)
        self.bias = torch.nn.Parameter(bias)
        
        # Linear Decision Rules (initialize to zero)
        self.W = nn.Parameter(torch.FloatTensor(np.zeros((size_in, dimension_alpha))).requires_grad_())
        
        # initialize nominal weights and biases
        torch.nn.init.kaiming_uniform_(self.weight, a=torch.math.sqrt(5)) # weight init
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / torch.math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init
    
    def forward(self, x, a):
        """
        Forward pass
        Args:
            x: input tensors/ features
            a: binaries for missing data
        """        
        # !!!!! x: is the output of the previous layer
        # !!!!! a: binary with the size of the original feature vector    
        # !!!!! The elementwise product x*(1-a) is implemented outside of the LDR_Layer
        
        return ( (self.weight@x.T).T + self.bias)  + torch.sum( (self.W@a.T).T*(x), dim = 1).reshape(-1,1)
    
class Adaptive_LDR_Regression(nn.Module):        
    def __init__(self, input_size, hidden_sizes, output_size, target_col, fix_col, activation=nn.ReLU(), apply_LDR = True, 
                 projection = False, UB = 1, LB = 0, Gamma = 1, train_adversarially = True, budget_constraint = 'inequality'):
        super(Adaptive_LDR_Regression, self).__init__()
        """
        Adaptive Robust Regression with Linear Decision Rules
        Args:
            input_size, hidden_sizes, output_size: Standard arguments for declaring an MLP. If hidden_size == [], then we have a linear model
            activation: Activation function for hidden layers
            apply_LDR (boolean): If True, then applies LDR at each layer. If False, then we get a (static) robust model
            projection (boolean): Projection onto a feasible box, applied only on the prediction step
            UB, LB: Support of the target variable for the projection step
            budget_constraint: {'inequality', 'equality'}, sets the budget constraint in the uncertainty set
        """
        # Initialize learnable weight parameters
        self.num_features = input_size
        self.dimension_alpha_mask = input_size # Dimension of vector of binary variables that model missing data
        self.output_size = output_size
        self.apply_LDR = apply_LDR # False: Do not apply LDRs, equivalent to robust regression
        self.projection = projection # Project forecasts back to feasible set
        self.UB = torch.FloatTensor([UB]) # Support for random variable
        self.LB = torch.FloatTensor([LB])
        self.target_col = torch.tensor(target_col, dtype=torch.int32)   # Columns that could be missing
        self.fix_col = torch.tensor(fix_col, dtype=torch.int32) # Columns that are always available
        self.train_adversarially = train_adversarially 
        self.gamma = Gamma # Budget of uncertainty
        self.budget_constraint = budget_constraint # {'inequality', 'equality'} determines the budget uncertainty set
        
        # create sequential model
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        layers = []
        
        for i in range(len(layer_sizes) - 1):
            if apply_LDR == True:
                # Custom linear layer with linear decision rules
                layers.append(LDR_Layer(layer_sizes[i], layer_sizes[i + 1], self.dimension_alpha_mask))            
            else:
                # Standard linear layer
                layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))            
            
            if i < len(layer_sizes) - 2:
                layers.append(activation)
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x, a):
        """
        Forward pass
        Args:
            x: input tensors/ features
            a: binary to model missing data (a==1, feature is missing), same size as x
        """
        # First LDR layer
        x_imp = x*(1-a)        
        h_inter = x_imp.clone()
        if self.apply_LDR:
            for i in range(len(self.model)):
                if i % 2 == 0:
                    # LDR layer            
                    h_inter = self.model[i].forward(h_inter, a)
                else:
                    h_inter = self.model[i].forward(h_inter)
            return h_inter
        else:
            return self.model(x_imp)
    
    def predict(self, X, alpha, project = True):
        # used for inference only
        #!!!!!! X is zero-imputed already but X*(1-a) == X, so there is no error
        if torch.is_tensor(X):
            temp_X = X
        else:
            temp_X = torch.FloatTensor(X.copy())

        if torch.is_tensor(alpha):
            temp_alpha = alpha
        else:
            temp_alpha = torch.FloatTensor(alpha.copy())

        with torch.no_grad():     
            y_hat = self.forward(temp_X, temp_alpha)                            
            if self.projection or project:
                return (torch.maximum(torch.minimum(y_hat, self.UB), self.LB)).detach().numpy()
            else:
                return y_hat
            
    def estimate_loss(self, y_hat, y_target):
        # estimate custom loss function, *elementwise*
        mse_i = torch.square(y_target - y_hat)        
        loss_i = torch.sum(mse_i, 1)
        return loss_i

    def adversarial_train_model(self, train_loader, val_loader, 
                               optimizer, epochs = 20, patience=5, verbose = 0, warm_start_nominal = True, 
                               freeze_weights = False, attack_type = 'greedy'):
        ''' Adversarial training to learn linear decision rules.
            Assumes pre-trained weights are passed to the nominal model, only used for speed-up'''
        best_train_loss = float('inf')
        best_val_loss = float('inf')
        early_stopping_counter = 0
        best_weights = copy.deepcopy(self.state_dict())
                    
        if warm_start_nominal:
            
            print('Train model for nominal case// Warm-start adversarial model')            
            for epoch in range(epochs):

                average_train_loss = epoch_train(self, train_loader, optimizer)
                val_loss = epoch_train(self, val_loader)

                if (verbose != -1) and (epoch%25 == 0):
                    print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {average_train_loss:.4f} - Val Loss: {val_loss:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_weights = copy.deepcopy(self.state_dict())
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                    if early_stopping_counter >= patience:
                        print("Early stopping triggered.")
                        # recover best weights
                        self.load_state_dict(best_weights)
                        break
        
        if (freeze_weights==True)and(self.apply_LDR==False):
            print('Freeze nominal layer weights')
            for layer in self.model.children():
                if isinstance(layer, nn.Linear):
                    layer.weight.requires_grad = False
                    layer.bias.requires_grad = False
                else:
                    layer.weight.requires_grad = False
                    layer.bias.requires_grad = False


        print('Adversarial training')
        best_train_loss = float('inf')
        best_val_loss = float('inf')
        early_stopping_counter = 0
        best_weights = copy.deepcopy(self.state_dict())

        for epoch in range(epochs):
            average_train_loss = adversarial_epoch_train(self, train_loader, optimizer, attack_type)      
            val_loss = adversarial_epoch_train(self, val_loader, None, attack_type)

            if (verbose != -1)and(epoch%10 == 0):
                print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {average_train_loss:.4f} - Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = copy.deepcopy(self.state_dict())
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= patience:
                    print("Early stopping triggered.")
                    break
        # recover best weights
        self.load_state_dict(best_weights)
        self.best_val_loss = best_val_loss
        return    

    def train_model(self, train_loader, val_loader,  optimizer, epochs = 20, patience=5, verbose = 0):
        ''' Normal model training'''
        best_train_loss = float('inf')
        best_val_loss = float('inf')
        early_stopping_counter = 0
        best_weights = copy.deepcopy(self.state_dict())
                                
        print('Train model for nominal case')            
        for epoch in range(epochs):

            average_train_loss = epoch_train(self, train_loader, optimizer)
            val_loss = epoch_train(self, val_loader)

            if (verbose != -1) and (epoch%25 == 0):
                print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {average_train_loss:.4f} - Val Loss: {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = copy.deepcopy(self.state_dict())
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= patience:
                    print("Early stopping triggered.")
                    break
        # recover best weights
        self.load_state_dict(best_weights)
        self.best_val_loss = best_val_loss
        return        
 
##################################################################


        
class MLP(nn.Module):        
    def __init__(self, input_size, hidden_sizes, output_size, activation=nn.ReLU(), sigmoid_activation = False, 
                 projection = False, UB = 1, LB = 0):
        super(MLP, self).__init__()
        """
        Standard MLP for regression
        Args:
            input_size, hidden_sizes, output_size: standard arguments for declaring an MLP
            
            output_size: equal to the number of combination weights, i.e., number of experts we want to combine
            sigmoid_activation: enable sigmoid function as a final layer, to ensure output is in [0,1]
            
        """
        # Initialize learnable weight parameters
        self.num_features = input_size
        self.output_size = output_size
        self.sigmoid_activation = sigmoid_activation
        self.projection = projection
        self.UB = torch.FloatTensor([UB])
        self.LB = torch.FloatTensor([LB])
        

        # create sequential model
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(activation)
                    
        self.model = nn.Sequential(*layers)
        if (self.sigmoid_activation) and (self.projection == False):
            self.model.add_module('sigmoid', nn.Sigmoid())
                    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: input tensors/ features
        """
        # if self.projection:
        #     return (torch.maximum(torch.minimum( self.model(x), self.UB), self.LB))
        # else:
        return self.model(x)
            
    def epoch_train(self, loader, opt=None):
        """Standard training/evaluation epoch over the dataset"""
        total_loss = 0.
        
        for X,y in loader:
            
            y_hat = self.forward(X)
            
            #loss = nn.MSELoss()(yp,y)
            loss_i = self.estimate_loss(y_hat, y)                    
            loss = torch.mean(loss_i)
            
            if opt:
                opt.zero_grad()
                loss.backward()
                opt.step()
            total_loss += loss.item() * X.shape[0]
            
        return total_loss / len(loader.dataset)
    
    def predict(self, x):
        # used for inference only
        if torch.is_tensor(x):
            with torch.no_grad():         
                if self.projection:
                    return (torch.maximum(torch.minimum( self.model(x), self.UB), self.LB)).detach().numpy()
                else:
                    return self.model(x).detach().numpy()
        else:
            tensor_x = torch.FloatTensor(x.copy())
            with torch.no_grad():         
                if self.projection:
                    return (torch.maximum(torch.minimum( self.model(tensor_x), self.UB), self.LB)).detach().numpy()
                else:
                    return self.model(tensor_x).detach().numpy()
            
    def estimate_loss(self, y_hat, y_target):
        
        # estimate custom loss function, *elementwise*
        mse_i = torch.square(y_target - y_hat)        
        loss_i = torch.sum(mse_i, 1)
        
        return loss_i
            
    def train_model(self, train_loader, val_loader, optimizer, epochs = 20, patience=5, verbose = 0):
        
        best_train_loss = float('inf')
        best_val_loss = float('inf')
        early_stopping_counter = 0
        best_weights = copy.deepcopy(self.state_dict())
        
        for epoch in range(epochs):

            average_train_loss = self.epoch_train(train_loader, optimizer)
            val_loss = self.epoch_train(val_loader)
            if verbose != -1:
                print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {average_train_loss:.4f} - Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = copy.deepcopy(self.state_dict())
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= patience:
                    print("Early stopping triggered.")
                    # recover best weights
                    self.load_state_dict(best_weights)
                    return

    def evaluate(self, data_loader):
        # evaluate loss criterion/ used for estimating validation loss
        self.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch_data in data_loader:
                
                # features, wind, and generator data for each batch 
                x_batch = batch_data[0]
                y_batch = batch_data[1]
                
                # forward pass: predict weights and combine forecasts
                output_hat = self.forward(x_batch)
                
                y_hat = output_hat
                                
                loss_i = self.estimate_loss(y_hat, y_batch)
                
                loss = torch.mean(loss_i)
                total_loss += loss.item()
                
        average_loss = total_loss / len(data_loader)
        return average_loss

def project_onto_l1_ball(x, eps):
    """
    Compute Euclidean projection onto the L1 ball for a batch.
    
      min ||x - u||_2 s.t. ||u||_1 <= eps
    
    Inspired by the corresponding numpy version by Adrien Gaidon.
    
    Parameters
    ----------
    x: (batch_size, *) torch array
      batch of arbitrary-size tensors to project, possibly on GPU
      
    eps: float
      radius of l-1 ball to project onto
    
    Returns
    -------
    u: (batch_size, *) torch array
      batch of projected tensors, reshaped to match the original
    
    Notes
    -----
    The complexity of this algorithm is in O(dlogd) as it involves sorting x.
    
    References
    ----------
    [1] Efficient Projections onto the l1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
    """
    original_shape = x.shape
    x = x.view(x.shape[0], -1)
    mask = (torch.norm(x, p=1, dim=1) < eps).float().unsqueeze(1)
    mu, _ = torch.sort(torch.abs(x), dim=1, descending=True)
    cumsum = torch.cumsum(mu, dim=1)
    arange = torch.arange(1, x.shape[1] + 1, device=x.device)
    rho, _ = torch.max((mu * arange > (cumsum - eps)) * arange, dim=1)
    theta = (cumsum[torch.arange(x.shape[0]), rho.cpu() - 1] - eps) / rho
    proj = (torch.abs(x) - theta.unsqueeze(1)).clamp(min=0)
    x = mask * x + (1 - mask) * proj * torch.sign(x)
    return x.view(original_shape)
