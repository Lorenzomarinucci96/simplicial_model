import random
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
from tqdm import tqdm
from IPython.display import display, clear_output
import os
import cvxpy as cp

def norma_spettrale(matrix):
    # Calcola gli autovalori della matrice
    eigenvalues = np.linalg.eigvals(matrix)
    # Trova il massimo valore assoluto degli autovalori
    spectral_norm = np.max(np.abs(eigenvalues))
    return spectral_norm


def compute_polynomial_matrix(A1,A2, coefficients):
    """
    Computes the polynomial of two square matrices A1 and A2 given the coefficients of the polynomial.

    Parameters:
    - A1: Square numpy matrix.
    - A2: Square numpy matrix.
    - coefficients: List of coefficients [h_0, h_1, ..., h_{2M+1}] of the polynomial.

    Returns:
    - result: Numpy matrix, the result of the polynomial applied to A.
    """
    n = A1.shape[0]
    result = np.zeros_like(A1, dtype=float)
    power_A1 = A1  # Start with A1^1
    power_A2 = A2  # Start with A2^1
    for i in range(1,M+1):
        result = coefficients[i] * power_A1 + coefficients[M+i] * power_A2 + result
        power_A1 = power_A1@A1
        power_A2 = power_A2@A2
    # Add h_0*I at the end
    result += coefficients[0] * np.eye(n)
    return result

def sampling_matrix(n, p=0.7, percentage_cut=0):
    """
    Generates a random diagonal binary matrix of dimension n x n. The first n-cut
    elements of the diagonal are either 0 or 1, with the probability of 1 being p,
    while the others are zeros.

    Parameters:
    - n: The dimension of the matrix (int).
    - p: The probability of a diagonal element being 1 (float between 0 and 1).
    - percentage_cut: Percentage of zeros in the diagonal

    Returns:
    - A numpy array of shape (n, n) representing the diagonal binary matrix.
    """
    if type(p)==float: P=[p]*n
    else: P=p
    cut=int(percentage_cut*n)
    # Generate a 1D array of n with n-cut random binary values (0 or 1)  and the remaining entries equal to zero
    diagonal_elements=np.zeros(n)
    for i in range(n-cut):
     diagonal_elements[i] = np.random.choice([0, 1], p=[1-P[i], P[i]])

    # Create a diagonal matrix from the 1D array
    return np.diag(diagonal_elements)


def compute_incidence_matrices(simplicial_complex):
    """
    Computes incidence matrices for a given simplicial complex.

    Parameters:
    - simplicial_complex: A dictionary where keys are dimensions and values are lists of simplices in that dimension.

    Returns:
    - incidence_matrices: A dictionary of incidence matrices for the simplicial complex, indexed by simplex dimension.
    """
    incidence_matrices = {}
    for dim in range(1, max(len(simplicial_complex), 2)):
        if dim not in simplicial_complex:
            continue

        num_simplices_dim = len(simplicial_complex[dim])
        num_simplices_prev_dim = len(simplicial_complex[dim - 1])
        incidence_matrix = np.zeros((num_simplices_prev_dim, num_simplices_dim))

        for i, simplex_higher in enumerate(simplicial_complex[dim]):
            for j, simplex_lower in enumerate(simplicial_complex[dim - 1]):
                # Check if simplex_lower is a face of simplex_higher
                if set(simplex_lower).issubset(simplex_higher):
                    # Calculate orientation
                    orientation = 1
                    for vertex in simplex_lower:
                        if simplex_higher.index(vertex) % 2 != 0:
                            orientation *= -1
                    incidence_matrix[j, i] = orientation

        incidence_matrices[dim] = incidence_matrix

    return incidence_matrices

def generate_simplicial_complex(avg_nodes, max_dimension, p, variance, load_path = None, save = True):
    """
    Generates a random simplicial complex based on the provided parameters.

    Parameters:
    - avg_nodes: Average number of nodes.
    - variance: Variance for the number of nodes.
    - max_dimension: Maximum dimension of simplices to generate.
    - p: Probability vector for simplex inclusion.
    - load_path: If different from None, tries to load a pickle containing a simplicial complex. Useful for keeping the complex fixed across different simulations.
    - save: If true, saves the output variable in the current path as a pickle.

    Returns:
    - simplicial_complex: A dictionary where keys are dimensions and values are lists of simplices in that dimension.
    - incidence_matrices: A dictionary of incidence matrices for the simplicial complex, indexed by simplex dimension.
    - tot_num_simplices: The total number of simplices (node+edges+triangles+...)
    """
    done = 0
    tried = 0
    while not done:
      if load_path is None or tried == 1:
        num_nodes = int(np.random.normal(avg_nodes, variance,1)[0])
        nodes = range(num_nodes)
        simplicial_complex = {0: [tuple([node]) for node in nodes]}
        all_order_present = 0
        while all_order_present == 0:
          tot_num_simplices = 0
          all_order_present = 1
          for dim in range(1, max_dimension + 1):
              simplicial_complex[dim] = []
              num_simplices_dim = 0
              for simplex in itertools.combinations(nodes, dim + 1):
                  if all(tuple(sorted(face)) in simplicial_complex[dim - 1] for face in itertools.combinations(simplex, dim)):
                      if random.random() < p[dim-1]:  # Adjust probability as needed
                          simplicial_complex[dim].append(tuple(simplex))
                          tot_num_simplices += 1
                          num_simplices_dim += 1
              if num_simplices_dim == 0:
                all_order_present = 0
        tot_num_simplices += num_nodes
        incidence_matrices = compute_incidence_matrices(simplicial_complex)
        if save:
          with open('simplicial_complex.pkl', 'wb') as f:
            pkl.dump([simplicial_complex, incidence_matrices, tot_num_simplices], f)
          print("Complex correctly saved on the current path, veryfing if valid...")
        done = 1
        return simplicial_complex, incidence_matrices, tot_num_simplices
      if load_path is not None and tried == 0:
        try:
          with open(load_path+'/simplicial_complex.pkl','rb') as f:
            simplicial_complex, incidence_matrices, tot_num_simplices = pkl.load(f)
          print("Found an existing complex...Loaded!")
          done = 1
          return simplicial_complex, incidence_matrices, tot_num_simplices
        except:
          print("No complex found at the inserted path!")
          load_path = None
          tried = 1

def create_groundtruth_filter(incidence_mats, M):
  """
  Generates a ground truth filter based on the incidence matrices of a simplicial complex.

  Parameters:
  - incidence_mats: Incidence matrices of a simplicial complex.
  - M: The maximum order of the polynomial filter.
 
  Returns:
  - h_true: The vector of ground truth coefficients for the polynomial filter.
  """

  
  h_true = np.random.normal(0,1,2*M+1)

  return h_true

""" old version


  def create_cov_matrix_noise(n_edges, noise_array):
    
    Generates the diagonal covariance matrix of the noise random vectors

    Parameters:
    - n_edges: Number of edges of the complex.
    - noise_array: Possible variances of noises vector.

    Return:
    - cov_matrix_noise: Random covariance matrix of indipendent noises.
    
    diagonal=np.zeros(n_edges)
    if len(noise_array)==1:
       diagonal[:] = noise_array[0]
    else:
     diagonal[: n_edges-int(n_edges*.7)] = noise_array[0] 
     diagonal[n_edges-int(n_edges*.7) : n_edges-int(n_edges*.5)] = noise_array[1]
     diagonal[n_edges-int(n_edges*.5) :] = noise_array[2]
    #cov_matrix_noise = np.diag(random.choices(noise_array, k=n_edges))
    cov_matrix_noise=np.diag(diagonal)
    return cov_matrix_noise
"""

##new version 

def create_cov_matrix_noise(n_edges, noise_array,f_v=False):
   
  diagonal_elements = np.random.choice(noise_array, size = n_edges)
  if f_v == True: 
    diagonal_elements[:int(0.4 * n_edges)]=(int(0.4 * n_edges))*[noise_array[0]]
    diagonal_elements[int(0.4 * n_edges):int(0.6 * n_edges)]=(int(0.6 * n_edges)-int(0.4 * n_edges))*[noise_array[1]]
    diagonal_elements[int(0.6 * n_edges):int(0.8 * n_edges)]=(int(0.8 * n_edges)-int(0.6 * n_edges))*[noise_array[2]]
    diagonal_elements[int(0.8 * n_edges):]=(n_edges-int(0.8 * n_edges))*[noise_array[3]]
   
  return np.diag(diagonal_elements)
   

def create_signal(incidence_matrices, n, M, h_true, p_sampling, cov_matrix_noise, percentage_cut = 0,norm=False):
  """
  Simulates a signal on the edges of a simplicial complex using a delayed/shifted polynomial filter.

  Parameters:
  - incidence_matrices: Incidence matrices of a simplicial complex.
  - n: Number of time steps to simulate.
  - M: Order of the polynomial filter.
  - h_true: Ground truth coefficients of the polynomial filter.
  - p_sampling: Probability of sampling a variable.
  - percentage_cut: Percentage of unused variables.
  - cov_matrix_noise: Covariance matrix of the noise.

  Returns:
  - Y: The observations.
  - X: The shifted signals.
  - D: The sampling operators.
  """
  #print("Generating data...")
  n_edges = incidence_matrices[1].shape[1]
  L_d = incidence_matrices[1].T@incidence_matrices[1]

  if norm == True: L_d=L_d/norma_spettrale(L_d)
  L_u = incidence_matrices[2]@incidence_matrices[2].T

  if norm == True: L_u=L_u/norma_spettrale(L_u)
  Y = np.zeros((n,n_edges))
  X_no_shift = np.random.normal(0,1,(n_edges,M+n))
  X_shift = np.zeros((n,n_edges,2*M+1))
  D = np.zeros((n, n_edges, n_edges))
  mean_noise=np.zeros(n_edges)
  L_d_vector=[]
  L_u_vector=[]
  for i in range(1,M+1):
   L_d_vector.append(np.linalg.matrix_power(L_d, i))
   L_u_vector.append(np.linalg.matrix_power(L_u, i))
  for t in range(M,M+n):
    for i in range(1,M+1):
      X_shift[t-M,:,i] = L_d_vector[i-1]@X_no_shift[:,t-i]
      X_shift[t-M,:,i+M] = L_u_vector[i-1]@X_no_shift[:,t-i]
    X_shift[t-M,:,0] = X_no_shift[:,t]
    D[t-M,:,:] = sampling_matrix(n_edges, p_sampling, percentage_cut)
    Y[t-M,:] = D[t-M,:,:]@(X_shift[t-M,:,:]@h_true + np.random.multivariate_normal(mean_noise,cov_matrix_noise))
  #print("...Done!")
  return Y, X_shift, D


def generate_dataset(n_data_points, max_num_simplices, min_num_simplices, avg_nodes, max_dimension, \
                     p_complex, variance_complex, M, p_sampling, sigma_noise = None, cov_matrix_noise = None, save = True, percentage_cut = 0,load_path = None, normalize=False, fixed_variances=False):
  """
  Generates a dataset consisting of signals on simplicial complexes obtained through Eq. (5).

  Parameters:
  - n_data_points: Number of time steps to simulate., i.e. number of data points to generate.
  - max_num_simplices: Maximum number of simplices in the complexes.
  - min_num_simplices: Minimum number of simplices in the complexes.
  - avg_nodes: Average number of nodes in the simplicial complexes.
  - max_dimension: Maximum dimension of simplices in the simplicial complexes.
  - p_complex: Probability vector for simplex inclusion in each dimension.
  - variance_complex: Variance for the number of nodes.
  - M: Order of the polynomial filter.
  - p_sampling: probability of sampling an edge, i.e. probability of having a 1 on the diagonal of D(n)
  - sigma_noise: array of possible noise variances. It is only used if cov_matrix_noise is None to generate a new covariance matrix for the noises
  - cov_matrix_noise: Covariance matrix of the noise.
  - save: If true, saves the output simplex variable in the current path as a pickle.
  - percentage_cut: Percentage of unused variables.

  Returns:
  - data: A dictionary containing the generated dataset, including incidence matrices, the simulated signals, ground truth filter coefficients, the total number of simplices and the covariance matrix of the noise.
  """
  valid_data_point = 0
  data = []
  valid_complex = 0
  while not valid_complex:
      _, incidence_mats, num_simplices = generate_simplicial_complex(avg_nodes, max_dimension, p_complex, variance_complex, load_path, save)
      if num_simplices < max_num_simplices and num_simplices >= min_num_simplices:
        valid_complex = 1
        print("Valid complex generated/loaded!")
      else:
       print("The generated complex does not respect the imposed constraints... Trying again!")
  h_true = create_groundtruth_filter(incidence_mats, M)
  n_edges = incidence_mats[1].shape[1]
  if cov_matrix_noise is None:
    cov_matrix_noise = create_cov_matrix_noise(n_edges, sigma_noise,f_v=fixed_variances)

  Y, X_shift, D = create_signal(incidence_mats,n_data_points, M, h_true, p_sampling, cov_matrix_noise, percentage_cut, norm = normalize)
  data = {"incidence_matrices":incidence_mats, "Y": Y, "X": X_shift, "D": D, \
               "h_true": h_true, "num_simplices": num_simplices, "cov_matrix_noise": cov_matrix_noise}
  return data

def LMS(step_size, Y, X, D, h_true, S = None):
  """
  This function implements the Topological Least Mean Squares (TopoLMS) algorithm in Algorithm 1.

  Parameters:
  - step_size: A float representing the step size or learning rate of the LMS algorithm.
  - Y: A numpy array of shape (n_samples, n_outputs) representing the desired signal(s). n_samples is the number of samples, and n_outputs is the dimensionality of the output space.
  - Y: The observations.
  - X: The shifted signals.
  - D: The sampling operators.
  - h_true: The vector of ground truth coefficients for the polynomial filter.
  - S: weights of the scalar product of the norm. If None, S is considered to be the Identity matrix
  Returns:
  - h: The estimated coefficients after the final iteration.
  - MSE: A list containing the mean squared errors between the estimated coefficients and the true coefficients at each iteration.
  """
  h = np.zeros(X.shape[2])
  MSE = []
  if S is None:
     for t in tqdm(range(Y.shape[0])):
      h = h + step_size*X[t,:,:].T@D[t,:,:]@(Y[t,:]-X[t,:,:]@h)
      MSE.append(np.linalg.norm(h-h_true)**2)
  else:
     for t in tqdm(range(Y.shape[0])):
      h = h + step_size*X[t,:,:].T@D[t,:,:]@(Y[t,:]-X[t,:,:]@h)
      MSE.append((h-h_true).T@S@(h-h_true))

  return h,MSE




def evaluate_R_X(incidence_matrices, p_sampling, M, percentage_cut=0, normalize = False):
     """
     Parameters:
     -incidence_matrices: Incidence matrices of the complex
     -p_sampling: Probability of the diagonal elements of the sampling matrix to be one
     -M: Order of the polynomial filter
     -percentage_cut: percentage of unused variables
     Returns:
     not singular: If True, the covariance matrix isn't singular
     -R_X: Covariance matrix of the filtered regressors
     """
     n_edges = incidence_matrices[1].shape[1]
     if type(p_sampling)==float: p_sampling=[p_sampling]*n_edges
     cut=int(percentage_cut*n_edges)
     diagonal_elements=np.zeros(n_edges)
     diagonal_elements[:n_edges-cut]=p_sampling[:n_edges-cut]
     P=np.diag(diagonal_elements)
     
     L_d = incidence_matrices[1].T@incidence_matrices[1]

     if normalize == True: L_d=L_d/norma_spettrale(L_d)
     L_u = incidence_matrices[2]@incidence_matrices[2].T

     if normalize == True: L_u=L_u/norma_spettrale(L_u)
     R_X=np.zeros((2*M+1,2*M+1))
     R_X[0][0]=np.trace(P)
     for k in range (1,M+1):
          Ld_power=np.linalg.matrix_power(L_d, k)
          Lu_power=np.linalg.matrix_power(L_u, k)

          R_X[k][k]=np.trace(Lu_power.T @ P @ Lu_power)
          R_X[k+M][k+M]=np.trace(Ld_power.T @ P @ Ld_power)
          
          R_X[k][k+M]=np.trace(Lu_power.T @ P @ Ld_power)
          R_X[k+M][k]=R_X[k][k+M]
     if (np.linalg.matrix_rank(R_X)==R_X.shape[1]): not_singular=1 #True if R_X is not singular
     else: not_singular=0
     return not_singular, R_X





def evaluate_MSE_limit(R_X, mu, p_sampling, M, incidence_matrices, cov_matrix_noise, percentage_cut=0, S = None,normalize = False):
  """
  Evaluates the theorical limit of the MSE of the algorithm

  Parameters:
  - R_X: Covariance matrix of the filtered regressors.
  - mu: A float representing the step size or learning rate of the LMS algorithm.
  - p_sampling: Probability of the diagonal elements of the sampling matrix to be one
  - incidence_matrices: Incidence matrices of the complex.
  - percentage_cut: percentage of unused variables.
  - cov_matrix_noise: Covariance matrix of the noise.
  - S: weights of the scalar product of the norm. If None, S is considered to be the Identity matrix
  Return:
  - MSE_limit: theorical limit of the MSE expected value
  """

  B=np.eye(2*M+1)-mu*R_X
  n_edges=incidence_matrices[1].shape[1]
  if type(p_sampling)==float: p_sampling=[p_sampling]*n_edges
  cut=int(percentage_cut*n_edges)
  diagonal_elements=np.zeros(n_edges)
  diagonal_elements[:n_edges-cut]=p_sampling[:n_edges-cut]
  P=np.diag(diagonal_elements)
  
  L_d = incidence_matrices[1].T@incidence_matrices[1]
  
  if normalize == True: L_d=L_d/norma_spettrale(L_d)
  L_u = incidence_matrices[2]@incidence_matrices[2].T
  
  if normalize == True: L_u=L_u/norma_spettrale(L_u)
  G = np.zeros((2*M+1,2*M+1))
  G[0][0]=np.trace(P @ cov_matrix_noise)

  for k in range (1,M+1):
          Ld_power = np.linalg.matrix_power(L_d, k)
          Lu_power = np.linalg.matrix_power(L_u, k)

          G[k][k]=np.trace(Lu_power.T @ P @ cov_matrix_noise @ Lu_power)
          G[k+M][k+M]=np.trace(Ld_power.T @ P @ cov_matrix_noise @ Ld_power)
          
          G[k][k+M]=np.trace(Lu_power.T @ P @ cov_matrix_noise @ Ld_power)
          G[k+M][k]=G[k][k+M]

  F = np.kron(B.T,B.T)
  if S is None:
   S_flattened = np.eye(B.shape[1]).flatten(order='F')
  else:
   S_flattened = S.flatten(order='F')
  g = G.flatten(order='F')
  MSE_limit = mu*mu*g.T@np.linalg.inv(np.eye(F.shape[1])-F)@S_flattened


  return MSE_limit,G



   ############## Distributed functions


def LMS_distributed(step_size, data,edge_wise_MSE = False):
   """
   This function implements the distributed Topological Least Mean Squares (TopoLMS) algorithm in Algorithm 1.

   Parameters:
   - step_size: A float representing the step size or learning rate of the LMS algorithm.
   - data: Dictionary containing the features of the model
   Returns:
   - h: The estimated coefficients after the final iteration.
   - MSE: A list containing the mean squared errors between the estimated coefficients and the true coefficients at each iteration.
   """
   MSE = []
   res = [] #square errors vector
   X=data["X"]
   Y=data["Y"]
   h_true=data["h_true"]
   D=data["D"]  
   h = np.zeros(X[0].shape).T
   p= np.zeros(X[0].shape).T
   L_d=data["incidence_matrices"][1].T @ data["incidence_matrices"][1]
   A=np.where(L_d != 0, 1, 0) # neighborhood matrix
   column_sum = np.sum(A, axis=0)
   A=A/column_sum #normalization, such that the columns sum at 1 
   for t in tqdm(range(Y.shape[0])):
      for k in tqdm(range(Y.shape[1])):
        p[:,k] = h[:,k] + step_size*D[t,k,k]*X[t,k,:].T*(Y[t,k]-X[t,k,:]@h[:,k]) #adapt
     
      for j in tqdm(range(Y.shape[1])):
        h[:,j] = p @ A[:,j] #combine
        #h[:,j] = np.sum(h[:,A[:,j]!=0],axis=1)/column_sum[j] combine 
        res.append(np.linalg.norm(h[:,j]-h_true)**2)
      
      if edge_wise_MSE == False : MSE.append(np.sum(res))
      else: MSE.append(res)
      res=[]
      clear_output()
   return h , MSE, A.T


def evaluate_R_X_distributed(incidence_matrices, p_sampling, M, normalize = False):
     """
     Parameters:
     -incidence_matrices: Incidence matrices of the complex
     -p_sampling: Probability of the diagonal elements of the sampling matrix to be one
     -M: Order of the polynomial filter
     -normalize: True if the laplacians are normalized
     Returns:
     -R_X: Covariance matrix of the filtered regressors
     """
     n_edges = incidence_matrices[1].shape[1]
     if type(p_sampling)==float: p_sampling=[p_sampling]*n_edges
     
     L_d = incidence_matrices[1].T@incidence_matrices[1]

     if normalize == True: L_d=L_d/norma_spettrale(L_d)
     L_u = incidence_matrices[2]@incidence_matrices[2].T

     if normalize == True: L_u=L_u/norma_spettrale(L_u)
     R_X=np.zeros(((2*M+1)*n_edges,(2*M+1)*n_edges))
     for j in range(n_edges):
       R_X_j=np.zeros((2*M+1,2*M+1))
       R_X_j[0][0]=p_sampling[j]
       for k in range (1,M+1):
          Ld_power=np.linalg.matrix_power(L_d, k)
          Lu_power=np.linalg.matrix_power(L_u, k)

          R_X_j[k][k]=p_sampling[j]*Ld_power[j,:].T @ Ld_power[j,:]
          R_X_j[k+M][k+M]=p_sampling[j]*Lu_power[j,:].T @ Lu_power[j,:]
       R_X[j*(2*M+1):(j+1)*(2*M+1),j*(2*M+1):(j+1)*(2*M+1)] = R_X_j  
     
     return R_X




def evaluate_MSE_limit_distributed(R_X, mu, p_sampling, M, incidence_matrices, cov_matrix_noise, A,normalize = False,S = None):
  """
  Evaluates the theorical limit of the MSE of the algorithm

  Parameters:
  - R_X: Covariance matrix of the filtered regressors.
  - mu: A float representing the step size or learning rate of the LMS algorithm.
  - p_sampling: Probability of the diagonal elements of the sampling matrix to be one
  - incidence_matrices: Incidence matrices of the complex.
  - cov_matrix_noise: Covariance matrix of the noise.
  - S: weights of the scalar product of the norm. If None, S is considered to be the Identity matrix
  Return:
  - MSE_limit: theorical limit of the MSE expected value
  """
  n_edges=incidence_matrices[1].shape[1]
  A=np.kron(A,np.eye(2*M+1))
  B=A @ (np.eye((2*M+1)*n_edges)-mu*R_X)
  if type(p_sampling)==float: p_sampling=[p_sampling]*n_edges
  
  L_d = incidence_matrices[1].T@incidence_matrices[1]
  
  if normalize == True: L_d=L_d/norma_spettrale(L_d)
  L_u = incidence_matrices[2]@incidence_matrices[2].T
  
  if normalize == True: L_u=L_u/norma_spettrale(L_u)
  
  G=np.zeros(((2*M+1)*n_edges,(2*M+1)*n_edges))
  for j in range(n_edges):
    G_j=np.zeros((2*M+1,2*M+1))
    G_j[0][0]=p_sampling[j]*cov_matrix_noise[j][j]

    for k in range (1,M+1):
          Ld_power = np.linalg.matrix_power(L_d, k)
          Lu_power = np.linalg.matrix_power(L_u, k)
          
          G_j[k][k]=p_sampling[j]*cov_matrix_noise[j,j] *Ld_power[j,:].T @ Ld_power[j,:]
          G_j[k+M][k+M]=p_sampling[j]*cov_matrix_noise[j,j] *Lu_power[j,:].T @ Lu_power[j,:]
    G[j*(2*M+1):(j+1)*(2*M+1),j*(2*M+1):(j+1)*(2*M+1)] = G_j


  if S is None: MSE_limit = mu*mu*np.trace(A @ G @ A.T @ np.linalg.inv(np.eye(B.shape[1])-(B.T@B.T)))
  else: 
     F = np.kron(B.T,B.T)
     S_flattened = S.flatten(order='F')
     R=A @ G @ A.T
     r = R.flatten(order='F')
     MSE_limit = mu*mu*r.T@np.linalg.inv(np.eye(F.shape[1])-F)@S_flattened


  return MSE_limit,G



def evaluate_MSE_limit_distributed_2(R_X, mu, p_sampling, M, incidence_matrices, cov_matrix_noise, A,normalize = False,S = None, iterations = 1000):
  """
  Evaluates the theorical limit of the MSE of the algorithm

  Parameters:
  - R_X: Covariance matrix of the filtered regressors.
  - mu: A float representing the step size or learning rate of the LMS algorithm.
  - p_sampling: Probability of the diagonal elements of the sampling matrix to be one
  - incidence_matrices: Incidence matrices of the complex.
  - cov_matrix_noise: Covariance matrix of the noise.
  - S: weights of the scalar product of the norm. If None, S is considered to be the Identity matrix
  Return:
  - MSE_limit: theorical limit of the MSE expected value
  """
  n_edges=incidence_matrices[1].shape[1]
  A=np.kron(A,np.eye(2*M+1))
  B=A @ (np.eye((2*M+1)*n_edges)-mu*R_X)
  if type(p_sampling)==float: p_sampling=[p_sampling]*n_edges
  
  L_d = incidence_matrices[1].T@incidence_matrices[1]
  
  if normalize == True: L_d=L_d/norma_spettrale(L_d)
  L_u = incidence_matrices[2]@incidence_matrices[2].T
  
  if normalize == True: L_u=L_u/norma_spettrale(L_u)
  
  G=np.zeros(((2*M+1)*n_edges,(2*M+1)*n_edges))
  for j in range(n_edges):
    G_j=np.zeros((2*M+1,(2*M+1)))
    G_j[0][0]=p_sampling[j]*cov_matrix_noise[j][j]

    for k in range (1,M+1):
          Ld_power = np.linalg.matrix_power(L_d, k)
          Lu_power = np.linalg.matrix_power(L_u, k)
          
          G_j[k][k]=p_sampling[j]*cov_matrix_noise[j,j] *Ld_power[j,:].T @ Ld_power[j,:]
          G_j[k+M][k+M]=p_sampling[j]*cov_matrix_noise[j,j] *Lu_power[j,:].T @ Lu_power[j,:]
    G[j*(2*M+1):(j+1)*(2*M+1),j*(2*M+1):(j+1)*(2*M+1)] = G_j


  if S is None: MSE_limit = mu*mu*np.trace(A @ G @ A.T @ np.linalg.inv(np.eye(B.shape[1])-(B.T@B)))
  else: 
     H = np.eye(B.shape[0])
     B_power = B
     B_T_power = B.T
     for it in range(iterations):
        H = H + B_T_power @ S @ B_power
        B_power = B_power @ B
        B_T_power = B_T_power @ B.T
     MSE_limit = mu*mu*np.trace(A @ G @ A.T @ H)


  return MSE_limit,G