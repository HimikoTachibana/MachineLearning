"""Basic code for assignment 1."""

import numpy as np
import pandas as pd
import scipy.stats as stats

np.set_printoptions(suppress=True)
def load_unicef_data():
    """Loads Unicef data from CSV file.

    Retrieves a matrix of all rows and columns from Unicef child mortality
    dataset.

    Args:
      none

    Returns:
      Country names, feature names, and matrix of values as a tuple (countries, features, values).

      countries: vector of N country names
      features: vector of F feature names
      values: matrix N-by-F
    """
    fname = 'SOWC_combined_simple.csv'

    # Uses pandas to help with string-NaN-numeric data.
    data = pd.read_csv(fname, na_values='_')
    # Strip countries title from feature names.
    features = data.axes[1][1:]
    # Separate country names from feature values.
    countries = data.values[:,0]
    values = data.values[:,1:]
    # Convert to numpy matrix for real.
    values = np.asmatrix(values,dtype='float64')

    # Modify NaN values (missing values).
    mean_vals = np.nanmean(values, axis=0)
    inds = np.where(np.isnan(values))
    values[inds] = np.take(mean_vals, inds[1])
    return (countries, features, values)



def normalize_data(x):
    """Normalize each column of x to have mean 0 and variance 1.
    Note that a better way to normalize the data is to whiten the data (decorrelate dimensions).  This can be done using PCA.

    Args:
      input matrix of data to be normalized

    Returns:
      normalized version of input matrix with each column with 0 mean and unit variance

    """
    mvec = x.mean(0)
    stdvec = x.std(axis=0)
    
    return (x - mvec)/stdvec
    


def linear_regression(x, t, basis, reg_lambda=0, degree=0):
    """Perform linear regression on a training set with specified regularizer lambda and basis

    Args:
      x is training inputs
      t is training targets
      reg_lambda is lambda to use for regularization tradeoff hyperparameter
      basis is string, name of basis to use
      degree is degree of polynomial to use (only for polynomial basis)

    Returns:
      w vector of learned coefficients
      train_err RMS error on training set
      """

    # TO DO:: Complete the design_matrix function.
    # e.g. phi = design_matrix(x,basis, degree)
    
    phi=design_matrix(x,basis,degree)
    phi_cross=np.linalg.pinv(phi)
    #t_tranpose=t.T
    # TO DO:: Compute coefficients using phi matrix
    if(reg_lambda==0):
        w=phi_cross.dot(t)
    if(reg_lambda!=0):
       # print("Inside lambda if: ")
        n_col=phi.shape[1]
        #r=phi.T.dot(phi) + reg_lambda * np.identity(n_col)
        r=reg_lambda*np.identity(n_col)+phi.T.dot(phi)
        r=np.linalg.inv(r)
	#r=np.linalg.inv(r)
        z=r@phi.T
        w=z@t
    #w = phi_cross.dot(t)

    # Measure root mean squared error on training data.
    # Basic algorithim goes as follows:
    # 	1. We take Equation 3.12  * 1/n 
    #      Then we math.sqrt( of the equation obtained in 1.)

    # t_est: variable for estimation of targets
    t_est= phi.dot(w)
    
    # variable to calculate the difference between our target and estimate
    # target is the left operand, estimate is right operand
    diff=t-t_est
    
    # Square all the elements
    diff_squared=np.power(diff,2)

    # Sum up all the elements of diff_squared, i.e take square of
    # all elements then sum them up

    sig_squared=diff_squared.sum()

    # multiply by 1/2 as specified in PRML

    half_sig_squared=0.5*(sig_squared)

    # Divide by population size and square root
    population_size= t.shape[0]

    rmse_bforesqrt=half_sig_squared/population_size

    train_err = np.sqrt(rmse_bforesqrt)

    return (w, train_err)

def format_list(format_specifier,amount):
   result=[]
   for i in range(amount):
     result.append(format_specifier)
   return result

## origColAmount parameter is the amount of columns in our original
## input matrix i.e features with examples
def format_string_phi(format_list,origColAmount):
    sep=""
    restring=""
    for i in range(len(format_list)):
      if(i%origColAmount==0):
        sep="|"
      if((i)%origColAmount != 0):
        sep="..."
      restring+=format_list[i]+sep
    return restring

def print_design_matrix(design_matrix,format_string_phi):
    for row in design_matrix:
      x=row.tolist()
      to_tuple=tuple(x[0])
      print((format_string_phi)%(to_tuple))

def design_matrix(x, basis, degree=0):
    """ Compute a design matrix Phi from given input datapoints and basis.
	Args:
      x matrix of input datapoints
      basis string name of basis

    Returns:
      phi design matrix
    """
    # TO DO:: Compute desing matrix for each of the basis functions
    if basis == 'polynomial':
        result=None
        for i in range(1,degree+1):
          newMatrix=np.power(x,i)
          if result is None:
            result=newMatrix
          else:
            result=np.hstack((result,newMatrix))
        #initialize a column of ones to concat to final result
        res_rows=result.shape[0]
        ones_col=np.ones((res_rows,1))
        phi=np.hstack((ones_col,result))
        #phi=result[...,2:]
    elif basis == 'ReLU':
        result=None
        newMatrix=np.negative(x)
        newMatrix=np.add(newMatrix,5000)

        reLUtrix=np.maximum(newMatrix,0,newMatrix)
        if result is None:
            result=reLUtrix
        else:
            result=np.hstack((result,reLUtrix))
        res_rows=result.shape[0]
        ones_col=np.ones((res_rows,1))
        phi = np.hstack((ones_col,result))
        # Debug statement feel free to comment out
        #print("Value of phi",phi)
    else:
        assert(False), 'Unknown basis %s' % basis

    return phi


def evaluate_regression(x, t, w, basis, degree):
    """Evaluate linear regression on a dataset.
	Args:
      x is evaluation (e.g. test) inputs
      w vector of learned coefficients
      basis is string, name of basis to use
      degree is degree of polynomial to use (only for polynomial basis)
      t is evaluation (e.g. test) targets

    Returns:
      t_est values of regression on inputs
      err RMS error on the input dataset 
      """
  	# TO DO:: Compute t_est and err 
    #w_tranpose=w.T


    # My logic goes as follows:
    # Definition of test error is when you run the trained
    # model against a dataset that it hasn't been exposed to
    # this dataset is known as the testset 

    # As such the basic algorithm goes as follows:
    # We do not need to recompute the weights but we need to recompute
    # phi for our test data

    # As such, we are interested in how well our trained weights
    # estimate against the test data so we matrix multiply our
    # weights against the phi from our test data
    # thus t_est = w_train.T*phi(x) since we want to know how well our
    # trained model estimates against the training data
    # but in implementation we do phi(x)*w_train
    # to match array dimensions 


    #Compute design matrix from test data 
    phi=design_matrix(x,basis,degree)
    phi_cross=np.linalg.pinv(phi)

    # Compute testing weights // just in case we require this variable
    #if(t is not None):
        #w_test=phi_cross.dot(t)
    #w_test=phi_cross.dot(t)

    # We want to be able to index into our target vector

    #t_est=phi.dot(w_test)
    #if (t is not None):
       # testing_estimate=phi.dot(w_test)
    #testing_estimate=phi.dot(w_test)

     # Estimate of our targets according to test data against learned 
     # coefficients
    t_est=phi.dot(w)
    #print("t_est",t_est)
    #t_est = None

    # We calculate the RMS error as follows
    # Take equation 3.12 of PRML and modify as follows
    # My logic:
    #    The equation given in PRML gives the SSE (sum of squares error)
    #    By definition the  MSE (mean squared error) takes the SSE and divides 
    #    it by population size, we also preserve the 1/2 constant 
    #    throughout our calcuations 
    #    Afterwards we take our MSE and square root it.

    # Compute difference between target and estimate

    if(t is not None):
        
        diff=t-t_est
        # Square all observations
        diff_squared=np.power(diff,2)
        # Sum up all the observations in our vector
        sig_squared=diff_squared.sum()
        half_sig_squared=0.5*(sig_squared)
        # Calculate population size
        population_size=t.shape[0]
        rmse=np.sqrt(half_sig_squared/population_size)
        err=rmse
    else:
        err=None

    #diff=t-t_est


    # Square all observations 
    #diff_squared=np.power(diff,2)

    # Sum up all the observations in our vector
    #sig_squared=diff_squared.sum()

    #half_sig_squared=0.5*(sig_squared)

    # Calculate population size
    #population_size=t.shape[0]

    #rmse=np.sqrt(half_sig_squared/population_size)
    #err = rmse
    #print("err inside function",err)
    #err=rmse
    return (t_est, err)

"""
countries,features,values=load_unicef_data()

print("values: ",values)
print("Type of values: ",type(values))
values_cols=values.shape[1]
degree=4
format_specifierG="%8.2f"
design_matrix=design_matrix(values,"polynomial",degree)
design_cols=design_matrix.shape[1]

flist=format_list(format_specifierG,design_cols)
format_string_phi=format_string_phi(flist,values_cols)
#print_design_matrix(design_matrix,format_string_phi)

#column 2 is our target value
columnZwei=values[:,0]

print("Column 2 of our data is: ",columnZwei)
"""

## before abstracting away for design matrix

#degree=3

# we do 1 to degree+1 to compensate for 0 indexing
#result=None
#for i in range(1,degree+1):
   #print("i: ",i)
   #result = 
   #newMatrix=np.power(values,i)
   #if result is None:
     #result=newMatrix
   #else:
     #result=np.hstack((result,newMatrix))

#print("value of result is: ",result)

"""
(countries,features,values)=load_unicef_data()


#print(values[:,7:])
x=values[:,7:]
#print("value of x: ",x)

GNI=x[:,3]

print("Value of GNI is: ",GNI)
lifeExpectancy=x[:,4]

print("Life expectancy is: ",lifeExpectancy)

literacy=x[:,5]
print("Literacy is: ",literacy)
"""
