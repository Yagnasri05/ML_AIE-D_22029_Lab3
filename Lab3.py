import numpy as np
import pandas as pd
from numpy.linalg import matrix_rank
import statistics
import matplotlib.pyplot as plt

def customer_data(df):
    # A1 
    # Segmentation of data into different matrices
    A = df.iloc[:, [1, 2, 3]]
    C = df.iloc[:, [4]]

    # Dimensions of the matrices
    dimA = np.shape(A)
    dimC = np.shape(C)

    # Rank of matrix
    rA = matrix_rank(A)
    
    # A2 Use the Pseudo-inverse to calculate the model vector X for predicting the cost of the products available with the vendor.
    # Pseudo inverse of matrix A
    pA = np.linalg.pinv(A)
    
    # Dimension of the pseudo matrix
    dimPA = np.shape(pA)

    # Finding the vector X
    x = np.dot(pA, C)

    return A, C, dimA, dimC, rA, pA, dimPA, x

def classify_customers(df):
    # A3 Mark all customers (in “Purchase Data” table) with payments above Rs. 200 as RICH and others as POOR. Develop a classifier model to categorize customers into RICH or POOR class based on purchase behavior.
    df['Rich/Poor'] = ['Rich' if x > 200 else 'Poor' for x in df['Payment (Rs)']]
    return df

def irctc_stock_data(df1):
    # A4
    mean = statistics.mean(df1['Price'])
    var = statistics.variance(df1['Price'])

    wednesday = df1[df1['Day'] == 'Wednesday']['Price']
    wd = pd.to_numeric(wednesday)
    if len(wd)>0:
        wed_mean = statistics.mean(wd)
        wed_prob_profit = len(wednesday[wednesday > 0]) / len(wednesday)
        wed_cond_prob = len(wednesday[wednesday > 0]) / len(df1[df1['Day'] == 'Wednesday'])
    else:
        wed_mean=None
        wed_prob_profit = None
        wed_cond_prob = None

    april = df1[df1['Month'] == 'Apr']['Price']
    ap = pd.to_numeric(april)
    if len(wd)>0:
        apr_mean = statistics.mean(ap)
    else:
        apr_mean=None

    loss = len(df1[df1['Chg%'] < 0]) / len(df1)

    plt.scatter(df1['Day'], df1['Chg%'])
    plt.xlabel('Day of the Week')
    plt.ylabel('Chg%')
    plt.title('Chg% vs Day of the Week')
    plt.show()

    return mean, var, wed_mean, apr_mean, loss, wed_prob_profit, wed_cond_prob

# Reading the excel file
df = pd.read_excel(r"C:\Users\madda\Downloads\LabSession1Data.xlsx", sheet_name='Purchase data')
A, C, dimA, dimC, rA, pA, dimPA, x = customer_data(df)
classifier = classify_customers(df)

# A4
df1 = pd.read_excel(r"C:\Users\madda\Downloads\LabSession1Data.xlsx", sheet_name='IRCTC Stock Price')
mean, var, wed_mean, apr_mean, loss, wed_prob_profit, wed_cond_prob = irctc_stock_data(df1)

# Print Statements
print("Matrix A is:")
print(A)
print("Matrix C is:")
print(C)
print("Dimension of the vector space A:", dimA)
print("Dimension of vector C:", dimC)
print("The rank of matrix A is:", rA)

#A2 print statemnts
print("The pseudo inverse of A is:")
print(pA)
print("Dimension of the pseudo-inverse matrix is:", dimPA)
print("Vector X:")
print(x)

#A3 print statements
print("The classification of customers based on their payments are:")
print(classifier)

#A4 print statements
print("Mean of Price data:", mean)
print("Variance of Price data:", var)
print("Sample mean of Wednesday prices:", wed_mean)
print("Sample mean of April prices:", apr_mean)
print("Probability of making a loss over the stock:", loss)
print("Probability of making a profit on Wednesday:", wed_prob_profit)
print("Conditional probability of making profit given that today is Wednesday:", wed_cond_prob)
