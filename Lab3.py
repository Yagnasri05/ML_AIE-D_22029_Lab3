import numpy as np
import pandas as pd
from numpy.linalg import matrix_rank
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def customer_data(df):
    #A1
    A=df.iloc[:,[1,2,3]]
    print("Matrix A is:")
    print(A)
    C=df.iloc[:,[4]]
    print("Matrix C is:")
    print(C)

    dimA=np.shape(A)
    dimC=np.shape(C)
    print("Dimension of the vector space A :",dimA)
    print("Dimension of vector C:",dimC)

    rA=matrix_rank(A)
    print("The rank of matrix A is :",rA)

    pA=np.linalg.pinv(A)
    print("The psuedo inverse of A is :")
    print(pA)

    dimPA=np.shape(pA)
    print("Dimension of the pseudo inverse matrix is :",dimPA)
     
     # A2
    x=np.dot(pA,C)
    print(" vector X :")
    print(x)
    c=np.dot(pA,x)
    for i in range(0,len(c)):
        b=""
        if c>200:
            b+="Rich"
        else:
            b+="Poor"
        
#A3 
def classify_customers(df):
    features = ["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]
    X = df[features]
    y = df['Category']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    classifier = DecisionTreeClassifier(random_state=42)
    classifier.fit(X_train, y_train)
    df['Predicted Category'] = classifier.predict(X)
    return df

df = pd.read_excel(r"C:\Users\madda\Downloads\LabSession1Data.xlsx",sheet_name='Purchase data')
print(df)
customer_data(df)
classify_customers(df)
