import pandas as pd
import numpy as np
glass = pd.read_csv("D:\\ExcelR Data\\Assignments\\KNN\\glass.csv")

# Training and Test data using 
from sklearn.model_selection import train_test_split
train,test = train_test_split(glass,test_size = 0.2) # 0.2 => 20 percent of entire data 

# KNN using sklearn 
# Importing Knn algorithm from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier as KNC

accuracy = []

# Calculating accuracy for nearest neighbours  between 1 and 40
for i in range(1, 40):  # for i in range(1,40,2): it will take only odd Numbers between 1-40
    knn = KNC(n_neighbors=i)
    # Fitting with training data 
    knn.fit(train.iloc[:,0:9],train.iloc[:,9])
    train_acc = np.mean(knn.predict(train.iloc[:,0:9])==train.iloc[:,9]) # 59%
    test_acc =np.mean(knn.predict(test.iloc[:,0:9])==test.iloc[:,9])# 69%

# Calculating accuracy for nearest neighbours  between 1 and 10

for i in range(1, 10):    # for i in range(1,10,2): it will take only odd Numbers between 1-10
    knn = KNC(n_neighbors=i)
    # Fitting with training data 
    knn.fit(train.iloc[:,0:9],train.iloc[:,9])
    train_acc = np.mean(knn.predict(train.iloc[:,0:9])==train.iloc[:,9])# 66%
    test_acc =np.mean(knn.predict(test.iloc[:,0:9])==test.iloc[:,9])# 72%