import pandas as pd
import numpy as np

Zoo = pd.read_csv("D:\\ExcelR Data\\Assignments\\KNN\\Zoo.csv")

# to see data columns
Zoo.columns

# Training and Test data using 
from sklearn.model_selection import train_test_split
train,test = train_test_split(Zoo,test_size = 0.2) # 0.2 => 20 percent of entire data 

# KNN using sklearn 
# Importing Knn algorithm from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier as KNC

accuracy = []

# Calculating accuracy for nearest neighbours  between 1 and 40
for i in range(1, 30):
    knn = KNC(n_neighbors=i)
    # Fitting with training data 
    knn.fit(train.iloc[:,1:17],train.iloc[:,17])
    train_acc = np.mean(knn.predict(train.iloc[:,1:17])==train.iloc[:,17])#72%
    test_acc =np.mean(knn.predict(test.iloc[:,1:17])==test.iloc[:,17])#61%
    
# Calculating accuracy for nearest neighbours  between 1 and 10
for i in range(1, 10):
    knn = KNC(n_neighbors=i)
    # Fitting with training data 
    knn.fit(train.iloc[:,1:17],train.iloc[:,17])
    train_acc = np.mean(knn.predict(train.iloc[:,1:17])==train.iloc[:,17])#82%
    test_acc =np.mean(knn.predict(test.iloc[:,1:17])==test.iloc[:,17])#71%
    
