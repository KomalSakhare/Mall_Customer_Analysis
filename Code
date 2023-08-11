# Imported the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# to ignore all warning messages that may arise during the execution
import warnings
warnings.filterwarnings("ignore")


# To load the dataset
df = pd.read_csv("Mall_Customers.csv")

# Displays the first 5 rows
# Columns in the dataset are CustomerID, Gender, Age, Annual Income and Spending Score
df.head()


# Displays randomly selected number of rows from the dataset
df.sample(7)


# Specifies the shape (number of rows and columns), In this dataset there are 200 rows and 5 columns
df.shape


# Provides the statistical description of only numeric column in the dataset
df.describe()


# Provides information such as data-type of the column and check for null values 
df.info()


# It provides the unique number of data points present in each column
df.nunique()


# Another way to check the total number of null values in the dataset
df.isnull().sum()


# The heatmap provides the correlation between the columns
# Each column is highly correlated with itself
sns.heatmap(df.corr(),annot=True)


# Droped the CustomerID column as it does not add value to the dataset and is not relevant to the context
df.drop(["CustomerID"],axis=1,inplace=True)


# Representation of pairplot , which helps to identify the pattenrs and how the columns are related with each other(linear or non - linear)
sns.pairplot(df)


# Distribution of age
# There are customers of a wide variety of ages.
sns.distplot(df["Age"])


# Distribution of Annnual Income
# Most of the annual income falls between 50K to 85K.
sns.distplot(df["Annual Income (k$)"])


# Spending Score Distribution
# The maximum spending score is in the range of 40 to 60.
sns.distplot(df["Spending Score (1-100)"])


# Countplot, which represents the number of male and female present in the dataset of mall customer
# More female customers than male.
sns.countplot(data=df,x="Gender")
age1 = df.Age[(df.Age <= 25) & (df.Age >= 18)]
age2 = df.Age[(df.Age <= 35) & (df.Age >= 26)]
age3 = df.Age[(df.Age <= 45) & (df.Age >= 36)]
age4 = df.Age[(df.Age <= 55) & (df.Age >= 46)]
age5 = df.Age[df.Age >= 56]

x = ["18-25","26-35","36-45","46-55","55+"]
y = [len(age1.values),len(age2.values),len(age3.values),len(age4.values),len(age5.values)]

plt.figure(figsize=(15,6))
sns.barplot(x=x, y=y ,palette="rocket")
plt.title("Number of Customer and Ages")
plt.xlabel("Age")
plt.ylabel("Number of Customer")
plt.show()


# Scatter plot to show Spending Score Vs Annual Income
plt.figure(figsize=(10,6))
sns.scatterplot(x = 'Annual Income (k$)',y = 'Spending Score (1-100)',  data = df  ,s = 70 )
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)') 
plt.title('Spending Score (1-100) vs Annual Income (k$)')
plt.show()


# Imported the important library for k-means clustering
from sklearn.cluster import KMeans


# Here, calculated the Within Cluster Sum of Squared Errors (WSS) for different values of k
# This is used to identify the value of k
# This value of K gives us the best number of clusters to make
# The below graph is known as elbow graph
# In the graph, after 5 the drop is minimal, so we take 5 to be the number of clusters.
wcss = []
for k in range(1,11):
    kmeans = KMeans(n_clusters=k, init="k-means++")
    kmeans.fit(df.iloc[:,1:])
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(12,6))    
plt.grid()
plt.plot(range(1,11),wcss, linewidth=2, color="red", marker ="8")
plt.xlabel("K Value")
plt.xticks(np.arange(1,11,1))
plt.ylabel("WCSS")
plt.show()


# Here I have provided the number of clusters to be formed derived from the previous step
# and fitted the data for training
km = KMeans(n_clusters=5)
clusters = km.fit_predict(df.iloc[:, 1:])
df["label"] = clusters


# 5 different clusters have been formed based on the value of k
# The red cluster is the customers with the least income and least spending score.
# The blue cluster is the customers with the most income and most spending score.
plt.figure(figsize=(10,6))
sns.scatterplot(x = 'Annual Income (k$)',y = 'Spending Score (1-100)',hue="label",  
                 palette=['green','orange','brown','dodgerblue','red'], legend='full',data = df  ,s = 60 )
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)') 
plt.title('Spending Score (1-100) vs Annual Income (k$)')
plt.show()


# 5 different clusters have been formed based on the value of k
# People having age between 20 - 40 have a high spending score
plt.figure(figsize=(10,6))
sns.scatterplot(x = 'Age',y = 'Spending Score (1-100)',hue="label",  
                 palette=['green','orange','brown','dodgerblue','red'], legend='full',data = df  ,s = 60 )
plt.xlabel('Age')
plt.ylabel('Spending Score (1-100)') 
plt.title('Spending Score (1-100) vs Age')
plt.show()


# worked on 3 types of data. 
# Apart from the spending score and annual income of customers, I have also take in the age of the customers.

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111, projection='3d')
cluster0 = ax.scatter(df.Age[df.label == 0], df["Annual Income (k$)"][df.label == 0], df["Spending Score (1-100)"][df.label == 0], c='blue', s=60)
cluster1 = ax.scatter(df.Age[df.label == 1], df["Annual Income (k$)"][df.label == 1], df["Spending Score (1-100)"][df.label == 1], c='red', s=60)
cluster2 = ax.scatter(df.Age[df.label == 2], df["Annual Income (k$)"][df.label == 2], df["Spending Score (1-100)"][df.label == 2], c='green', s=60)
cluster3 = ax.scatter(df.Age[df.label == 3], df["Annual Income (k$)"][df.label == 3], df["Spending Score (1-100)"][df.label == 3], c='orange', s=60)
cluster4 = ax.scatter(df.Age[df.label == 4], df["Annual Income (k$)"][df.label == 4], df["Spending Score (1-100)"][df.label == 4], c='purple', s=60)
ax.view_init(30, 185)
plt.xlabel("Age")
plt.ylabel("Annual Income (k$)")
ax.set_zlabel('Spending Score (1-100)')
ax.legend([cluster0, cluster1, cluster2, cluster3, cluster4], ['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4'])
plt.show()


# Imported the library named pickle to pickle the clustering module
import pickle


# Now after the training of K-means model, I have pickled it and save it to a file named clustering_file for later use.
pickle.dump(km, open('clustering_file', 'wb'))


# To use the K-means model in the future without retraining, simply unpickle it from the file.
file = pickle.load(open('clustering_file', 'rb'))


new = [
    [25, 40, 80],   # Sample data point 1 with age, annual income, and spending score
    [45, 60, 20],   # Sample data point 2 with age, annual income, and spending score
    [30, 70, 50],   # Sample data point 3 with age, annual income, and spending score
]


#Predicting the new data points 
new_data= file.predict(new)

print("Cluster for new data points:")
print(new_data)
