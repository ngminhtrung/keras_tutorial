
# coding: utf-8

# ### The whimsical dataset

# In[3]:


import os
import pandas as pd
import warnings
warnings.simplefilter("ignore")


# Source of data:
# 
# In the interest open data science, the collection of answers given by attendees is available under a CC-BY-SA 4.0 license. Please credit it as "© Anaconda Inc. 2018" if you use the anonymized data, [available as a CSV file](https://goo.gl/WgTQMX). If you wish to code along with the rest of this notebook, save it locally as Learning about Humans learning ML.csv (or adjust your code as needed).
# 

# In[5]:


file_path = "D:\Learning\AI - Machine Learning - Deep Learning\Dataset\Learning about Humans learning ML.csv"


# In[6]:


humans = pd.read_csv(os.path.join(file_path))


# In[7]:


humans.head()


# In[8]:


humans.drop("Timestamp", axis=1, inplace=True)


# In[9]:


humans["Education"] = humans["Years of post-secondary education (e.g. BA=4; Ph.D.=10)"].str.replace(r'.*=','').astype(int)


# In[10]:


humans['Education'].head()


# In[11]:


humans.drop('Years of post-secondary education (e.g. BA=4; Ph.D.=10)', axis=1, inplace=True)


# In[12]:


humans.head()


# ### Eyeballing data

# In[13]:


humans.describe()


# In[15]:


humans.describe(include=['object', 'int', 'float'])


# ### Data cleanup

# In[16]:


human_dummies = pd.get_dummies(humans)
list(human_dummies.columns)


# ### Classification: Choosing features and a target

# In[17]:


X = human_dummies.drop("How successful has this tutorial been so far?", axis=1)


# In[18]:


y = human_dummies["How successful has this tutorial been so far?"] >= 8


# ### Conventional names and shapes

# In[19]:


y.head()


# In[20]:


X.iloc[:5, :5]


# ### Train/ test split

# In[21]:


from sklearn.model_selection import train_test_split


# In[22]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


# In[23]:


print("Training features/ target: ", X_train.shape, y_train.shape)


# In[24]:


print("Testing features/ target: ", X_test.shape, y_test.shape)


# ### Choosing an algorith: Decision Trees and Random Forests

# In[25]:


from sklearn.ensemble import RandomForestClassifier


# In[26]:


rf = RandomForestClassifier(n_estimators=10, random_state=0)


# In[27]:


rf.fit(X_train, y_train)


# In[28]:


rf.score(X_test, y_test)


# In[29]:


from sklearn.tree import DecisionTreeClassifier


# In[30]:


tree = DecisionTreeClassifier(max_depth=7, random_state=0)


# In[31]:


tree.fit(X_train, y_train)


# In[32]:


tree.score(X_test, y_test)


# ### Feature importances

# In[33]:


tree = DecisionTreeClassifier(max_depth=7, random_state=0)


# In[34]:


tree.fit(X, y)


# In[35]:


tree.score(X, y)


# In[36]:


get_ipython().run_line_magic('matplotlib', 'inline')

pd.Series(tree.feature_importances_, index=X.columns).plot.barh(figsize=(18,7))


# ### Cut points in a Decision Tree

# In[39]:


from sklearn.tree import export_graphviz
import sys, subprocess
from IPython.display import Image

export_graphviz(tree, feature_names=X.columns, class_names=['failure','success'],
                out_file='ml-good.dot', impurity=False, filled=True)
subprocess.check_call([sys.prefix+'/bin/dot','-Tpng','ml-good.dot',
                       '-o','ml-good.png'])
Image('ml-good.png')


# ### Quick comparision of many classifiers in scikit-learn

# In[40]:


from sklearn.neural_network import MLPClassifier


# In[42]:


from sklearn.neighbors import KNeighborsClassifier


# In[43]:


from sklearn.svm import SVC


# In[44]:


from sklearn.gaussian_process import GaussianProcessClassifier


# In[45]:


from sklearn.tree import DecisionTreeClassifier


# In[47]:


from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


# In[48]:


from sklearn.naive_bayes import GaussianNB


# In[50]:


from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


# In[52]:


from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

classifiers = {
    "KNN(3)"       : KNeighborsClassifier(3), 
    "Linear SVM"   : SVC(kernel="linear"), 
    "RBF SVM"      : SVC(gamma=2, C=1), 
    "Gaussian Proc": GaussianProcessClassifier(1.0 * RBF(1.0)),
    "Decision Tree": DecisionTreeClassifier(max_depth=7), 
    "Random Forest": RandomForestClassifier(max_depth=7, n_estimators=10, max_features=4), 
    "Neural Net"   : MLPClassifier(alpha=1), 
    "AdaBoost"     : AdaBoostClassifier(),
    "Naive Bayes"  : GaussianNB(), 
    "QDA"          : QuadraticDiscriminantAnalysis()
}


# In[53]:


for name, classifier in classifiers.items():
    classifier.fit(X_train, y_train)
    score = classifier.score(X_test, y_test)
    print("{:<15}| score = {:.3f}".format(name, score))


# ### Regression

# In[54]:


from sklearn import datasets

[attr for attr in dir(datasets) if not attr.startswith('_')]


# ### Boston Housing Prices

# In[55]:


boston = datasets.load_boston()
print(boston.DESCR)


# ### Working with the example datasets

# In[57]:


# Sample datasets can be easily converted to Pandas/ Dask/ PySpark DataFrames if desired


# In[58]:


boston_df = pd.DataFrame(boston.data, columns=boston.feature_names)


# In[59]:


boston_df.head()


# In[60]:


# The x and y are stored in standard attributes
X = boston.data
y = boston.target


# ### Comparing a gaggle of regressors

# In[61]:


from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


# In[62]:


from sklearn.model_selection import cross_val_predict


# In[63]:


from sklearn.neighbors import KNeighborsRegressor


# In[64]:


from sklearn.linear_model import LinearRegression, RANSACRegressor


# In[65]:


from sklearn.gaussian_process import GaussianProcessRegressor


# In[66]:


from sklearn.svm import SVR
from sklearn.svm import LinearSVR


# In[67]:


regressors =[
    LinearRegression(),
    RANSACRegressor(),
    GaussianProcessRegressor(),
    KNeighborsRegressor(n_neighbors=9, metric='manhattan'),
    SVR(),
    LinearSVR(),
    SVR(kernel='linear')
]


# In[69]:


for model in regressors:
    predictions = cross_val_predict(model, X, y, cv=10)
    print(model)
    print("\tExplained variance: ", explained_variance_score(y, predictions))
    print("\tMean absolute error: ", mean_absolute_error(y, predictions))
    print("\tR2 score: ", r2_score(y, predictions))


# ### Hyperparameters

# ### Grid Search

# In[70]:


from sklearn.model_selection import GridSearchCV

knr = KNeighborsRegressor()

parameters = {'n_neighbors': [5,6,7,8,9,10,11,12],
              'weights': ['uniform', 'distance'],
              'metric': ['minkowski', 'chebyshev', 'manhattan']
             }


# In[71]:


grid = GridSearchCV(knr, parameters)


# In[72]:


model = grid.fit(X, y)


# In[73]:


print(model)


# In[74]:


predictions = cross_val_predict(model, X, y, cv=10)


# In[75]:


print("Explained variance: ", explained_variance_score(y, predictions))


# In[76]:


print("R2 score: ", r2_score(y, predictions))


# ###  Clustering

# In[77]:


from sklearn import cluster

spectral = cluster.SpectralClustering(n_clusters = 4, eigen_solver = 'arpack', affinity = 'nearest_neighbors')


# In[78]:


spectral.fit(boston.data)


# In[79]:


boston_df['category'] = spectral.labels_
boston_df['price'] = boston.target
house_clusters = boston_df.groupby('category').mean().sort_values('price')
house_clusters.index = ['low', 'mid_low', 'mid_high', 'high']
house_clusters[['price', 'CRIM', 'RM', 'AGE', 'DIS']]

