
# coding: utf-8

# In[1]:


import os
import tarfile
from six.moves import urllib


# In[3]:


DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"


# In[5]:


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


# In[6]:


fetch_housing_data()


# In[7]:


import pandas as pd


# In[8]:


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


# ## Take a Quick Look at the Data Structure

# In[9]:


housing = load_housing_data()
housing.head()


# In[10]:


housing.info()


# There are 20,640 instances in the dataset, which means that it is fairly small by Machine Learning standards, but it’s perfect to get started. Notice that the total_bedrooms attribute has only 20,433 non-null values, meaning that 207 districts are missing this feature. We will need to take care of this later.

# In[11]:


housing["ocean_proximity"].value_counts()


# In[12]:


housing.describe()


# In[15]:


# only in a Jupyter notebook
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[16]:


housing.hist(bins=50, figsize=(20, 15))
plt.show()


# ## 3. Create a Test Set

# In[18]:


import numpy as np


# In[19]:


def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


# In[20]:


train_set, test_set = split_train_test(housing, 0.2)


# In[21]:


print(len(train_set), "train + ", len(test_set), "test")


# In[23]:


import hashlib


# In[29]:


def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] <256 * test_ratio


# In[30]:


def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]


# In[31]:


housing_with_id = housing.reset_index()


# In[32]:


housing_with_id.head()


# In[33]:


train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")


# In[34]:


train_set.head()


# In[35]:


test_set.head()


# In[42]:


housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")


# In[43]:


from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)


# In[44]:


housing['median_income'].hist(bins=50, figsize=(20, 15))
plt.show()


# In[52]:


housing['income_cat'] = np.ceil(housing['median_income']/1.5)


# In[53]:


housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)


# In[54]:


housing['income_cat'].unique()


# In[55]:


from sklearn.model_selection import StratifiedShuffleSplit


# In[56]:


split = StratifiedShuffleSplit(n_splits = 1, test_size=0.2, random_state=42)


# In[57]:


for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# In[59]:


housing['income_cat'].value_counts() / len(housing)


# In[60]:


for set in (strat_train_set, strat_test_set):
    set.drop(['income_cat'], axis=1, inplace=True)


# ## 4. Discover and Visualize the Data to Gain Insights

# In[61]:


housing = strat_train_set.copy()


# ### 4.1 Visualizing Geographical Data

# In[63]:


housing.plot(kind='scatter', x='longitude', y='latitude', alpha = 0.1 )


# The radius of each circle represents the district’s population (option s), and the color represents the price (option c). We will use a predefined color map (option cmap) called jet, which ranges from blue (low values) to red (high prices):

# In[66]:


housing.plot(kind='scatter', 
             x='longitude', 
             y='latitude', 
             alpha = 0.4,
             s=housing['population']/100,
             label='population',
             c='median_house_value',
             cmap=plt.get_cmap('jet'),
             colorbar=True)
plt.legend()


# ## 4. Looking for Correlations

# In[67]:


corr_matrix = housing.corr()


# In[68]:


corr_matrix['median_house_value'].sort_values(ascending=False)


# In[70]:


from pandas.tools.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))


# In[71]:


housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)


# ### 4.3 Experimenting with Attribute Combinations

# In[72]:


housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]


# In[73]:


corr_matrix = housing.corr()


# In[76]:


corr_matrix['median_house_value'].sort_values(ascending=False)


# ## 4. Prepare the Data for Machine Learning Algorithms

# In[77]:


housing = strat_train_set.drop("median_house_value", axis=1)


# In[78]:


housing_labels = strat_train_set["median_house_value"].copy()


# ### 4.1 Data Cleaning

# In[79]:


from sklearn.preprocessing import Imputer

imputer = Imputer(strategy="median")


# In[80]:


housing_num = housing.drop('ocean_proximity', axis = 1)


# In[82]:


imputer.fit(housing_num)


# In[83]:


imputer.statistics_


# In[84]:


imputer.strategy


# In[86]:


housing_num.median().values


# In[87]:


X = imputer.transform(housing_num)


# ### 4.2 Handling Text and Categorical Attributes

# In[88]:


from sklearn.preprocessing import LabelEncoder


# In[89]:


encoder = LabelEncoder()


# In[90]:


housing_cat = housing['ocean_proximity']


# In[91]:


housing_cat_encoded = encoder.fit_transform(housing_cat)


# In[92]:


housing_cat_encoded


# In[93]:


print(encoder.classes_)


# In[94]:


from sklearn.preprocessing import OneHotEncoder


# In[95]:


encoder = OneHotEncoder()


# In[96]:


housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))


# In[98]:


housing_cat_1hot


# In[99]:


housing_cat_1hot.toarray()


# In[100]:


from sklearn.preprocessing import LabelBinarizer


# In[105]:


encoder = LabelBinarizer(sparse_output=True)


# In[106]:


housing_cat_1hot = encoder.fit_transform(housing_cat)


# In[107]:


housing_cat_1hot

