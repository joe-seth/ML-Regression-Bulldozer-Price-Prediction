#!/usr/bin/env python
# coding: utf-8

# ## End to End Machine Learning Regression - Bulldozer Price Prediction
# ![bulldozer](https://upload.wikimedia.org/wikipedia/commons/6/6d/CatD9T.jpg)

# # What I am going to cover
# In this notebook, I am going to predict the sale prices of bulldozers using Machine Learning models
# 
# ## 1. Problem Definition
# > How well can I predict the future sale price of a bulldozer given its characterristics and previous examples of home much similar bulldozers have been sold for?
# 
# ## 2. Data
# The data is sourced from https://www.kaggle.com/competitions/bluebook-for-bulldozers/data
# **About the data**
# The data for this competition is split into three parts:
# 
#     Train.csv is the training set, which contains data through the end of 2011.
#     Valid.csv is the validation set, which contains data from January 1, 2012 - April 30, 2012 You make predictions on this set throughout the majority of the competition. Your score on this set is used to create the public leaderboard.
#     Test.csv is the test set, which won't be released until the last week of the competition. It contains data from May 1, 2012 - November 2012. Your score on the test set determines your final rank for the competition.
# 
# The key fields are in train.csv are:
# 
#     SalesID: the uniue identifier of the sale
#     MachineID: the unique identifier of a machine.  A machine can be sold multiple times
#     saleprice: what the machine sold for at auction (only provided in train.csv)
#     saledate: the date of the sale
# 
# 
# ## 3. Evaluation
# The evaluation metric for this problem is RMSLE
# 
# ## 4. Features
# Data Dictionary.xlsx
# 
# ## 5. Modelling
# The problem will be modelled on atleast 3 machine learning models
# 
# ## 6. Experiments
# I will repeat step 5 while performing hyperparameter tuning

# In[1]:


# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly as px


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:


# Import training and validation sets
df = pd.read_csv("/content/TrainAndValid.csv", low_memory=False)
df.columns


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


df.isna().sum()


# ## Make Plots to visualize the saleprice and saledates

# In[7]:


fig, ax = plt.subplots()
ax.scatter(df['saledate'], df['SalePrice']);


# In[8]:


df.SalePrice.plot.hist();


# Since the dates are not in the format that I desire, Parse the dates using the `parse_date` parameter. This parameter reorganizes the dates into year-month-day format. (datetime64)

# In[9]:


# Import data  again but this time parse the dates
df = pd.read_csv('/content/TrainAndValid.csv',
                 low_memory = False,
                 parse_dates =['saledate'])


# In[10]:


df.saledate.dtype


# In[11]:


df.saledate[:1000]


# In[12]:


fig, ax = plt.subplots()
ax.scatter(df['saledate'], df['SalePrice']);


# From the scatter plot, most sales were done during the period of 2005 - 2008. Likewise the period between 2009-2012. Bulldozers worth more than 120000 were purchased in very few numbers.

# ### Sort DataFrame by saledate

# In[13]:


# Sort by  order of dates
df.sort_values(by=['saledate'], inplace=True, ascending=True)
df.saledate.head(20)


# In[14]:


# Make a copy of the original dataframe
df_tmp = df.copy()
df_tmp.saledate.tail()


# ## Feature Engineering

# In[15]:


# Create a funtion that splits saledate
import pandas as pd

def add_date_columns(df, date_column_name='saledate'):
    # Ensure the input DataFrame contains the specified date column
    if date_column_name not in df.columns:
        raise ValueError(f"The DataFrame does not contain a column named '{date_column_name}'")

    # Convert the 'saledate' column to a datetime object
    df[date_column_name] = pd.to_datetime(df[date_column_name])

    # Add 'saleYear', 'saleMonth', and 'saleDay' columns
    df['saleYear'] = df[date_column_name].dt.year
    df['saleMonth'] = df[date_column_name].dt.month
    df['saleDay'] = df[date_column_name].dt.day

    return df

# Usage:
df_tmp = add_date_columns(df_tmp)


# In[16]:


df.columns


# In[17]:


# Remove the saledate column
df_tmp.drop('saledate', axis=1)
df_tmp.tail()


# ## Feature Engineering

# In[18]:


df_tmp['saleYear'] = df_tmp.saledate.dt.year
df_tmp['saleMonth'] = df_tmp.saledate.dt.month
df_tmp['saleDay'] = df_tmp.saledate.dt.day


# In[19]:


# Remove saledate
df_tmp.drop('saledate', axis=1, inplace=True)
df_tmp.tail()


# In[20]:


# Check the values of different columns
df_tmp.state.value_counts()


# ## Creating Visualizations

# In[21]:


df_tmp.corr()


# In[22]:


machine_appendix = pd.read_csv('/content/Machine_Appendix.csv')
machine_appendix.head()


# In[23]:


# Merge Machine appendix to the dataframe
df_merged = df_tmp.merge(machine_appendix, how='left')
df_merged.head()


# In[24]:


df_merged.corr()


# In[25]:


# Using seaborn to make stylish visualizations
import seaborn as sns
import matplotlib.pyplot as plt

# Generate the correlation matrix (assuming df_tmp is your DataFrame)
corr_matrix = df_merged.corr()

# Create a larger figure
plt.figure(figsize=(10, 8))

# Create the heatmap
ax = sns.heatmap(corr_matrix, center=0, annot=True, fmt=".2f", linewidths=0.5, cmap="YlGnBu")

# Show the plot
plt.show()


# In[26]:


df_merged.corr()['SalePrice'].sort_values(ascending=False)


# It appears that the PrimaryLower variable exhibits the most significant negative correlation with SalePrice, while the YearMade column shows the strongest positive correlation with SalePrice. This conclusion aligns with the intuitive expectation that the YearMade column's positive impact on SalePrice is due to newer vehicles having higher auction prices because of their enhanced features and greater compatibility, whereas older vehicles tend to command lower prices due to their limited features and compatibility.
# 
# 

# **Note:** It is possible that categorical data might exhibit a stronger correlation with the target value.

# In[27]:


import plotly.express as px
fig = px.scatter(df_merged,
                 x='PrimaryLower',
                 y='SalePrice',
                 color='ProductGroup',
                 opacity=0.8,
                 hover_data=['MachineID'],
                 title='Primarylower vs SalePrice')
fig.update_traces(marker_size=5)
fig.show()


# A similar observation is evident in the chart, showcasing a discernible inverse relationship between "Primarylower" and "SalePrice."

# In[28]:


import plotly.express as px

fig = px.histogram(df_merged,
                  x='ProductGroup',
                  y="SalePrice",
                  color='ProductGroup',
                  hover_data=['saleYear'],
                  title='ProductGroup vs Count of SalePrice',
                  histfunc='count')  # Use 'count' to display the count of SalePrice

fig.update_layout(bargap=0.1)
fig.show()


# The graph above indicates that bulldozers categorized as "TEX" tend to have the highest prices, which could be attributed to a greater demand or potentially better compatibility. This observation is certainly worth further investigation

# In[29]:


fig = px.scatter(df_merged,
                 x='saleYear',
                 y='SalePrice',
                 color='ProductGroup',
                 opacity=0.5,
                 hover_data=['saleYear'],
                 log_x=True,
                 title='Yearmade vs SalePrice')
fig.update_traces(marker_size=5)
fig.show()


# An interesting trend can be seen in the above graph.
# 
# The vehicles an increasing trend where frthe vehicles which are more mordern are having higher prices.

# In[30]:


fig = px.histogram(df_merged,
                 x='Hydraulics',
                 y='SalePrice',
                 color='ProductGroup',
                 hover_data=['saleYear'],
                 title='Hydraulics vs. SalePrice',
                  histfunc='count')

fig.update_layout(bargap=0.1)
fig.show()


# 2 valve and Standard are most common in all product groups

# In[31]:


fig = px.histogram(df_merged,
                 x='ProductSize',
                 y='SalePrice',
                 color='ProductGroup',
                 hover_data=['saleYear'],
                 title='ProductSize vs. SalePrice',
                 histfunc='count')

fig.update_layout(bargap=0.1)
fig.show()


# In[32]:


fig = px.histogram(df_merged,
                 x='Enclosure',
                 y='SalePrice',
                 color='ProductGroup',
                 hover_data=['saleYear'],
                 title='Enclosure vs. SalePrice')

fig.update_layout(bargap=0.1)
fig.show()


# In[33]:


fig  = px.histogram(df_merged,
                   x='state',
                   y='SalePrice',
                   hover_data=['saleYear'],
                   title="State Vs SalePrice",
                   histfunc='count')
fig.update_layout(bargap=0.1)
fig.show()


# Where we can see that the bulldozers which are made in Florida or Texas have the highest price because
# 
# The product quality remains good in such cities.
# There would be a indrustrial area which leads to in high quality products in less input cost.

# In[34]:


fig = px.histogram(df_merged,
                 x='Blade_Width',
                 y='SalePrice',
                 hover_data=['saleYear'],
                 title='Blade_Width vs. SalePrice',
                 histfunc='count')

fig.update_layout(bargap=0.1)
fig.show()


# The bulldozers with medium blade width are having the highest price because people do not want too big or two small blades.

# In[35]:


df_merged[df_merged['saleYear']==2000]


# In[36]:


df_merged['saleYear'].unique()


# ## Input and Target Columns

# In[37]:


df_merged.columns


# In[38]:


input_cols = ['SalesID', 'MachineID', 'ModelID', 'datasource',
       'auctioneerID', 'YearMade', 'MachineHoursCurrentMeter', 'UsageBand',
       'fiModelDesc', 'fiBaseModel', 'fiSecondaryDesc', 'fiModelSeries',
       'fiModelDescriptor', 'ProductSize', 'fiProductClassDesc', 'state',
       'ProductGroup', 'ProductGroupDesc', 'Drive_System', 'Enclosure',
       'Forks', 'Pad_Type', 'Ride_Control', 'Stick', 'Transmission',
       'Turbocharged', 'Blade_Extension', 'Blade_Width', 'Enclosure_Type',
       'Engine_Horsepower', 'Hydraulics', 'Pushblock', 'Ripper', 'Scarifier',
       'Tip_Control', 'Tire_Size', 'Coupler', 'Coupler_System',
       'Grouser_Tracks', 'Hydraulics_Flow', 'Track_Type',
       'Undercarriage_Pad_Width', 'Stick_Length', 'Thumb', 'Pattern_Changer',
       'Grouser_Type', 'Backhoe_Mounting', 'Blade_Type', 'Travel_Controls',
       'Differential_Type', 'Steering_Controls', 'saleYear', 'saleMonth',
       'saleDay', 'MfgYear', 'fiManufacturerID', 'fiManufacturerDesc',
       'PrimarySizeBasis', 'PrimaryLower', 'PrimaryUpper']
target_col = 'SalePrice'


# ## Modelling
# 
# Building a machine learning model

# In[39]:


df.dtypes


# ## Converting objects to categories

# In[40]:


# Find the columns which contain strings
for label, columns in df_tmp.items():
    if pd.api.types.is_string_dtype(columns):
        df_tmp[label] = columns.astype('category').cat.as_ordered()


# In[41]:


df_tmp.info()


# In[42]:


df_tmp.state.cat.categories


# In[43]:


df_tmp.state.cat.codes


# In[44]:


print((df_tmp.isna().sum() / len(df_tmp) * 100).round(2).astype(str) + '%')


# ### Check numeric

# In[45]:


for label, columns in df_tmp.items():
    if pd.api.types.is_numeric_dtype(columns):
        print(label)


# In[46]:


# check for which numeric columns have null values
for label, columns in df_tmp.items():
    if pd.api.types.is_numeric_dtype(columns):
        if pd.isnull(columns).sum():
            print(label)


# In[47]:


# Fill numeric rows with the median
for label, columns in df_tmp.items():
    if pd.api.types.is_numeric_dtype(columns):
        if pd.isnull(columns).sum():
            # Add a binary column which telss us if the data was missing or not
            df_tmp[label+"_is_missing"] = pd.isnull(columns)
            # Fill missing numeric values with median
            df_tmp[label] = columns.fillna(columns.median())


# In[48]:


# Check if there is any null numeric value
for label, columns in df_tmp.items():
    if pd.api.types.is_numeric_dtype(columns):
        if pd.isnull(columns).sum():
            print(label)


# ### Filling and turning categorical variables into numbers

# In[49]:


# Check for columns which aren't numeric
for label, columns in df_tmp.items():
    if not pd.api.types.is_numeric_dtype(columns):
        print(label)


# In[50]:


# Converting the categorical variables
for label, columns in df_tmp.items():
    if not pd.api.types.is_numeric_dtype(columns):
        # Add a binary column to indicate missing value
        df_tmp[label+"_is_missing"] = pd.isnull(columns)
        # Turn categories into numbers and add + !
        df_tmp[label] = pd.Categorical(columns).codes + 1


# In[51]:


df_tmp.isna().sum()


# In[52]:


# Import the model
from sklearn.ensemble import  RandomForestRegressor

# Instantiate the model
model = RandomForestRegressor()

# Create X and y
X = df_tmp.drop('SalePrice', axis=1)
y = df_tmp['SalePrice']

# Fit the model
model.fit(X, y)


# In[59]:


# Score the model
model.score(df_tmp.drop('SalePrice', axis=1), df_tmp['SalePrice'])


# ## Splitting the data into train/Valid

# In[60]:


df_val = df_tmp[df_tmp.saleYear == 2012]
df_train = df_tmp[df_tmp.saleYear != 2012]

len(df_val), len(df_train)


# In[61]:


# Split the data into X and y
X_train , y_train = df_train.drop("SalePrice", axis=1), df_train.SalePrice
X_valid, y_valid = df_val.drop("SalePrice", axis=1), df_val.SalePrice

X_train.shape, y_train.shape, X_valid.shape, y_valid.shape


# ### Building a evaluation function

# In[63]:


# Create evaluation function for RMSLe
from sklearn.metrics import mean_squared_log_error, mean_absolute_error, r2_score

def rmsle(y_test, y_preds):
    """
    Calculates rmsle between predicions and true labels
    """
    return np.sqrt(mean_squared_log_error(y_test, y_preds))

# Create function to evaluate model on a few different labels
def show_scores(model):
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_valid)
    scores = {"Training MAE": mean_absolute_error(y_train, train_preds),
             "Valid MAE": mean_absolute_error(y_valid, val_preds),
             "Training RMSLE": rmsle(y_train, train_preds),
             "Valid RMSLE": rmsle(y_valid, val_preds),
             "Training R^2": r2_score(y_train, train_preds),
             "Valid R^2": r2_score(y_valid, val_preds)}
    return scores


# ### Testing the model on a subset

# In[64]:


# Use max_samples to input the number of rows
model = RandomForestRegressor(n_jobs=-1,
                             random_state=42,
                             max_samples= 10000)
model.fit(X_train, y_train)


# In[65]:


# Show scores
show_scores(model)


# ### Hyperparameter Tuning with RandomizedSearch CV
# 

# In[69]:


from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Different RandomForestRegressor Hyperparameters
rf_grid = {'n_estimators': np.arange(10, 100, 10),
           'max_depth': [None, 3, 5, 10],
           'min_samples_split': np.arange(2, 20, 2),
           'min_samples_leaf': np.arange(1, 20, 2),
           'max_features': [0.5, 1, 'sqrt', 'auto'],
           'max_samples': [10000]}

# Instantiate RandomizedSearchCV model
rs_model = RandomizedSearchCV(estimator=RandomForestRegressor(n_jobs=-1, random_state=42),
                              param_distributions=rf_grid,
                              n_iter=2, # You can increase this if you have more computing power
                              cv=5,
                              verbose=2,  # You can set verbose to 2 for more detailed output
                              random_state=42)  # Add random_state for reproducibility

# Fit the RandomizedSearchCV model
rs_model.fit(X_train, y_train)


# In[70]:


# Find the best Randomized parameters
rs_model.best_params_


# In[71]:


# Evaluate the RandomizedSearchCV model
show_scores(rs_model)


# ## Training a model with the best hypeparameters
# I visited stackoverflow and found these to be the most ideal hyperparameters tuned by engineers with enough computing power

# In[73]:


# Most ideal hyperparameters
ideal_model= RandomForestRegressor(n_estimators=40,
                                   min_samples_leaf=1,
                                   min_samples_split=14,
                                   max_features=0.5,
                                   n_jobs=-1,
                                   max_samples=None)

# Fit the ideal model
ideal_model.fit(X_train, y_train)


# In[74]:


show_scores(ideal_model)


# ## Making prediction on the test data

# In[107]:


test = pd.read_csv('/content/Test.csv',
                   low_memory=False,
                   parse_dates=['saledate'])
test


# In[108]:


print((test.isna().sum() / len(test) * 100).round(2).astype(str) + "%")


# In[109]:


test.info()


# # Fill in missing in Test data and convert into numerical

# In[110]:


# Create a function that
def preprocess_data(df):
  """
  Returns a transformed df
  """
  df['saleYear'] = df.saledate.dt.year
  df['saleMonth'] = df.saledate.dt.month
  df['saleDay']  = df.saledate.dt.day

  df.drop('saledate', axis=1, inplace=True)

  # Fill the numeric row with median
  for label, columns in df.items():
    if pd.api.types.is_numeric_dtype(columns):
      if pd.isnull(columns).sum():
        # Add binary column which tells us if the data was missing
        df[label+"_is_missing"] = pd.isnull(columns)
        # Fill missing numerics values with median
        df[label] = columns.fillna(columns.median())

  # Filled categorical missing data and turn categories into numbers
    if not pd.api.types.is_numeric_dtype(columns):
      df[label+"_is_missing"] = pd.isnull(columns)
      # Add +1 to the category code becaues pandas encodes missing categories
      df[label] = pd.Categorical(columns).codes + 1

  return df


# In[111]:


# Process the test data
df_test = preprocess_data(test)
df_test.head()


# In[112]:


# Make prediction on updated test data
test_preds= ideal_model.predict(df_test)


# ## Find how the columns difffer using python sets

# In[113]:


set(X_train.columns) - set(df_test.columns)


# In[118]:


# Manually adjust df_test to have auctioneerId_is_missing
df_test['auctioneerID_is_missing'] = False
df_test.head()


# Finally the test data  has the same features as the training data. Now make predictions

# In[119]:


# Make predictions on the test data
test_preds = ideal_model.predict(df_test)


# In[124]:


auctioneerID_index = X_train.get_loc('auctioneerID_is_missing')

# Insert 'auctioneerID_is_missing' into the corresponding position in test_names
df_test = list(df_test)
df_test.insert(auctioneerID_index, 'auctioneerID_is_missing')

# Convert test_names back to a pandas Index
df_test = pd.Index(df_test)

print(df_test))


# In[126]:


# Reorder the columns in df_test to match the order in X_train
df_test = df_test[X_train.columns]

# Make predictions on the test data
test_preds = ideal_model.predict(df_test)


# In[127]:


test_preds


# We'll, I've made predictions but they are not in the format that kaggle requires

# In[128]:


# Format in to meet the required kaggle format
df_preds = pd.DataFrame()
df_preds['SalesID'] = df_test["SalesID"]
df_preds['SalesPrice'] = test_preds
df_preds


# ## Feature Importance
# Which attributes of the data were most relevant in predicting the price?

# In[137]:


# Find Features importances
ideal_model.feature_importances_


# In[143]:


# Create a funcion to plot the feature importance
def plot_features(columns, importances, n=20):
  df = (pd.DataFrame({'features': columns,
                     'feature_importances': importances})
                     .sort_values('feature_importances', ascending=False)
                     .reset_index(drop=True))

  # Plot the dataframe
  fig, ax = plt.subplots()
  ax.barh(df['features'][:n], df['feature_importances'][:20])
  ax.set_ylabel('Features')
  ax.set_xlabel("Feature importance")
  ax.invert_yaxis()


# In[144]:


plot_features(X_train.columns, ideal_model.feature_importances_)


# ## Feature Importance Conclusion
# - **YearMade**: This feature suggests the year in which the equipment was made. It's likely ranked highest because the age of heavy machinery can significantly impact its value and condition.
# 
# - **ProductSize**: The size or capacity of the equipment often plays a crucial role in its functionality and cost.
# 
# - **SaleYear**: The year of the sale can be important as market conditions and economic factors change over time.
# 
# - **fiSecondaryDesc**: The secondary description of the product may describe its additional features, which can influence its value.
# 
# - **Enclosure**: The type of enclosure (e.g., open, cab) can impact the usability and cost of the machinery.
# 
# - **fiBaseModel**: The base model of the equipment serves as a fundamental characteristic.
# 
# - **fiProductClassDesc**: This description likely includes details about the product class, which can be indicative of its purpose and capabilities.
# 
# - **fiModelDesc**: The specific model description provides detailed information about the equipment.
# 
# - **ModelID**: An identifier for the model, which can help distinguish between different types of machinery.
# 
# - **SalesID**: A unique identifier for the sale event.
# 
# - **ProductSize_is_missing**: A binary indicator for missing values in the 'ProductSize' feature.
# 
# - **MachineID**: A unique identifier for the machine itself.
# 
# - **Coupler_System_is_missing**: A binary indicator for missing values in the 'Coupler_System' feature.
# 
# - **fiModelDescriptor**: Additional descriptors of the equipment model.
# 
# - **Coupler_System**: The type of coupler system used can be relevant to the equipment's versatility and value.
# 
# - **fiModelDescriptor_is_missing**: A binary indicator for missing values in the 'fiModelDescriptor' feature.
# 
# - **SaleMonth**: The month of the sale, which may reflect seasonal variations in equipment demand.
# 
# - **Blade_Width**: The width of the blade (applicable to certain types of machinery).
# 
# - **Tire_Size**: The size of the tires can affect mobility and performance.
# 
# - **State**: The location (state) where the equipment is sold can impact local market conditions.
# 

# ## Using XGBoost

# In[131]:


# Import XGBoost
from xgboost import XGBRegressor

# Instantiate the model
model_2 = XGBRegressor(n_jobs=-1, random_state=42, n_estimators=20, max_depth=4)

# Fit the model
model_2.fit(X_train, y_train)


# In[132]:


model_2.score(X_train, y_train)


# In[134]:


importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)


# In[135]:


importance_df.head(10)


# In[136]:


plt.title('Feature Importance')
sns.barplot(data=importance_df.head(10), x='importance', y='feature');


# In[145]:


import matplotlib.pyplot as plt
from xgboost import plot_tree
from matplotlib.pylab import rcParams
get_ipython().run_line_magic('matplotlib', 'inline')

rcParams['figure.figsize'] = 30,30


# In[146]:


plot_tree(model_2, rankdir='LR', num_trees=1);


# In[147]:


# Evaluating the XGB metrics
show_scores(model_2)


# ### XGBoost Hyperparameter Tuning
# While i did try to tune the hyperparameters, it is difficult to get a very big leap in performance by just using parameter tuning or slightly better models. The max score for GBM was 0.3362, while XGBoost gave 0.3371. This is a decent improvement but not something very substantial.

# ## Using Linear Regression

# In[148]:


# Import the model
from sklearn.linear_model import LinearRegression

# Instantiate the model
model_3 = LinearRegression()

# Fit the model
model_3.fit(X_train, y_train)


# In[150]:


model_3.score(X_train, y_train)


# In[ ]:




