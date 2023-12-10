#!/usr/bin/env python
# coding: utf-8

# # Final Project
# ## Alexis Simmons
# ### 12/13/2023

# ### Q1

# In[5]:


import pandas as pd

s = pd.read_csv('social_media_usage.csv')
print("Dimensions (rows, columns):", s.shape)

num_rows = len(s)
print("Number of rows:", num_rows)

s.info()
s_description = s.describe()
print("Descriptive Statistics:")
print(s_description)


# ### Q2

# In[3]:


import pandas as pd
import numpy as np

def clean_sm(x):
    x = np.where(x == 1, 1, 0)
    return x

# Toy DataFrame
data = {
    'A': [1, 2, 0],
    'B': [0, 1, 1]
}
toy_df = pd.DataFrame(data)

toy_df_cleaned = toy_df.apply(clean_sm)
print(toy_df_cleaned)


# ### Q3

# In[16]:


import pandas as pd
import numpy as np

ss = s[['web1h', 'income', 'educ2', 'par', 'marital', 'gender', 'age']].copy()
ss.columns = ['sm_li', 'income', 'education', 'parent', 'married', 'female', 'age']
def clean_sm(x):
    x = np.where(x == 1, 1, 0)
    return x
ss['sm_li'] = clean_sm(ss['sm_li'])
ss['income'] = pd.to_numeric(ss['income'], errors='coerce')
ss['education'] = pd.to_numeric(ss['education'], errors='coerce')
ss['age'] = pd.to_numeric(ss['age'], errors='coerce')
ss.dropna(inplace=True)
print(ss.groupby('sm_li')['income'].value_counts())

import seaborn as sns
sns.pairplot(ss, hue='sm_li', vars=['income', 'education', 'age'])
print(ss.groupby('sm_li').mean())


# ### Q4

# In[5]:


y = ss['sm_li']
X = ss.drop('sm_li', axis=1)


# ### Q5

# In[17]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ####  The data is divided for training and testing for modeling. The training part (X_train) has 80% of the original data with features for training, and y_train tagged along with it, holding the target values used for training. The testing chunk (X_test) has 20% of the data with separate features to check how well the trained model performs. Its corresponding target values for evaluation were kept in y_test. We kept 20% for testing with 'test_size=0.2' and ensured consistent splitting using 'random_state=42'. This split lets us train the model and see how new data reacts. 

# ### Q6

# In[7]:


from sklearn.linear_model import LogisticRegression

logistic_model = LogisticRegression(class_weight='balanced', random_state=42)
logistic_model.fit(X_train, y_train)


# ### Q7

# In[8]:


from sklearn.metrics import accuracy_score, confusion_matrix

y_pred = logistic_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)


# ### Q8

# In[15]:


import pandas as pd
from sklearn.metrics import confusion_matrix

conf_matrix = confusion_matrix(y_test, y_pred)
columns = ['Predicted Not LinkedIn', 'Predicted LinkedIn']
index = ['Actual Not LinkedIn', 'Actual LinkedIn']
confusion_df = pd.DataFrame(conf_matrix, columns=columns, index=index)
print("Confusion Matrix Users:")
print(confusion_df)


# ### Q9

# #### Precision measures the accuracy of positive predictions by comparing correctly predicted positives to the total predicted positives (true positives + false positives). It's calculated as the ratio of true positives to the sum of true positives and false positives. Recall, also known as sensitivity or true positive rate, looks at the proportion of actual positives correctly predicted by the model. It's determined by the ratio of true positives to the sum of true positives and false negatives. The F1 score represents the mean of precision and recall, having a balanced measure, especially useful in scenarios with imbalanced classes. It's computed as 2 times the product of precision and recall divided by the sum of precision and recall.
# 
# #### In practice, precision might be more suitable when the cost of false positives is high. For example, in medical diagnosis, where false positives could lead to unnecessary treatments. Recall is needed when the cost of false negatives is high, such as in fraud detection, where missing a fraudulent transaction is more critical than a false alert. Lastly, the F1 Score balances precision and recall and is useful when there's an uneven class distribution or when both false positives and false negatives carry importance.
# 
# 
# 

# ### Q10

# In[18]:


import pandas as pd
from sklearn.linear_model import LogisticRegression

scenario_1 = pd.DataFrame([[8, 7, 2, 1, 1, 42]], columns=['income', 'education', 'parent', 'married', 'female', 'age'])
scenario_2 = pd.DataFrame([[8, 7, 2, 1, 1, 82]], columns=['income', 'education', 'parent', 'married', 'female', 'age'])
prob_scenario_1 = logistic_model.predict_proba(scenario_1)[:, 1]  
prob_scenario_2 = logistic_model.predict_proba(scenario_2)[:, 1]  

print(f"Probability of using LinkedIn for scenario 1: {prob_scenario_1[0]:.4f}")
print(f"Probability of using LinkedIn for scenario 2: {prob_scenario_2[0]:.4f}")

