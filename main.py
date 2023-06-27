# -*- coding: utf-8 -*-
"""
Original file is located at
    https://colab.research.google.com/drive/161q5y_9IIcrU401uiVk2o-XlHLC7RCef

# AI Assignment #2

Topic: **Exploratory Data Analysis**
<br>
<br>
Following is the link to actual live notebook:
https://colab.research.google.com/drive/161q5y_9IIcrU401uiVk2o-XlHLC7RCef?usp=sharing

importing libraries
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from scipy.stats import chi2_contingency
plt.style.use('ggplot')
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

"""Loading our Dataset and Getting Familoar with the Data"""

df = pd.read_csv('stellaris-crew-data.csv')

"""# Data Cleaning


*   Understand our Data at Suface Level
*   Look through our Dataset for Duplicate and Missing Values
*   Obtaining a Subset of data that's more practical to work with

Here We are Checking The Shape (Row , Columns)
"""

df.shape

"""Looking at few values from our dataset to get basic understanding of what kind of information we are dealing with"""

df.head()

"""Checking Datatypes for all the columns our Dataset so that we know every columns is assigned the right datatype."""

df.dtypes

"""Here We are Checking If there are any Missing Valus.<br>
There are no Missing Values.<br>

"""

df.isna().sum()

"""Checking for duplicated rows."""

df.loc[df.duplicated()]

"""Checking if any crew_member_id has been repeated.<br>
Since it is a column that is supposed to be unique<br>
"""

df.loc[df.duplicated(subset=['crew_member_id'])]

"""Dropping crew_member_id as this column does not provide us with any useful information"""

df = df.drop(['crew_member_id'],axis=1)
df.head()

"""Describe Function Provides us with useful information about or dataset (e.g counts, mean etc.)"""

df.describe()

"""#Univariate analysis


*   Analyzing our Columns Individually
*   Plotting Histograms to get Better Understanding of Trends


"""

df['age'].value_counts()

"""Visualizing Age<br>
From This we get the info that most group members are above the age of 30
"""

ax = df['age'].plot(kind='hist',bins=10,title='Age Trend')
ax.set_xlabel('Age in Years')

"""Here We are Trying To Analyze the Trends in Technical Skills.
Generally the Technical Skills lie between 4-5
"""

ax = df['technical_skill'].plot(kind='kde',title='Technical Skills Trend')
ax.set_xlabel('Technical Skills')

"""Plotting Gender Distribution"""

gender_counts = df['gender'].value_counts()
plt.bar(gender_counts.index, gender_counts.values)

plt.xlabel('Gender')
plt.ylabel('Count')
plt.title('Gender Distribution')

"""Plotting Role Distribution:
We can Conclude From Plotting this that we have good balanced distribution for all the specialized roles
"""

specialized_role_counts = df['specialized_role'].value_counts()
plt.bar(specialized_role_counts.index, specialized_role_counts.values)

plt.xlabel('Specialized Role')
plt.ylabel('Count')
plt.title('Specialized Role Distribution')

"""#Multivariate Analysis


*   Here We will try to understand the relationship between different columns
*   Understand our Data how It relates

We Suspected that Specialized Role is Mapped to an Assigned Task
Here We Confirmed This using Cross Tab
"""

cross_tab = pd.crosstab(df['specialized_role'], df['assigned_task'])
print(cross_tab)

"""Since The Two Variables Seem to Be Co Dependent We Will try Performing Chi2Test to Confirm this.<br>

Results: <br>
Since the p-value is smaller than the significance level of 0.05, we can reject the null hypothesis of independence. This indicates that there is a statistically significant association between the 'specialized_role' and 'assigned_task' variables in your dataset.
"""

contingency_table = pd.crosstab(df['specialized_role'], df['assigned_task'])

chi2, p_value, dof, expected = chi2_contingency(contingency_table)

print("Chi-square test statistic:", chi2)
print("P-value:", p_value)
print("Degrees of freedom:", dof)
print("Expected frequencies:")
print(expected)

plt.figure(figsize=(8, 6))
sns.heatmap(contingency_table, cmap='YlGnBu', annot=True, fmt='d')
plt.xlabel('Assigned Task')
plt.ylabel('Specialized Role')
plt.title('Association between Specialized Role and Assigned Task')
plt.show()

"""We are going to drop Assigned Task as it overfits our model"""

df=df.drop(['assigned_task'],axis=1)

"""Now we are going to plot how experience relates to specialized_role x assigned_task"""

ax = sns.scatterplot(x='experience',
                y='specialized_role',
                hue='gender',
                data=df)
plt.show()

"""Now we are going to plot how time since last meal relates to specialized_role x assigned_task"""

ax = sns.scatterplot(x='time_since_last_meal',
                y='specialized_role',
                data=df)
plt.show()

"""Time Since Last Meal Does not provide any valuble info so we are going to drop this column"""

df = df.drop(['time_since_last_meal'],axis=1)
df.head()

"""#Encoding:"""

label_encoder = LabelEncoder()
df['gender_encoded'] = label_encoder.fit_transform(df['gender'])
df['specialized_role_encoded'] = label_encoder.fit_transform(df['specialized_role'])

print(df.head())

scaler = StandardScaler()
df['age_standardized'] = scaler.fit_transform(df[['age']])
df['experience_standardized'] = scaler.fit_transform(df[['experience']])

df=df.drop(['age', 'gender', 'experience', 'specialized_role'],axis=1)
print(df.head())
print(df.columns)

"""#Splitting:"""

X = df.drop('specialized_role_encoded', axis=1)
y = df['specialized_role_encoded']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

train_role_proportions = y_train.value_counts(normalize=True)
test_role_proportions = y_test.value_counts(normalize=True)

print("Training Set Specialized Role Proportions:")
print(train_role_proportions)

print("\nTesting Set Specialized Role Proportions:")
print(test_role_proportions)

"""#Training:"""

clf = DecisionTreeClassifier()

clf.fit(X_train, y_train)

"""#Prdicting:"""

y_pred = clf.predict(X_test)
accuracy = (y_pred == y_test).mean()
print("Accuracy:", accuracy)

"""#Report

* Precision: It measures the accuracy of the positive predictions (crew members correctly classified as a particular specialized role). In this case, the precision is very high, indicating that the model rarely misclassifies crew members.

* Recall: It represents the ability of the classifier to identify all positive instances correctly. A high recall suggests that the model can effectively capture crew members with a specific specialized role.

* F1-score: It is the harmonic mean of precision and recall, providing a balanced measure of the model's performance. The F1-score considers both precision and recall, and a high F1-score indicates a good balance between the two metrics.
"""

print("Classification Report:")
report = classification_report(y_test, y_pred)
print(report)

"""#Decision Tree"""

from sklearn.tree import export_graphviz
import pydotplus
from IPython.display import Image

class_names = [str(class_name) for class_name in clf.classes_]

dot_data = export_graphviz(clf, out_file=None, feature_names=X.columns, class_names=class_names, filled=True, rounded=True)


graph = pydotplus.graph_from_dot_data(dot_data)


print("Full Classification Tree:")
Image(graph.create_png())

small_tree_dot_data = export_graphviz(clf, out_file=None, feature_names=X.columns, class_names=class_names, filled=True, rounded=True, max_depth=3)

small_tree_graph = pydotplus.graph_from_dot_data(small_tree_dot_data)


print("Smaller Classification Tree:")
Image(small_tree_graph.create_png())

"""What Tree Tells Us:



*   The most important featur in the classification is age
*   The Gini Index for Age is 0.75 which show that it splits our data set really effectively
* The least effective feature is technical skill as its distribution was really bad as we evaluated from univariate analysis. That most people were skilled between 3.0-4.0


"""