# AI Assignment #2 - Exploratory Data Analysis

This repository contains code for an exploratory data analysis (EDA) assignment. The purpose of this analysis is to gain insights from a dataset and perform classification using a decision tree classifier.

## Dataset

The dataset used for this analysis is the "stellaris-crew-data.csv" file. It contains information about crew members, including their age, gender, technical skills, specialized role, assigned task, and other attributes. The dataset is sourced from [link to the dataset source, if applicable].

## Libraries Used

The following libraries were imported and used in the code:

- Pandas: For data manipulation and analysis
- Numpy: For numerical computations
- Matplotlib and Seaborn: For data visualization
- Scipy: For statistical tests
- Scikit-learn: For data preprocessing, model training, and evaluation

Run the following command to download all the used libraries<br/>
- pip install pandas numpy matplotlib seaborn scipy scikit-learn pydotplus IPython


## Data Cleaning

The dataset was first inspected for duplicate and missing values. It was found that there were no duplicates or missing values in the dataset. The "crew_member_id" column, which did not provide useful information, was dropped from the dataset.

## Univariate Analysis

The analysis of individual columns was performed to understand the trends and distributions within the dataset. Histograms were plotted to visualize the age distribution and density plots were used to analyze the distribution of technical skills. The gender distribution and specialized role distribution were also visualized using bar plots.

## Multivariate Analysis

The relationships between different columns were explored to understand their dependencies. Cross-tabulation was used to analyze the association between specialized roles and assigned tasks. A chi-square test was conducted to confirm the statistical significance of this association. Scatter plots were created to investigate the relationship between experience and specialized role, as well as time since last meal and specialized role.

## Encoding

Label encoding was applied to the "gender" and "specialized_role" columns to convert categorical variables into numerical representations. Standardization was performed on the "age" and "experience" columns using the StandardScaler from scikit-learn.

## Splitting the Data

The dataset was split into training and testing sets with a test size of 30%. Stratified sampling was used to ensure that the proportions of specialized roles were maintained in both the training and testing sets.

## Training

A decision tree classifier was trained on the training set using the DecisionTreeClassifier from scikit-learn.

## Predicting

The trained classifier was used to predict the specialized roles of the crew members in the testing set. The accuracy of the model was calculated as the percentage of correctly classified instances.

## Report

A classification report was generated, providing precision, recall, and F1-score metrics for each specialized role. These metrics measure the performance of the model in classifying crew members into their respective roles.

## Decision Tree

A full classification tree and a smaller version of the tree (limited to a maximum depth of 3) were visualized using the graphviz library. These trees illustrate the decision-making process of the classifier, with nodes representing features and branches representing decision rules.

## Conclusion

The analysis revealed that age was the most important feature in the classification, while technical skill had a relatively low impact. The decision tree model achieved high precision and recall for most specialized roles, indicating its effectiveness in classifying crew members.

For more details and the actual code, please refer to the [live notebook](https://colab.research.google.com/drive/161q5y_9IIcrU401uiVk2o-XlHLC7RCef?usp=sharing).

