# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# Load Data Sets
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

print(train_df.shape)
train_df.head()
# Exploratory Data Analysis (EDA) 
train_df.isnull().sum()
# Visualizing Survival Counts 
sns.countplot(x='Survived', data=train_df)
plt.title('Survival Count')
plt.show()
#  Survival by Sex 
sns.countplot(x='Survived', hue='Sex', data=train_df)
plt.title('Survival Rate by Gender')
plt.show()
#  Age Distribution 
train_df['Age'].plot.hist(bins=30, figsize=(10,5))
# class Distribution by Pclass 
sns.countplot(x='Pclass', data=train_df)