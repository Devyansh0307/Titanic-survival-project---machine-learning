import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

print(train_df.shape)
train_df.head()

test_df['Age'].fillna(test_df['Age'].median(), inplace=True)
test_df['Fare'].fillna(test_df['Fare'].median(), inplace=True)
test_df.drop('Cabin', axis=1, inplace=True)

sex = pd.get_dummies(test_df['Sex'], drop_first=True)
embarked = pd.get_dummies(test_df['Embarked'], drop_first=True)

test_df = pd.concat([test_df, sex, embarked], axis=1)
test_df.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)

# Re-import and preprocess train dataset
train_df = pd.read_csv("train.csv")

# Fill missing values
train_df['Age'].fillna(train_df['Age'].median(), inplace=True)
train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)
train_df['Fare'].fillna(train_df['Fare'].median(), inplace=True)
train_df.drop('Cabin', axis=1, inplace=True)

# Convert categorical variables
sex = pd.get_dummies(train_df['Sex'], drop_first=True)
embarked = pd.get_dummies(train_df['Embarked'], drop_first=True)
train_df = pd.concat([train_df, sex, embarked], axis=1)
train_df.drop(['Sex', 'Embarked', 'Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)

# Split features and target
X = train_df.drop("Survived", axis=1)
y = train_df["Survived"]

# Train model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Align test data columns with training columns
test_df = test_df.reindex(columns=X.columns, fill_value=0)

test_predictions = model.predict(test_df)

submission = pd.DataFrame({
    "PassengerId": pd.read_csv('test.csv')['PassengerId'],
    "Survived": test_predictions
})

submission.to_csv('titanic_submission.csv', index=False)

feature_importance = pd.Series(model.feature_importances_, index=X.columns)
feature_importance.nlargest(10).plot(kind='barh')
plt.title('Feature Importance')
plt.show()