import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("adult.csv", header=None)

df.head()
columns = ["Age" , "Job", "Weights", "Education", "Education-Num", "Maritial-Status", "Occupation",
           "Relationship", "Race", "Gender", "Capital-gain", "Capital-loss", "Weekly-hours", "Country" ,"Income"]

df.columns = columns
df.head()
print(f'Head:\n{df.head()}')
print(f'Tail:\n{df.tail()}')
print(f'Shape:\n{df.shape}')
print(f'Information:\n{df.info()}')
print(f'Description:\n{df.describe()}')
df.replace('?',np.nan, inplace = True)
df.dropna(inplace = True)
duplicates = df[df.duplicated]
df.drop_duplicates(inplace = True)

for c in df.select_dtypes(include=['object']).columns:
    if c != 'Income':
        df[c] = df[c].astype('category').cat.codes

for column in df.columns:
    print(f"Unique values in {column}: {df[column].unique()}")
    print(f"Value counts for {column}:\n{df[column].value_counts()}\n")

def norm_min_max(category):
    return (category - category.min()) / (category.max() - category.min())
numerical = ['Age', 'Weights', 'Education-Num', 'Weekly-hours']
for num in numerical:
    df[num] = norm_min_max(df[num])

df['Income'] = df['Income'].astype(str)
df['Income'] = df['Income'].apply(lambda x: 1 if x.strip() == '>50K' else 0)
df.head()
print("Correlation matrix:")
numerical_columns = df.select_dtypes(include=[np.number]).columns
correlation_matrix = df[numerical_columns].corr()
print(correlation_matrix)

print("Box plots for numerical features:")
df[numerical].plot(kind='box', subplots=True, layout=(2, 2), figsize=(10, 10))

print("Distribution plots for numerical features:")
df[numerical].hist(figsize=(10, 10))

def split(data, test_size=0.2):
    np.random.seed(42)
    shuffle = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_size)
    test_indices = shuffle[:test_set_size]
    train_indices = shuffle[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]
train_data, test_data = split(df)

X_train = train_data.drop("Income", axis=1)
y_train = train_data["Income"]
X_test = test_data.drop("Income", axis=1)
y_test = test_data["Income"]
print(f'Y Train: {y_train[:10]}')
print(f'Y Test {y_test[:10]}')
def evaluation(y_test,y_predicted):
    accuracy = np.mean(y_test == y_predicted)
    
    true_positive = np.sum((y_test == 1) & (y_predicted == 1))
    false_positive = np.sum((y_test == 0) & (y_predicted == 1))
    precision = true_positive / (true_positive + false_positive)
    
    false = np.sum((y_test == 1) & (y_predicted == 10))
    recall = true_positive / (true_positive + false)

    f1 = 2 * (precision * recall) / (precision + recall)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_gnb = gnb.predict(X_test)

bnb = BernoulliNB()
bnb.fit(X_train, y_train)
y_pred_bnb = bnb.predict(X_test)

mnb = MultinomialNB()
mnb.fit(X_train, y_train)
y_pred_mnb = mnb.predict(X_test)

print("Evaluation for Gaussian Naive Bayes:")
evaluation(y_test.values, y_pred_gnb)
print("Evaluation for Bernoulli Naive Bayes:")
evaluation(y_test.values, y_pred_bnb)
print("Evaluation for Multinomial Naive Bayes:")
evaluation(y_test.values, y_pred_mnb)