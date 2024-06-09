import pandas as pd  #Importing pandas library for data manipulation and analysis
from sklearn.feature_extraction.text import TfidfVectorizer  # For calculating TF-IDF values
from sklearn.model_selection import train_test_split  # For splitting the dataset into training and testing sets
from sklearn.naive_bayes import MultinomialNB  # Naïve Bayes classifier
from sklearn.svm import SVC  # Support Vector Machine classifier
from sklearn.tree import DecisionTreeClassifier  # Decision Tree classifier
from sklearn.metrics import accuracy_score, f1_score  # For evaluating the performance of the classifiers

# Step 1: Load and preprocess the dataset
data = pd.read_csv('data.csv')  # Load the dataset from the CSV file
data['Review Text'] = data['Review Text'].str.lower().replace('[^\w\s]', '').replace('\d+', '').strip()
# Preprocess the 'Review Text' column by converting to lowercase, removing punctuation, and stripping whitespace

# Step 2: Calculate TF-IDF values
tfidf = TfidfVectorizer()  # Create an instance of the TF-IDF vectorizer
tfidf_matrix = tfidf.fit_transform(data['Review Text'])  # Calculate the TF-IDF matrix
tfidf_data = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names())
# Convert the TF-IDF matrix to a DataFrame with feature names as columns

# Step 3: Add the "Label" column based on the "Rating" column
data['Label'] = data['Rating'].apply(lambda x: 1 if x > 3 else 0 if x == 3 else -1)
# Add a new column 'Label' based on the 'Rating' column using lambda function and apply()

# Step 4: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(tfidf_data, data['Label'], test_size=0.1, random_state=42)
# Split the dataset and assign the training and testing data to respective variables

# Step 5: Train and evaluate the machine learning algorithms
naive_bayes = MultinomialNB()  # Create an instance of Naïve Bayes classifier
svm = SVC()  # Create an instance of Support Vector Machine classifier
decision_tree = DecisionTreeClassifier()  # Create an instance of Decision Tree classifier

naive_bayes.fit(X_train, y_train)  # Train Naïve Bayes classifier
svm.fit(X_train, y_train)  # Train Support Vector Machine classifier
decision_tree.fit(X_train, y_train)  # Train Decision Tree classifier

nb_predictions = naive_bayes.predict(X_test)  # Generate predictions using Naïve Bayes classifier
svm_predictions = svm.predict(X_test)  # Generate predictions using Support Vector Machine classifier
dt_predictions = decision_tree.predict(X_test)  # Generate predictions using Decision Tree classifier

# Step 6: Calculate accuracy and F1 score for each algorithm
nb_accuracy = accuracy_score(y_test, nb_predictions)  # Calculate accuracy using predicted and true labels
nb_f1_score = f1_score(y_test, nb_predictions, average='macro')  # Calculate F1 score using predicted and true labels
svm_accuracy = accuracy_score(y_test, svm_predictions)  # Calculate accuracy using predicted and true labels
svm_f1_score = f1_score(y_test, svm_predictions, average='macro')  # Calculate F1 score using predicted and true labels
dt_accuracy = accuracy_score(y_test, dt_predictions)  # Calculate accuracy using predicted and true labels
dt_f1_score = f1_score(y_test, dt_predictions, average='macro')  # Calculate F1 score using predicted and true labels

# Step 7: Compare the performance of the algorithms
print("Naïve Bayes Accuracy:", nb_accuracy)  # Print the accuracy of Naïve Bayes classifier
print("Naïve Bayes F1 Score:", nb_f1_score)  # Print the F1 score of Naïve Bayes classifier
print("Support Vector Machine Accuracy:", svm_accuracy)  # Print the accuracy of Support Vector Machine classifier
print("Support Vector Machine F1 Score:", svm_f1_score)  # Print the F1 score of Support Vector Machine classifier
print("Decision Tree Accuracy:", dt_accuracy)  # Print the accuracy of Decision Tree classifier
print("Decision Tree F1 Score:", dt_f1_score)  # Print the F1 score of Decision Tree classifier

# Step 8: Compare the algorithms
if nb_accuracy > svm_accuracy and nb_accuracy > dt_accuracy:
    print("Naïve Bayes performs better.")  # Print the result if Naïve Bayes has the highest accuracy
elif svm_accuracy > nb_accuracy and svm_accuracy > dt_accuracy:
    print("Support Vector Machine performs better.")  # Print the result if Support Vector Machine has the highest accuracy
else:
    print("Decision Tree performs better.")  # Print the result if Decision Tree has the highest accuracy