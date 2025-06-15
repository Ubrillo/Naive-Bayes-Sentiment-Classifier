

import nltk #import nltk library for preprocessing
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer # library for
import spacy # import spacy library
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
from collections import defaultdict
import copy
import re # import regular expression
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nlp = spacy.load('en_core_web_sm')
import csv
import math

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer


file_path = 'Restaurant_Reviews.tsv' # Filepath to your TSV file
data = []
with open(file_path, 'r') as file:
    tsv_reader = csv.reader(file, delimiter='\t')  # Specify tab delimiter
    for row in tsv_reader:
        data.append((row[0], row[1]))
data.remove(('Review', 'Liked'))# remove the data heading from the list

from sklearn.model_selection import train_test_split
features = []
labels = []
for review in data:
    features.append(review[0])
    labels.append(review[1])
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42) #split dataset into train and test data
train_data = [(x, y) for x, y in zip(x_train, y_train)]
test_data = [(x, y) for x, y in zip(x_test, y_test)]

# NAIVE BAYES CLASSIFIER
#preprocess a specified text
#return the processed text as string
def preprocess(text):
    import string
    porter = PorterStemmer()
    stop_words = set(stopwords.words('english')) #remove stop words
    cleaned_text = re.sub(r'[^\w\s]', '', text)  # This removes all punctuations and non-numeric characters
    #cleaned_text = cleaned_text.replace('...', '') # Specifically remove ellipsis
    words = nltk.word_tokenize(cleaned_text)#tokenize text
    words = [word.lower() for word in words if word.lower() not in stop_words] #lower case each word
    words = [word for word in words if word not in string.punctuation]#removing punctuations
    text = ' '.join(words) #joins all words from the word list
    words = nlp(text)  #process the text with spacy
    words  = [token.lemma_ for token in words] #lemmatize each word
    text = ' '.join(words) #joins all elements words list

    return  text

#Calcuates the prior probability of the classes using train dataset
#calculates the term frequency of each word in the train dataset
#returns the the prior probability and term frequency
def review_prob(data):
    prior = defaultdict(int)
    frequency = defaultdict(lambda: defaultdict(int))
    for review, label in data:
        prior[label] += 1
        words = preprocess(review).split()
        for word in words:
            frequency[label][word] += 1
    total_reviews = len(data)
    for label in prior:
        prior[label] /= total_reviews
    return prior, frequency

#reivew - new review
#prior_prob - prior probability
#frequency - term frequency of each word in the doc
#return the class of a review
def classify_review(review, prior_prob, frequency):
    words = preprocess(review).split()
    posteriors = {}
    prob_positive = []
    prob_negative = []
    alpha = 1

    for label in prior_prob:
        class_size = sum(frequency[label].values())
        unique_words = len(frequency[label])
        posterior = prior_prob[label]
        log_posterior = math.log10(posterior)
        for word in words:
            prob = (frequency[label].get(word, 0) + alpha) / (class_size + (unique_words  * alpha))#applies laplacing smoothing
            if label == '1':
                prob_positive.append(prob)
            elif label == '0':
                prob_negative.append(prob)
            posterior *= prob
        posteriors[label] = posterior

    for pos, neg in zip(prob_positive, prob_negative):
        log_posterior += math.log10(pos/neg) #calcuate log_likelihood
    return max(posteriors, key=posteriors.get), log_posterior #return highest log-likelihood

#Evaluates naive bayes model
def naive_model_evaluation(review):
    prior_prob, frequency = review_prob(train_data)
    predictions = [] #stores prediction for each review in the test data
    for count, review in enumerate(test_data, start=1):
        result = classify_review(review[0], prior_prob, frequency)
        prediction = '1'
        if result[1] <= 0:
            prediction = '0'
        predictions.append(int(prediction))

    test_truth = [int(review[1]) for review in test_data] #retrieves test label of the dataset
    accuracy = accuracy_score(test_truth, predictions) #calculates accuracy
    precision = precision_score(test_truth, predictions) #calculates precision
    recall = recall_score(test_truth, predictions) #calculates recall
    f1 = f1_score(test_truth, predictions) #calculates F1-Score
    conf_matrix = confusion_matrix(test_truth, predictions) #calculates confusion matrix

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Confusion Matrix:\n", conf_matrix)

    # Plot the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels='')
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Naive Bayes Confusion Matrix')
    plt.show()

#classify new review using naive_bayes
#return string
def classify_naiveBayes(review):
    prior_prob, frequency = review_prob(train_data)
    result = classify_review(review, prior_prob, frequency)
    if result[1] <= 0:
        print(result)
        return "Negative Review"
    else:
        return "Positive Review"

#LOGISTIC REGRESSION
import pandas as pd
file_path = 'Restaurant_Reviews.tsv' # Filepath to your TSV file
# Reading TSV into a DataFrame and then converting to a list
df = pd.read_csv(file_path, sep='\t',
  converters = {
      'Column1': str.strip, #remove any trailing white space from column
      'Column2': str.strip,  #remove any trailing white space from column
  })
data = df.values.tolist()  # Convert DataFrame to a list of lists

def plotDistribution():
    result = df['Liked'].value_counts()
    result = result.reset_index()
    plt.figure(figsize=(7, 4))
    plt.bar(result['Liked'], result['count'], color=['green', 'red'], label=['positive', 'negative']) #plot bar  plot of the count of the labels in the dataset
    plt.title('Distribution of Liked Reviews')
    plt.xlabel('Liked')
    plt.ylabel('Count')
    plt.legend()
    plt.xticks(result['Liked'])  # Set x-ticks to match the "Liked" values (0 and 1)
    plt.show()

labels = df['Liked'] #retrive labels from the like colum of the dataframe
vectorizer = TfidfVectorizer() #loads tf-idf vectorizer
x_transformed  = vectorizer.fit_transform(df['Review']) #fit a corpus and transforms reviews docs
x_train, x_test, y_train, y_test = train_test_split(x_transformed, labels, test_size=0.2, random_state=42)#split the transformed dataset to train(80%) and test(20%)
model = LogisticRegression() #load regressin model
model.fit(x_train, y_train) #train the model with train dataset
y_pred = model.predict(x_test) #obtain prediction with test data

#Evaluates logistic regression
#visualize confusion matrix for logistic regresssion
def evauluate_logistic_regression():
    accuracy = accuracy_score (y_test , y_pred )# Calculate accuracy
    precision = precision_score(y_test, y_pred)# Calculate precision
    recall = recall_score(y_test, y_pred) # Calculate recall score
    f1 = f1_score(y_test, y_pred) # Calculate accuracy f1-score
    print (f" Accuracy : { accuracy }")
    print (f" Precision : { precision }")
    print (f" Recall : { recall }")
    print (f" F1 Score : { f1 }")

    # Confusion matrix
    conf_matrix2 = confusion_matrix(y_test, y_pred) #obtain the confusion matrix

    #plot confusion matrix for logistic regression
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix2, display_labels='')
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix for Logistic Regression')
    plt.show() #show plot


#predict and classify  a new review
def classify_logistic_regression(review):
    review_transformed = vectorizer.transform([review])
    prediction = model.predict(review_transformed)
    if prediction[0] == 1:
        return "Positive Review"
    else:
        return "Negative Review"

#ADVACNED PREDICTION WITH LOGISTIC REGRESSION FOR NEURAL SENTIMENT
#review - a new review text
def predict_review_advanced(review):
    prediction_prob = model.predict_proba(x_test) #obtain all prediction probabilities of test data
    review_transformed = vectorizer.transform([review]) #transform specified text using tf-idf
    prediction = model.predict(review_transformed) #obtain predictions using test data
    prediction_prob = model.predict_proba(review_transformed) #obtain all prediction probabilities of test data

    conf = max(prediction_prob[0]) #obtain highest prediction probability
    if conf > 0.6:
        if prediction[0] == 1:
            return "Positive Review"
        else:
            return "Negative Review"
    else:
        return "Neutral Review"



#ADVACNED PREDICTION WITH LOGISTIC REGRESSION ASPECT BASED SENTIMENT
import json
from sklearn.metrics import classification_report

#Load the JSONL file into a pandas DataFrame
def jsonl_to_dataframe(jsonl_file):
    data = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            entry = json.loads(line)
            text = entry['text']
            for label in entry['labels']:
                aspect = label['aspect']
                opinion = label['opinion']
                polarity = label['polarity']
                data.append({
                    'text': text,
                    'aspect': aspect,
                    'opinion': opinion,
                    'polarity': polarity
                })
    return pd.DataFrame(data)

# Parse the training JSONL file into a DataFrame
# Replace 'train_dataset.jsonl' with the path to your training JSONL file
train_dataframe = jsonl_to_dataframe('train.jsonl')
print("Parsed Training DataFrame:")
print(train_dataframe.head())

# Drop rows with NaN values
train_dataframe.dropna(inplace=True)

# Map polarity to binary labels ('positive' -> 1, 'negative' -> 0)
train_dataframe['label'] = train_dataframe['polarity'].map({'positive': 1, 'negative': 0})

# Ensure no NaN values remain in the label column
train_dataframe = train_dataframe[train_dataframe['label'].notna()]

# Combine aspect and opinion for input features
train_dataframe['features'] = train_dataframe['aspect'] + ' ' + train_dataframe['opinion']

# Vectorize the text using TF-IDF for training data
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_dataframe['features'])
y_train = train_dataframe['label']

# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Replace 'test_dataset.jsonl' with the path to your test JSONL file
test_dataframe = jsonl_to_dataframe('test.jsonl')
print("Parsed Test DataFrame:")
print(test_dataframe.head())

# Drop rows with NaN values
test_dataframe.dropna(inplace=True)

# Map polarity to binary labels ('positive' -> 1, 'negative' -> 0)
test_dataframe['label'] = test_dataframe['polarity'].map({'positive': 1, 'negative': 0})

# Ensure no NaN values remain in the label column
test_dataframe = test_dataframe[test_dataframe['label'].notna()]

# Combine aspect and opinion for input features
test_dataframe['features'] = test_dataframe['aspect'] + ' ' + test_dataframe['opinion']

#Vectorize the text using the same TF-IDF vectorizer for test data
X_test = vectorizer.transform(test_dataframe['features'])
y_test = test_dataframe['label']

#Evaluate the model on the test data
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))


print()
print()
