import numpy as np
import pandas as pd
import re
import gensim
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 255)


#Step 2: Load the Dataset

#Step 3: Preprocess the Text

#Step 4: Encode Labels

#Step 5: Split the Dataset

#Step 6: Train the Word2Vec Model

#Step 7: Convert Reviews to Sentence Vectors 
#Retrieve Vocabulary - >Obtain the vocabulary learned by the Word2Vec model.
#Convert Sentences to Word Vectors - >For each sentence, retrieve vectors for words that exist in the vocabulary.
#Compute Sentence Vectors (Averages) - > For each sentence, compute the average of the word vectors.

#Step 8: Train a Random Forest Classifier

#Step 9: Make Predictions

#Step 10: Evaluate the Model

df = pd.read_csv(r"C:\Users\Saw\Desktop\GEN_AI\IMDB Dataset.csv")

def remove_tags(raw_text):
    cleaned_text = re.sub(re.compile('<.*?>'), '', raw_text)
    return cleaned_text
df['review'] = df['review'].apply(remove_tags)

df['review'] = df['review'].apply(lambda x:x.lower())

#We’re going to use the built-in function from gensim to handle the cleaning and tokenization for us. So we’re calling gensim’s cleaner, which is gensim.utils.simple_preprocess. This will remove all punctuation, remove stop words and tokenize the given sentence.
df.columns = ["review","sentiment"]
# print(df.head())

# Clean data using the built in cleaner in gensim
df['review_clean'] = df['review'].apply(lambda x : gensim.utils.simple_preprocess(x))
# print(df.head())

#Now we’re going to go ahead and create our training and test set with 20% going to the test set

# Encoding the label column
df['label'] = df['sentiment'].map({'positive' : 1, 'negative' : 0})
# Split data into train and test sets
X_train, X_test, y_train,y_test = train_test_split(df["review_clean"],df["label"],test_size=0.2)

#Now training our model calling the word2vec model from the gensim package by passing necessary parameters are.
#size - size of the vectors we want
#window - number words before and after the focus word that it’ll consider as context for the word
#min_count - the number of times a word must appear in our corpus in order to create a word vector.

# Train the word2vec model
w2v_model = gensim.models.Word2Vec(X_train,vector_size=100,window=5,min_count=2)

#Retrieves the vocabulary (all unique words) learned by the Word2Vec model during training.
w2v_model.wv.index_to_key

#This part of the code converts each sentence in the training and test datasets into arrays of word vectors using the trained Word2Vec model.
#set(): Converts the list of words into a Python set for faster membership checking (i in words) when processing sentences.
words = set(w2v_model.wv.index_to_key)
#the variable words now contains all the words in the Word2Vec vocabulary.
#for ls in X_train: Iterates over each preprocessed review (sentence) in the training dataset X_train. Each ls represents a tokenized sentence (list of words). ls is a list of words (tokens) from a single review
#X_train = [["movie", "amazing"],["film", "boring", "waste"]] .First iteration: ls = ["movie", "amazing"] .Second iteration: ls = ["film", "boring", "waste"]
#Iterates over each tokenized sentence (ls) in the training dataset X_train.
#If a review is "The movie was amazing", after preprocessing: ls = ["movie", "amazing"] 
#[w2v_model.wv[i] for i in ls if i in words]
#For each word i in the tokenized sentence ls, check if i is in the words set (i.e., if it has a Word2Vec representation)
#for i in ls if i in words: For each word i in the sentence ls Check if the word exists in the Word2Vec vocabulary (words) If the word exists, retrieve its vector representation using w2v_model.wv[i].
#If "movie" and "amazing" exist in the vocabulary. w2v_model.wv["movie"] → [vector representation of "movie"] . 2v_model.wv["amazing"] → [vector representation of "amazing"] 
#Suppose the Word2Vec model has learned the following vector representations: w2v_model.wv["movie"] = [0.1, 0.3, ..., 0.2]  # A 100-dimensional vector . w2v_model.wv["amazing"] = [0.5, 0.4, ..., 0.1] 
#For ls = ["movie", "amazing"], the inner list comprehension produces [w2v_model.wv["movie"], w2v_model.wv["amazing"]]
#Resulting in: [[0.1, 0.3, ..., 0.2],  # Vector for "movie"
# [0.5, 0.4, ..., 0.1]   # Vector for "amazing"]
#Convert to NumPy Array: np.array([...]) Converts the list of Word2Vec vectors for the words in a sentence into a NumPy array.Each sentence becomes a 2D array where each row is a 100-dimensional vector representing a word.
#Example: For ls = ["movie", "amazing"], the result might look like:
#np.array([
#     [0.1, 0.3, ..., 0.2],  # Vector for "movie"
#     [0.5, 0.4, ..., 0.1]   # Vector for "amazing"
# ])
#The result is a list of NumPy arrays, where each array represents a tokenized sentence.
#Each array contains the Word2Vec vectors for the words in that sentence.


#STEPS Example
# X_train = [
#     ["movie", "amazing"], 
#     ["film", "boring", "waste"]
# ]
#words = {"movie", "amazing", "film", "boring", "waste"}  # Words learned by Word2Vec
#If the Word2Vec vectors are:
#w2v_model.wv["movie"] = [0.1, 0.3, ..., 0.2]
# w2v_model.wv["amazing"] = [0.5, 0.4, ..., 0.1]
# w2v_model.wv["film"] = [0.2, 0.6, ..., 0.8]
# w2v_model.wv["boring"] = [0.9, 0.7, ..., 0.3]
# w2v_model.wv["waste"] = [0.4, 0.2, ..., 0.5]
#The result of X_train_vect will be:
# [
#     np.array([
#         [0.1, 0.3, ..., 0.2],  # Vector for "movie"
#         [0.5, 0.4, ..., 0.1]   # Vector for "amazing"
#     ]),
#     np.array([
#         [0.2, 0.6, ..., 0.8],  # Vector for "film"
#         [0.9, 0.7, ..., 0.3],  # Vector for "boring"
#         [0.4, 0.2, ..., 0.5]   # Vector for "waste"
#     ])
# ]
X_train_vect = [np.array([w2v_model.wv[i] for i in ls if i in words])for ls in X_train]

X_test_vect = [np.array([w2v_model.wv[i] for i in ls if i in words])
                         for ls in X_test]


# Compute sentence vectors by averaging the word vectors for the words contained in the sentence
#This will store the computed sentence vectors for the training data.
X_train_vect_avg = []
#Each v represents the NumPy array for a single sentence.v is a 2D array where each row corresponds to the Word2Vec vector of a word in the sentence.
for v in X_train_vect:
    #Check If the Sentence Has Word Vectors: v.size checks if the array has any elements.If v is empty (e.g., all words in the sentence are out of vocabulary), proceed to the else block.
    if v.size:
        #Averages the vectors along the rows (i.e., across words) to produce a single 100-dimensional vector.
        X_train_vect_avg.append(v.mean(axis=0))
    else:
        #If a sentence has no valid words (e.g., all words are not in the Word2Vec vocabulary), the vector is set to a zero vector of length 100 (matching the dimensionality of Word2Vec vectors).
        #         v = np.array([
        #     [0.1, 0.2, 0.3],  # Word 1
        #     [0.4, 0.5, 0.6]   # Word 2
        # ])
        # v.mean(axis=0) = [0.25, 0.35, 0.45]  # Sentence vector
        X_train_vect_avg.append(np.zeros(100, dtype=float))
        
X_test_vect_avg = []
for v in X_test_vect:
    if v.size:
        X_test_vect_avg.append(v.mean(axis=0))
    else:
        X_test_vect_avg.append(np.zeros(100, dtype=float))
#Testing the length of the X_train_vect_avg


# Are our sentence vector lengths consistent?
# for i, v in enumerate(X_train_vect_avg):
#     print(len(X_train.iloc[i]), len(v))

#Now our training set is ready, let’s prepare the model.
#Fit RandomForestClassifier On Top Of Word Vectors
#So we’ll just import our random forest classifier, we’ll use our default parameters, and then we’ll train it on the averaged word vectors (X_train_vect_avg, y_train) .

## Instantiate and fit a basic Random Forest model on top of the vectors
rf = RandomForestClassifier()
rf_model = rf.fit(X_train_vect_avg,y_train.values.ravel())

#Now that we have our fit model, we’re going to call dot predict on that fit model and use the patterns that it learned in its training process and apply those to unseen text messages in the test data(X_test_vect_avg) and store those predictions in y_pred

## Use the trained model to make predictions on the test data
y_pred = rf_model.predict(X_test_vect_avg)

#And then lastly, we’ll import our evaluation functions, precision and recall, we’ll calculate those metrics, and then we’ll print them out.

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
accuracy = accuracy_score(y_test,y_pred)
# print('Precision: {} / Recall: {} / Accuracy: {}'.format(
#     round(precision, 3), round(recall, 3), round((y_pred==y_test).sum()/len(y_pred), 3)))

print(precision,recall,accuracy)