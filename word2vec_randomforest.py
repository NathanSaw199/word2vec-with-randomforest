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

#Represents all of the words that our Word2Vec model learned a vector for.Or put another way, it’s all of the words that appeared in the training data at least twice. So you can exp
w2v_model.wv.index_to_key

#We can find the most similar words based on word vectors from our trained model, let find the most similar words for “king”
# Find the most similar words to "king" based on word vectors from our trained model
# print(w2v_model.wv.most_similar('movie'))

#Generate aggregated sentence vectors based on the word vectors for each word in the sentence

words = set(w2v_model.wv.index_to_key)
X_train_vect = [np.array([w2v_model.wv[i] for i in ls if i in words])
                         for ls in X_train]
X_test_vect = [np.array([w2v_model.wv[i] for i in ls if i in words])
                         for ls in X_test]
#So we’re going to loop through this array of arrays that we created in the step above. Each sentance have different number of array vectors which may cause an error while we training the model
# Why is the length of the sentence different than the length of the sentence vector?
for i, v in enumerate(X_train_vect):
    print(len(X_train.iloc[i]), len(v))

# Compute sentence vectors by averaging the word vectors for the words contained in the sentence
X_train_vect_avg = []
for v in X_train_vect:
    if v.size:
        X_train_vect_avg.append(v.mean(axis=0))
    else:
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