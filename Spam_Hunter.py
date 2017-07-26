import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_table('C:\\Test_Data\\smsspamcollection\\SMSSpamCollection',
				   sep='\t',
				   header=None,
				   names=['label', 'sms_message'])

df['label'] = df.label.map({'ham': 0, 'spam': 1})

# documents = ['Hello, how are you!',
# 			 'Win money, win from home',
# 			 'Call me now.',
# 			 'Hello, Call hello you tomorrow']

# lower_documents = []
# for i in documents:
# 	lower_documents.append(i.lower())
# 	
# import string
# sans_punctuations_documents = []
# for i in lower_documents:
# 	sans_punctuations_documents.append(i.translate(str.maketrans('', '', string.punctuation)))
# 
# preprocessing_documents = []
# for i in sans_punctuations_documents:
# 	preprocessing_documents.append(i.split(' '))
# 
# import pprint
# from collections import Counter
# frequency_list = []
# for i in preprocessing_documents:
# 	frequency_counts = Counter(i)
# 	frequency_list.append(frequency_counts)

# count_vector = CountVectorizer()
# 
# count_vector.fit(documents)
# count_vector.get_feature_names()
# doc_array = count_vector.transform(documents).toarray()
# frequency_matrix = pd.DataFrame(doc_array,
# 								columns = count_vector.get_feature_names())

X_train, X_test, y_train, y_test = train_test_split(df['sms_message'],
													df['label'],
													random_state=1)

# print('Number of rows in the total set: {}'.format(df.shape[0]))
# print('Number of rows in the training set: {}'.format(X_train.shape[0]))
# print('Number of rows in the test set: {}'.format(X_test.shape[0]))

count_vector = CountVectorizer()
training_data = count_vector.fit_transform(X_train)
testing_data = count_vector.transform(X_test)

naive_bayes = MultinomialNB()
naive_bayes.fit(training_data, y_train)

predictions = naive_bayes.predict(testing_data)

print('Accuracy score: ', format(accuracy_score(y_test, predictions)))
print('Precision score: ', format(precision_score(y_test, predictions)))
print('Recall score: ', format(recall_score(y_test, predictions)))
print('F1 score: ', format(f1_score(y_test, predictions)))