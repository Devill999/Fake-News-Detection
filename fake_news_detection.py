import numpy as np
import pandas as pd
#importing datasets
dataset= pd.read_csv('train.csv')
print(dataset.shape)
dataset.head() #shows the first 5 entries of the dataset
#counting the number of missing values in the dataset
print(dataset.isnull().sum())
#filling the null values with empty string
dataset = dataset.fillna(' ')
#merging author name and news title for better results
dataset['content'] = dataset['author']+' '+ dataset['title']
#cleaning words and stemming
import re
import nltk
nltk.download('stopwords')          #downloads stopwords
from nltk.corpus import stopwords   #imports stopwords to our library 
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)  #replaces all characters which are not alphabets with space
    stemmed_content = stemmed_content.lower()     #converts all uppercase letters to lowercase
    stemmed_content = stemmed_content.split()     #splits all letters and keeps it in a list
    stemmed_content = [ps.stem(word) for word in stemmed_content if not word in stopwords.words('english')] #stemming process occurs and all the stopwords are removed
    stemmed_content = ' '.join(stemmed_content)#the splitted word are joined after stemming
    return stemmed_content
dataset['content'] = dataset['content'].apply(stemming)  #calling the function
print(dataset['content'])
#separating the data and label
x = dataset['content'].values
y = dataset['label'].values
#vectorizing the text i.e converting text into numerical data
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit(x)
x = vectorizer.transform(x)
#splitting the dataset into training and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size = 0.2, stratify = y, random_state = 2)
#training the model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train,y_train)
#accuracy score of training data
from sklearn.metrics import accuracy_score
x_train_prediction = model.predict(x_train)
training_accuracy = accuracy_score(x_train_prediction, y_train)
print('Accuracy score of the training data : ', training_accuracy)
#accuracy score of test data
from sklearn.metrics import accuracy_score
x_test_prediction = model.predict(x_test)
test_accuracy = accuracy_score(x_test_prediction, y_test)
print('Accuracy score of the test data : ', test_accuracy)   #accuracy score of the test data came out to be 97.9%


