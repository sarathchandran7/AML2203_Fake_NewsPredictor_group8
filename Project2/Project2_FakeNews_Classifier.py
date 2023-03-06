# importing essential packages
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Loading the dataset into a pandas dataframe
ds = pd.read_csv("train.csv")
print(ds.head())

# checking the missing values in our dataset and replacing them with an empty string
print(ds.isnull().sum())
ds = ds.fillna('')
print(ds.head())

# merging the author and title column as content
ds['content'] = ds['author'] + ' ' + ds['title']
print(ds['content'].head())

# Performing stemming on the content
port_stem = PorterStemmer()


def stemming(content):
    stemmed_cont = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_cont = stemmed_cont.lower()
    stemmed_cont = stemmed_cont.split()
    stemmed_cont = [port_stem.stem(word) for word in stemmed_cont if word not in stopwords.words('english')]
    stemmed_cont = " ".join(stemmed_cont)
    return stemmed_cont


ds['content'] = ds['content'].apply(stemming)
print(ds['content'])

# storing the data in x and y from training the model. we will be using content and labels to train our model
x = ds['content']
y = ds['label']
print(x)
print(y)

# Performing feature vectorization to convert textual data to numerical data
vectorizer = TfidfVectorizer()
vectorizer.fit(x)
x = vectorizer.transform(x)

# creating the training and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)

# Implementing the Logistic Regression model
model = LogisticRegression()
model.fit(x_train, y_train)

# Finding the accuracy score of our model on training data
x_train_predict = model.predict(x_train)
accuracy = accuracy_score(x_train_predict, y_train)
print(accuracy, "is the accuracy score of the training data")

# Finding the accuracy score of our model on test data and making prediction on test data sing our model
x_test_predict = model.predict(x_test)
accuracy = accuracy_score(x_test_predict, y_test)
print(accuracy, "is the accuracy score of the test data")


# predicting a news
news = x_test[-1]
prediction = model.predict(news)

if prediction == 1:
    print("Fake news")
else:
    print("Real news")













