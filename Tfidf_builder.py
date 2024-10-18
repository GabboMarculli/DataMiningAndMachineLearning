import pandas as pd 
import io
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split

def obtain_data():
    train = pd.read_csv("Balance_twitter_data.csv")
    train = train.dropna(axis=0)

    X = train['clean_text']
    y = train["category"]

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)
    
    count_vect = CountVectorizer()

    train_counts = count_vect.fit_transform(X_train)
    test_counts = count_vect.transform(X_test)

    tfidf_transformer = TfidfTransformer()
    train_tfidf = tfidf_transformer.fit_transform(train_counts)
    test_tfidf = tfidf_transformer.transform(test_counts)

    return train_tfidf, test_tfidf, y_train, y_test, count_vect, tfidf_transformer