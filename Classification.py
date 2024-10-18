################################# FUNZIONI CLASSIFICAZIONE ##################################
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from Evaluator import evaluation_of_model


# function definitions to train and classify with different algorithms
from File_controller import saveObject
import time
def classification(model, x_train, y_train, x_test):
    t = time.time()

    if model == 'SVM':
        classifier = LinearSVC()
    elif model == 'MNB':
        classifier = MultinomialNB()
    elif model == 'KNN':
        classifier = KNeighborsClassifier(n_neighbors = 5)
    elif model == 'RF':
        classifier = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=None)
    elif model == 'Adab':
        classifier = AdaBoostClassifier(n_estimators=100)
    elif model == 'DT':
        classifier = DecisionTreeClassifier(random_state=0)
    elif model == 'BNB':
        classifier = BernoulliNB(force_alpha = True)
    else:
        return -1

    saveObject(model, classifier)

    classifier.fit(x_train, y_train)
    predicted = classifier.predict(x_test)

    print(f'Time Taken: {round(time.time() - t)} seconds')

    return predicted


def TryAllModel(x_train, y_train, x_test, y_test):
    predicted = classification('SVM', x_train, y_train, x_test)
    print("SVM: \n")
    evaluation_of_model(y_test, predicted)

    predicted = classification('MNB', x_train, y_train, x_test)
    print("MNB: \n")
    evaluation_of_model(y_test, predicted)

    predicted = classification('DT', x_train, y_train, x_test)
    print("DT: \n")
    evaluation_of_model(y_test, predicted)

    predicted = classification('BNB', x_train, y_train, x_test)
    print("BNB: \n")
    evaluation_of_model(y_test, predicted)

    predicted = classification('KNN', x_train, y_train, x_test)
    print("KNN: \n")
    evaluation_of_model(y_test, predicted)

    predicted = classification('RF', x_train, y_train, x_test)
    print("RF: \n")
    evaluation_of_model(y_test, predicted)

    predicted = classification('Adab', x_train, y_train, x_test)
    print("ADAB: \n")
    evaluation_of_model(y_test, predicted)

from Tfidf_builder import obtain_data
train_tfidf, test_tfidf, y_train, y_test,count_vect, tfidf_transformer = obtain_data()

TryAllModel(train_tfidf, y_train, test_tfidf, y_test)

'''
from sklearn.pipeline import Pipeline

classifiers = [
    RandomForestClassifier(),
    AdaBoostClassifier(),
    KNeighborsClassifier(),
    MultinomialNB(),
    BernoulliNB(),
    LinearSVC(),
    DecisionTreeClassifier(random_state=0)
]

# Pipeline Classifier
pipelines = []

for classifier in classifiers:   
    pipelines.append(Pipeline([
        ('clf', classifier)
    ]))

from sklearn.model_selection import cross_val_predict
import time
import warnings
warnings.filterwarnings('ignore')

train_tfidf, test_tfidf, y_train, y_test, count_vect, tfidf_transformer = obtain_data()

for pipe in pipelines:
    t0 = time.time()
    predicted = cross_val_predict(pipe, train_tfidf, y_train, cv=10)
    t1 = time.time()
    t = (t1-t0)/10
    
    print("qui6")
    print("\n Evaluation: ", pipe['clf'], " \tTraining time: ", t)
    print(classification_report(train_tfidf, predicted, target_names=["negative", "neutral", "positive"]))
'''