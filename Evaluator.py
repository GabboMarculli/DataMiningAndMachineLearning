################################## FUNZIONI PER MISURARE QUALITA' #####################################

# spezzo alla quarta cifra decimale
def truncate_at_fourth(value):
    return float(f'{value:.4f}')

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def evaluation_of_model(y_test, y_pred):
    print(classification_report(y_test, y_pred))
    
    # Compute and plot the Confusion matrix
    cf_matrix = confusion_matrix(y_test, y_pred)

    sns.heatmap(cf_matrix, annot=True)

    plt.xlabel("Predicted values", fontdict = {'size':14}, labelpad = 10)
    plt.ylabel("Actual values"   , fontdict = {'size':14}, labelpad = 10)
    plt.title ("Confusion Matrix", fontdict = {'size':18}, pad = 20)
    plt.show()

model = ['SVM',
         'MNB',
         'KNN',
         'RF',
         'ADAB']


def from_initials_to_model(model):
    ris = []

    if model == 'SVM':
        ris = "Support Vector Machine"
    elif model == 'MNB':
        ris = "Multinomial Naive Bayes Classifier"
    elif model == 'KNN':
        ris = "K-Neighbors Classifier"
    elif model == 'RF':
        ris = "Random Forest Classifier"
    elif model == 'Adab':
        ris = "AdaBoostClassifier"

    return ris

from sklearn.metrics import accuracy_score
def best_accuracy(actual, predicted):
    maximum = 0
    max_index = 0

    for i in range(5):
        aux = accuracy_score(actual, predicted)
        if maximum < aux:
            maximum = aux
            max_index = i
    return [maximum, max_index]

