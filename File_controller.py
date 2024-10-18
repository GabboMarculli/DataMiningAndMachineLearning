import pickle

def saveObject(model, classifier):
    file = open(model + '.pickle' ,'wb')
    pickle.dump(classifier, file)
    file.close()
    
def loadObject(fileName):
    try:
        with open("\\" + fileName + '.pickle', 'rb') as file:
            obj = pickle.load(file)
            return obj
    except IOError:
        return None