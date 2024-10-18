from sklearn.svm import LinearSVC
import numpy as np
import pandas as pd
import io

from Tfidf_builder import obtain_data
train_tfidf, test_tfidf, y_train, y_test, count_vect, tfidf_transformer = obtain_data()

from File_controller import loadObject

model = loadObject('SVM')
model.fit(train_tfidf, y_train)
predicted = model.predict(test_tfidf)

import tkinter as tk
import tkinter.font as font

pos  = ['🔅', '🌼', '🎁', '❤', '🎉', '💙','😂']
neg  = ['👿', '☹', '😕', '😩', '😒', '🔋', '🍥']
neut = ['☯', '📒', '♚', '❊', '❁', '⇩' , '۩ ']

def convert_polarity(e):
    if e == [1.]:
        return pos
    if e == [-1.]:
        return neg
    if e == [0.]:
        return neut

def set_emoji(e):
     emoji.delete(0, tk.END)
     emoji.insert(0,convert_polarity(e))

def clear_all():
    Sentence.delete('1.0', 'end-1c')
    emoji.delete(0, tk.END)

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from PreProcessing import tweet_clean, filterStopWords
def get_sentence():
    sen = Sentence.get("1.0",'end-1c')

    sen = tweet_clean(sen)
    sen = filterStopWords(sen)
    sen = [sen]
    
    train_counts = count_vect.transform(sen)
    train_tfidf = tfidf_transformer.transform(train_counts)
    y_pred = model.predict(train_tfidf)
    
    set_emoji(y_pred)

root = tk.Tk()
root.geometry("500x400")
root.title("Emojifi")

# Title
myFont = font.Font(size=13)
title = tk.Label(root, text="Insert a sentence and receive an emoji !", fg='#0052cc', font=myFont)
title.place(x=90, y=1)

# Defining the first row
lblfrstrow = tk.Label(root, text="Sentence :", )
lblfrstrow.place(x=50, y=80)

Sentence = tk.Text(root, bg = "light yellow")
Sentence.place(x=150, y=80, width = 300, height=100)

lblsecrow = tk.Label(root, text="Emoji :")
lblsecrow.place(x=50, y=220)

emoji = tk.Entry(root, width=60)
emoji.place(x=150, y=220, width=300)

submitbtn = tk.Button(root, text="Submit",
                    bg='blue', command=get_sentence)
submitbtn.place(x=250, y=275, width=55)

clearbtn = tk.Button(root, text="Clear",
                    bg='blue', command=clear_all)
clearbtn.place(x=180, y=275, width=55)

root.mainloop()
