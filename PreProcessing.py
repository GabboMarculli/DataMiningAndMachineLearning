########################## FUNZIONI PER PULIRE LE FRASI #############################

import string
import re

# remove special characters
reg_exp_at = '(@[A-Za-z0-9]+)|(#[A-Za-z0-9]+)'
reg_exp_retweet = 'RT[/s]+'
reg_exp_http = 'http\S+'
reg_exp_hyperlinks = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
reg_exp_number = '[0-9]+'
reg_exp_alphabet = "[^a-zA-Z0-9]"
sequencePattern = r"(.)\1\1+"
seqReplacePattern = r"\1\1"


def tweet_clean(tweet):
    # Trasformo le maiuscole in minuscole
    tweet = str(tweet).lower()

    # remove @mentions or #hashtag
    rm_mention = re.sub(reg_exp_at, '', tweet)

    # remove retweet
    rm_rt = re.sub(reg_exp_retweet, '', rm_mention)

    # remove hyperlinks
    rm_links = re.sub(reg_exp_http, '', rm_rt)
    rm_links = re.sub(reg_exp_hyperlinks, '', rm_links)

    # remove numbers
    rm_nums = re.sub(reg_exp_number, '', rm_links)

    # replace characters except Digits and Alphabets with a space
    rm_nums = re.sub(reg_exp_alphabet, " ", rm_nums)

    # Replace 3 or more consecutive letters by 2 letter : 'heyyyyyyyyyy' become 'heyy'
    rm_nums = re.sub(sequencePattern, seqReplacePattern, rm_nums)

    # remove punctuations
    rm_p = [char for char in rm_nums
            if char not in string.punctuation]
    rm_pun = ''.join(rm_p)

    # remove multiple spaces
    rm_new = " ".join(rm_pun.split())
    cleaned = rm_new

    return cleaned


# rimuovi stopWords
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.stem import WordNetLemmatizer
nltk.download("wordnet")
nltk.download('omw-1.4')

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('words')

stop_words = set(stopwords.words('english'))

lemmatizer = WordNetLemmatizer()
def filterStopWords(row):
    word_tokens = word_tokenize(row)
    # delete stop words
    filtered_sentence = [w for w in word_tokens
                         if not w.lower() in stop_words]
    # lemmatize words
    lemmatized = [lemmatizer.lemmatize(w) for w in filtered_sentence]
    filtered_sentence = TreebankWordDetokenizer().detokenize(lemmatized)

    return filtered_sentence

import pandas as pd
import io

# train = pd.read_csv("C:\\Users\\gabri\\Desktop\\emojifi\\DATASET-emoji\\Train.csv")

# ------ PULISCO I DATI DI TRAIN
#train['clean_text'].dropna(how='any')
#train['category'].dropna(how='any')

#train['clean_text'] = train['clean_text'].apply(tweet_clean)
#print(train['clean_text'].head)
#train['clean_text'] = train['clean_text'].apply(filterStopWords)
#print(Xìtrain['clean_text'].head)

# save cleaned data
#train.to_csv("Balance_twitter_data.csv", index=False)

