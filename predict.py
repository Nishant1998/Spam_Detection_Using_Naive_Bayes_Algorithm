#!/usr/bin/env python
import pickle
from sklearn.naive_bayes import *
#from collections import Counter


def main():

    # reading classifier object from file
    with open('spam_classifier.mdl', 'rb') as scla:
        clf = pickle.load(scla)

    # getting user input.
    User_input = input("enter message:")

    # import Preprocessing class
    from preprocess import Preprocessing
    prepro = Preprocessing

    # processing user input
    inp = prepro.stem_message(prepro.rmov_stop_words(prepro.rmov_pun(User_input)))

    # get dictionary of most common 3000 words
    with open('dictionary_classifier.mdl', 'rb') as dic_c:
        wd = pickle.load(dic_c)
    word_dictionary = wd.most_common(3000)

    # making feature set of user input.
    features = make_feature(inp,word_dictionary)

    # making prediction.
    predictions = clf.predict(features)

    # print prediction
    if predictions[0] == 0:
        print("\nMessage is NOT SPAM")
    else:
        print("\nMessage is SPAM")

    input()


# making feature set
def make_feature(User_input,word_dictionary):
    features = []
    data= []
    words = User_input

    for entry in word_dictionary:
        data.append(words.count(entry[0]))
    features.append(data)
    return(features)


if __name__ == "__main__" : main()