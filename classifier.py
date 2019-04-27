#!/usr/bin/env python
# coding: utf-8

from collections import Counter
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score
import pickle

def main():

    #import Preprocessing class from preprocess
    from preprocess import Preprocessing
    prepro = Preprocessing

    #get data and lable from Preprocessing class
    X,Y = prepro.split_data()

    #convert data into features list
    feature_set,lable = make_dataset(X,Y)

    #split data into training and testing data
    X_train,X_test,Y_train,Y_test = tts(feature_set,lable, test_size=0.2)

    #making classifier object using Multinomial Naive Bayes
    classifier = MNB()

    #training the classifier with Trainining data feature and lables
    classifier.fit(X_train,Y_train)

    #testing the classifier
    predictions = classifier.predict(X_test)

    #calculate accuracy by comparing prediction make test data's lable
    print("Accuracy of Classifier :")
    print(accuracy_score(Y_test,predictions))

    #saving classifier in a file.
    with open('spam_classifier.mdl', 'wb') as scla:
        pickle.dump(classifier, scla)


def make_word_dictionary(X):
    word_dictionary = {}
    x = []

    #convert data into list
    for i in X:
        j=' '.join(i)
        a=j.split()
        x.append(a)

    #add count of word from data to dictionary
    for i in x:
        word_dictionary = Counter(word_dictionary) + Counter(i)

    #save dictionary
    with open('dictionary_classifier.mdl', 'wb') as dic_c:
        pickle.dump(word_dictionary, dic_c)

    return(word_dictionary)

def make_dataset(X,Y):

    #make dictionary
    wd = make_word_dictionary(X)
    #getting common 3000 words
    word_dictionary = wd.most_common(3000)


    feature_set = []
    lable = []

    #making feature set
    for (mess,spam) in zip(X,Y):
        data= []
        mess=' '.join(mess)
        words = mess.split()

        for entry in word_dictionary:
            data.append(words.count(entry[0]))
        feature_set.append(data)
        lable.append(spam)

    return(feature_set,lable)


if __name__ == "__main__": main()
