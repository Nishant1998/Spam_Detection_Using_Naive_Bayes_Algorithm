#!/usr/bin/env python

import sqlite3,string,nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os

class Preprocessing:
    def split_data():
        #connect data base
        conn = sqlite3.connect('data.db')

        x, y, b = [], [], []

        #read data
        a = conn.execute("select data,spam from sdata;")
        b = []
        for i in a:
            b.append([i[0],i[1]])
        for i in b:
            x.append([i[0]])
        for i in b:
            y.append([i[1]])
        return(x,y)

    # remove puntuation from message
    def rmov_pun(message):

        # making list of all puntuation marks list.
        puntuation = set(string.punctuation)

        # removing puntuation marks
        mess_pless = ''.join(ch for ch in message if ch not in puntuation)

        # convert to lower case
        mess_pless = mess_pless.lower()

        return mess_pless.split()

    # removing stop words
    def rmov_stop_words(mess_pless):

        # making list of all stopwords
        stop_words = stopwords.words('english')

        # removing stop words
        mess_sless = [w for w in mess_pless if not w in stop_words]
        return mess_sless

    # stem message
    def stem_message(mess_sless):
        # stem the similar meaning words.
        porter = PorterStemmer()
        stemmed = [porter.stem(word) for word in mess_sless]
        return stemmed

    def getdata():
        # connect data base
        conn = sqlite3.connect('data.db')

        # path of data store
        direc_ham = "email_data/ham/"
        direc_spam = "email_data/spam/"

        # make list of file name store in above path
        h_files = os.listdir(direc_ham)
        s_files = os.listdir(direc_spam)

        # making path of each file
        h_emails = [direc_ham + h for h in h_files]
        s_emails = [direc_spam + s for s in s_files]

        # processing ham mails
        for i in h_emails:
            f = open(i,'r',encoding='utf-8',errors='ignore')
            raw_mess = f.read()
            mess_sless  = Preprocessing.stem_message(Preprocessing.rmov_stop_words(Preprocessing.rmov_pun(raw_mess)))
            insert_mess = ' '.join(mess_sless)

            # insert processed mails in data base with lable 0
            sql = "insert into sdata values('{}',0);".format(insert_mess)
            conn.execute(sql)
            conn.commit()

        # processing spam mails
        for i in s_emails:
            f = open(i, 'r', encoding='utf-8',errors='ignore')
            raw_mess = f.read()
            mess_sless  = Preprocessing.stem_message(Preprocessing.rmov_stop_words(Preprocessing.rmov_pun(raw_mess)))
            insert_mess = ' '.join(mess_sless)

            # insert processed mails in data base with lable 1
            sql = "insert into sdata values('{}',1);".format(insert_mess)
            conn.execute(sql)
            conn.commit()


