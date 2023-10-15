import subprocess
subprocess.run(["pip", "install", "nltk"])


import streamlit as st
import pickle



from nltk.corpus import stopwords
import nltk
import string
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


def transform_text(text):

    # lower casing all the words
    text = text.lower()

    # making breaking the word and storing these words into a list
    text = nltk.word_tokenize(text)

    #keeping only digit and alphabates
    y=[]
    for i in text:          
        if i.isalnum():     
            y.append(i)

    text = y[:]
    y.clear()

    #Removing Stopwords and Punctuations
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    
    text = y[:]
    y.clear()

    #Stemmig all the word( bringing them to root words)
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)





tfidf = pickle.load( open('tfidf_vectorizer.pkl', 'rb'))
model = pickle.load( open('mnb_model.pkl', 'rb'))

st.title('Email/SMS Spam Classifier')

input_sms = st.text_input('Enter the message')

if st.button("Predict"):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)

    #2 vetorize

    vector_input_sms = tfidf.transform([transformed_sms])
    #3 pridict
    result = model.predict(vector_input_sms)


    #4 Display
    if result ==1:
        st.header("Spam")
    else:
        st.header("Not Spam")
