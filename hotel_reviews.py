import streamlit as st 
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer,WordNetLemmatizer
from nltk import word_tokenize
import re
import pandas as pd
import contractions
import inflect
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from tabulate import tabulate
import joblib
from nltk.tokenize import RegexpTokenizer


pickle_in = open("model_1.pkl", 'rb') 
model_1 = joblib.load(pickle_in)

pickle_in_tdidf = open("model_1_tfidf.pkl", 'rb') 
model_1_tfidf = joblib.load(pickle_in_tdidf)

def main():
    st.title('Hotel Rating')
    st.subheader("Analyze hotel reviews as positive or negative")

if __name__ == '__main__':
	main()
    
    
stop_words = stopwords.words('english') # remove stop words

def get_percentage(num):
    return "{:.2f}%".format(num*100)

def number_to_text(data):
    temp_str = data.split()
    string = []
    for i in temp_str:

    # if the word is digit, converted to
    # word else the sequence continues

        if i.isdigit():
            temp = inflect.engine().number_to_words(i)
            string.append(temp)
        else:
            string.append(i)
    outputStr = " ".join(string)
    return outputStr

ps = PorterStemmer()
def stem_text(data):
    tokens = word_tokenize(data)
    stemmed_tokens = [ps.stem(word) for word in tokens if word not in (stop_words)]
    return " ".join(stemmed_tokens)

def lemmatise(sentence):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(sentence)
    lemmatizer = WordNetLemmatizer()
    lemma_words=[lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(lemma_words)

def preprocess(sentence):
    sentence=str(sentence)
    sentence = sentence.lower()
    sentence=sentence.replace('{html}',"") 
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    rem_url=re.sub(r'http\S+', '',cleantext)
    rem_num = re.sub('[0-9]+', '', rem_url)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)  
    filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
    lemmatizer= WordNetLemmatizer()
    lemma_words=[lemmatizer.lemmatize(w) for w in filtered_words]
    return " ".join(lemma_words)


ReviewText = st.text_area('Enter the hotel review text', '')

if st.button("Analyze"):
    cleanReviewText = preprocess(ReviewText)
    #st.write(cleanReviewText)    
    textTfIDF    = model_1_tfidf.transform([cleanReviewText])
    predictedVal = model_1.predict(textTfIDF)
    predictedVal = predictedVal[0]
    #st.write(type(predictedVal))
    
    st.markdown("""
        <style>
        .big-font {
            font-size:20px !important;
        }
        
        </style>
        """, unsafe_allow_html=True)
    
    if predictedVal=="POSITIVE" :
        print("The review is Positive 😀") 
    elif predictedVal=="NEGATIVE" :
        print("The review is Negative 😔")

      
    st.markdown("<p class='big-font'>{}</p>".format(predictedVal),unsafe_allow_html=True)
    

        
    prdictionDist = model_1._predict_proba_lr(textTfIDF)
    
    
    dfRes = pd.DataFrame(columns=['Negative', 'Positive'])
    dfRes.loc[1, 'Negative'] = get_percentage(prdictionDist[0][0])
    dfRes.loc[1, 'Positive'] = get_percentage(prdictionDist[0][1])
    
        # CSS to inject contained in a string
    hide_dataframe_row_index = """
                <style>
                .row_heading.level0 {display:none}
                .blank {display:none}
                </style>
                """
        # Inject CSS with Markdown
   
    st.markdown(hide_dataframe_row_index, unsafe_allow_html=True)
    
    st.dataframe(dfRes)
