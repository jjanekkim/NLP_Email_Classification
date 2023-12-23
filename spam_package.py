# python version 3.11.3

import pandas as pd
import re # Python module to work with regular expressions ('regex')

import nltk
nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, roc_auc_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

pd.set_option('display.max_colwidth', None)

#----------------------------------------------------------------------------------------

def read_data(path):
    """This function reads a CSV file and returns the resulting dataframe."""
    df = pd.read_csv(path, encoding='latin-1')
    df.dropna(axis=1, how='any', inplace=True)
    df.columns = ['label', 'text']
    return df

#----------------------------------------------------------------------------------------

def clean_text_data(data):
    """This function cleans the email text through a series of steps:
    - Tokenizing the text into words.
    - Removing punctuations using Python's re module.
    - Eliminating stop words.
    - Removing single-letter words.
    - Removing words containing digits."""
    
    word_token = word_tokenize(data) # tokenize the text and returns the list of words
    stop_words = stopwords.words('english') # list of stop words
    clean_text = [re.sub(r'[^\w\s]', '', word) for word in word_token] # remove punctuations
    clean_text = [word for word in clean_text if word.lower() not in stop_words] # remove stop words
    clean_text = [word for word in clean_text if len(word)>2] # remove one letter from text
    clean_text = [word for word in clean_text if not any(char.isdigit() for char in word)] # remove any words with digit
    clean_text = ' '.join(clean_text)
    return clean_text

#----------------------------------------------------------------------------------------

def lemmatize_words(data):
    """This function executes lemmatization on the preprocessed text, converting words to their base or dictionary form."""
    
    lm = WordNetLemmatizer()
    lem_text = lm.lemmatize(data)
    return lem_text

#----------------------------------------------------------------------------------------

def pos_tag_text(data):
    """This function conducts part-of-speech tagging on the text, labeling each word with its corresponding part of speech (such as noun, verb, adjective, etc.)."""

    token_text = word_tokenize(data)
    tag_words = nltk.pos_tag(token_text)
    chosen_words = list(filter(lambda w: w[1]=='NNP' or w[1]=='NN' or w[1]=='JJ' or w[1]=="VB", tag_words))
    pos_text = ' '.join([word[0] for word in chosen_words])
    return pos_text

#----------------------------------------------------------------------------------------

def apply_data_cleaning(df):
    """This function executes an extensive cleaning process that involves:
    - Removing stop words
    - Stripping punctuations
    - Eliminating digits
    - Conducting lemmatization
    - Performing part-of-speech tagging
    """
    df['clean_txt'] = df['text'].apply(clean_text_data)
    df['lemma_txt'] = df['clean_txt'].apply(lemmatize_words)
    df['pos_txt'] = df['lemma_txt'].apply(pos_tag_text)
    return df

#----------------------------------------------------------------------------------------

def text_length(df):
    """This function creates additional columns in the dataset, computing the total values derived from the cleaned text, lemmatized text, and part-of-speech-tagged text."""
    df['txt_length'] = df['text'].apply(len)
    df['clean_len'] = df['clean_txt'].apply(len)
    df['lemma_len'] = df['lemma_txt'].apply(len)
    df['pos_len'] = df['pos_txt'].apply(len)
    return df

#----------------------------------------------------------------------------------------

def drop_empty_data(df):
    """This function removes rows containing empty text data after the text cleaning process."""
    empty_content = df[df['pos_len']==0]
    df.drop(index=empty_content.index, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

#----------------------------------------------------------------------------------------

def transform_target(df):
    """This function converts the target feature into binary format."""
    df['target'] = df['label'].map(lambda x: 1 if x=='spam' else 0)
    return df

#----------------------------------------------------------------------------------------

def countvect_data(df):
    """This function applies the Count Vectorization technique to transform the data."""
    count_vect = CountVectorizer(lowercase=True, max_features=1000) #max_feature at 1000 to downsize the data for faster training
    cv = count_vect.fit_transform(df['pos_txt'])
    cv_df = pd.DataFrame(cv.todense(), columns=count_vect.get_feature_names_out())
    return cv_df

#----------------------------------------------------------------------------------------

def tf_idf_data(df):
    """This function applies the TF-IDF (Term Frequency-Inverse Document Frequency) technique to transform the data."""
    tfidf_vect = TfidfVectorizer(lowercase=True, max_features=1000)
    tfidf = tfidf_vect.fit_transform(df['pos_txt'])
    tfidf_df = pd.DataFrame(tfidf.todense(), columns=tfidf_vect.get_feature_names_out())
    return tfidf_df

#----------------------------------------------------------------------------------------

def merge_dataset(df, trans_df):
    """This function combines/merges the dataset, potentially by concatenating or joining multiple data sources together."""
    df_feature = df[['txt_length','target']]
    df_final = trans_df.join(df_feature)
    return df_final

#----------------------------------------------------------------------------------------

def split_data(df):
    """This function divides the dataset into two separate dataframes: one containing dependent variables and the other containing independent variables."""
    x = df.drop(columns='target')
    y = df['target']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    return x_train, x_test, y_train, y_test

#----------------------------------------------------------------------------------------

def train_model(x_train, y_train):
    """This function is responsible for training both the XGBClassifier and LogisticRegression models."""
    xgb = XGBClassifier(random_state=0, n_estimators=1000, learning_rate=0.1)
    xgb_model = xgb.fit(X=x_train, y=y_train)

    logi = LogisticRegression(max_iter=1000)
    logi_model = logi.fit(X=x_train, y=y_train)

    return xgb_model, logi_model

#----------------------------------------------------------------------------------------

def make_prediction(test_df, model):
    """This function executes predictions based on the provided data or model."""
    prediction = model.predict(test_df)
    return prediction

#----------------------------------------------------------------------------------------

def model_evaluation(true_value, prediction):
    """This function generates a metrics dataframe containing precision score, recall score, and AUC score."""
    metrics_df = pd.DataFrame({
    'Logistic': {'Precision':precision_score(true_value, prediction),
                 'Recall':recall_score(true_value, prediction),
                 'ROC_AUC':roc_auc_score(true_value, prediction)}})
    return metrics_df