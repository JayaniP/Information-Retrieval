#--- Install Required Packages ---#
pip install elasticsearch
pip install nltk
pip install pandas

#--- Importing libraries ---#
import nltk
import re
import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

#--- Read csv file into a pandas dataframe ---#
df = pd.read_csv(r'D:\Information_Retrival\covid.csv')
print("Schema :", df.dtypes)
print("No of docs and columns :", df.shape)

#--- Summary of the DataFrame ---#
df_summary=df.isnull().sum().reset_index()
df_summary=df_summary.rename(columns={0:'null_rows'})
df_summary['total_rows']=len(df)
print(df_summary)

#--- Check missing values in dataframe ---#
df.isnull().sum()

#--- Drop missing values from 'abstract' and 'authors' column: ---#
d = df.dropna(subset=["abstract","authors"])
d.head()

#### Sentence Splitting, Tokenization and Normalization #### 

#### Splitting ####
# Consider two columns for Sentences splitting
d['title'],  d['abstract']

#--- Import sent_tokenize package from NLTK Library ---#
from nltk.tokenize import sent_tokenize

#--- Split Sentences for column 'title' ---#
for sentences in d['title']:
    all_sent = sent_tokenize(sentences)
    print(all_sent)
   
#--- Split Sentences for column 'abstract' ---#
for sentences in d['abstract']:
    all_sent = sent_tokenize(sentences)
    print(all_sent)

# Consider four columns for Word Splitting
d['title'], d['abstract'], d['authors'], d['journal']

#--- Import word_tokenize package from NLTK Library ---#
from nltk.tokenize import word_tokenize 
 
#--- Split Words for column 'title' ---# 
for words in d['title']:
    all_words = word_tokenize(words)
    print(all_words)

#--- Split Words for column 'abstract' ---# 
for words in d['abstract']:
    all_word = word_tokenize(words)
    print(all_word)

#--- Split Words for column 'authors' ---# 
for words in d['authors']:
    all_words = word_tokenize(words)
    print(all_words)

#--- Split Words for column 'journal' ---#
for words in d['journal']:
    all_words = word_tokenize(words)
    print(all_words)

#### Tokenization #####

#--- Using Regular Expression ---#
#--- Consider two columns for procesing ---#
d['title'],  d['abstract']

#--- Import RegexpTokenizer from NLTK Library ---#
from nltk.tokenize import RegexpTokenizer 
tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+\s*\n\s*\n\s*\w+|[^\w\s]+')

#--- Using Regular Expression form 'title' Column ---#
for reg in d['title']:
    all_reg = tokenizer.tokenize(reg)
    print(all_reg)

#--- Using RegularExpression form 'abstract' Column ---# 
for reg in d['abstract']:
    all_abstract_reg = tokenizer.tokenize(reg)
    print(all_abstract_reg)

#---Seperates the punctuation ---#
#--- Download punkt package --#
nltk.download('punkt')

#--- Import WordPunctTokenizer from NLTK Library ---#
from nltk.tokenize import WordPunctTokenizer 
tokenizer = WordPunctTokenizer() 

#---Remove Punctuation form 'title' Column  ---#
for pun in d['title']:
    all_punctuation = tokenizer.tokenize(pun)
    print(all_punctuation)

#---Remove Punctuation form 'abstract' Column  ---#
for pun in d['abstract']:
    all_abstract_pun = tokenizer.tokenize(pun)
    print(all_abstract_pun)

#---Remove Punctuation form 'authors' Column  ---#
for pun in d['authors']:
    all_authors_pun = tokenizer.tokenize(pun)
    print(all_authors_pun)

#### Normalization ####
#---  LowerCase the Data ---#
#--- Consider two columns to convert the data ---#
d['abstract'], d['journal']

#--- For 'abstract' column convert the data in lowercase ---#
for sentences in d['abstract']:
    lower_abstract = sentences.lower()
    print(lower_abstract)

#--- For 'journal' column convert the data in lowercase ---#
for sentences in d['journal']:
    lower_journal = sentences.lower()
    print(lower_journal)

#--- Removing stop words ---#
#--- Download stopwords package ---#
nltk.download("stopwords")

# Consider three columns for processing ---#
d['title'],  d['abstract'], d['authors']

#--- Comman words from corpus 'english' ---#
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
print(stop_words)

#--- Remove StopWords form 'title' Column ---#
for stops in d['title']:
    word_tokens = word_tokenize(stops)
removing_stopwords = [word for word in word_tokens if word not in stop_words]
print (removing_stopwords)

#--- Remove StopWords form 'abstract' Column ---#
for stops in d['abstract']:
    word_tokens = word_tokenize(stops)
removing_stopwords = [word for word in word_tokens if word not in stop_words]
print (removing_stopwords)

#--- Remove StopWords form 'authors' Column ---#
for stops in d['authors']:
    word_tokens = word_tokenize(stops)
removing_stopwords = [word for word in word_tokens if word not in stop_words]
print (removing_stopwords)

#### Stemming or Morphological Analysis ####

#--- Import Stemming Libraries ---#
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
porter_stemmer = PorterStemmer()

#### Stemming with words ####
#--- Stemming for 'title' column ---#
words = d['title']
for w in words:
    print(w, " : ",porter_stemmer.stem(w))

#--- Word Stemming for 'abstract' column ---#
words = d['abstract']
for w in words:
    print(w, " : ",porter_stemmer.stem(w))

#### Word Stemming using Sentences ####
#--- Download the wordnet package ---#
nltk.download('wordnet')

#---Import WordNetLemmatizer Library ---#
from nltk.stem.wordnet import WordNetLemmatizer
wnl = WordNetLemmatizer()

#--- Sentence Stemming for 'title' column ---#
for sentence in d['title']:
    words = word_tokenize(sentence)
    for w in words:
        print(w, " : ", porter_stemmer.stem(w),wnl.lemmatize(w))

#--- Sentence Stemming for 'abstract' column ---#
for sentence in d['abstract']:
    words = word_tokenize(sentence)
    for w in words:
          print(w, " : ", porter_stemmer.stem(w),wnl.lemmatize(w))

#### Using Lemmatize ####

#--- Word Lemmatize for 'title' column ---#
words = d['title']
for w in words:
    print(w, " : ",wnl.lemmatize(w))

#--- Word Lemmatize for 'abstract' column ---#
words = d['abstract']
for w in words:
    print(w, " : ",wnl.lemmatize(w))

#--- Sentence Lemmatize for 'title' column ---#
for sentence in d['title']:
    input_str = word_tokenize(sentence)
    for word in input_str:
        print(wnl.lemmatize(word))

#--- Sentence Lemmatize for 'abstract' column ---#
for sentence in d['abstract']:
    input_str = word_tokenize(sentence)
    for word in input_str:
        print(wnl.lemmatize(word))


#### Preproceesing on covid datasets ####

def pre_process(text):
    text = text.lower()                     #lowercase
    text = re.sub("","",text)               #remove tags
    text = re.sub("(\\d\\W)+"," ",text)     #remove special characters and digits
    token_words = nltk.word_tokenize(text) # Tokenization
    words = [w for w in token_words if not w in stop_words] # Removing Stop Words
    for word in words:
        text.join(wnl.lemmatize(word))
    return text

d["text"] = d["title"] + d["abstract"] + d['authors'] + d['journal']

# Pre proceesing on 'title' column
d["txt1"] = d["title"]
d["title1"] = d["txt1"].apply(lambda x:pre_process(x))

# Pre proceesing on 'abstract' column
d["txt2"] = d["abstract"]
d["abstract1"] = d["txt2"].apply(lambda x:pre_process(x))

# Pre proceesing on 'authors' column
d["txt3"] = d["authors"]
d["authors1"] = d["txt3"].apply(lambda x:pre_process(x))

# Pre proceesing on 'journal' column
d["txt4"] = d["journal"]
d["journal1"] = d["txt4"].apply(lambda x:pre_process(x))

docs=d['text'].tolist()
print(docs[0])

#--- Using CountVectorizer to calculate vocabulary ---#
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_df=0.85,stop_words="english")
word_count_vector=cv.fit_transform(docs)
list(cv.vocabulary_.keys())[:10]

#--- Using TF-IDF to search the keywords ---#
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_matrix = tfidf_transformer.fit(word_count_vector)
feature_names = cv.get_feature_names() 
vector = docs[0]
tf_idf_vector=tfidf_transformer.transform(cv.transform([vector]))

def sort(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""
    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []
    
    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    #create a tuples of feature,score
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    return results

#sort the tf-idf vectors by descending order of scores
sorted_items=sort(tf_idf_vector.tocoo())

#extract only the top n; n here is 1000
keywords=extract(feature_names,sorted_items,1000)
print(vector)

#--- Caluclate the weightage of words ---#
for k in keywords :
    print(k,keywords[k])


#### Load the data into Elastic Search ####
# ====== Connection ====== #

# Connection to ElasticSearch
es = Elasticsearch( ["localhost:9200"],
                    sniff_on_start=True,
                    sniff_on_connection_fail=True,
                    sniffer_timeout=60
                   )

# Simple index creation with no particular mapping
es.indices.create(index='covid',body={})

# ====== Inserting Documents ====== #

# Creating a simple Pandas DataFrame
dataframe = pd.DataFrame(data = {'title' : d["title1"], 'abstract': d["abstract1"], 'authors':d['authors1'], 'journal': d['journal1'] })
 
# Bulk inserting documents. Each row in the DataFrame will be a document in ElasticSearch
documents = dataframe.to_dict(orient='records')
bulk(es, documents, index='covid',doc_type='covid_data', raise_on_error=True)

#--- Refresh the index ---#
es.indices.refresh(index="covid")

# ====== Searching Documents ====== #

#--- Retrieving all documents in index (no query given) ---#
documents = es.search(index='covid',body={})['hits']['hits']
df = pd.DataFrame(documents)
print(df)

#--- check data is in there, and structure in there ---#
documents1 = es.search(body={"query": {"match_all": {}}}, index = 'covid')
df1 = pd.DataFrame(documents1)
print(df1)

#--- Retrieving documents in index that match a title ---#
documents2 = es.search(index='covid',body={"query":{"term":{"title" : "clinical"}}})['hits']['hits']
df2 = pd.DataFrame(documents2)
print(df2)

#--- Retrieving documents in index that match a abstract ---#
documents3 = es.search(index='covid',body={"query":{"term":{"abstract" : "objective"}}})['hits']['hits']
df3 = pd.DataFrame(documents3)
print(df3)





