import pandas as pd
from nltk import pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords
import string
import re
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from autocorrect import Speller
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import nest_asyncio
import uvicorn
import os
from fastapi.responses import HTMLResponse
from fastapi import Request
import csv
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.exceptions import NotFittedError
from typing import List
from scipy.sparse import csr_matrix
import numpy as np

# SCIENCE DATASET

def createScienceQueriesVector():
    queries = pd.read_csv("science_dataset/original_queries.csv")
    queries_vectorizer = TfidfVectorizer(stop_words='english')
    vectorized_queries = queries_vectorizer.fit_transform(queries['text'])

    with open('science_dataset/objects/queries_vectorizer.pkl', 'wb') as file:
        pickle.dump(queries_vectorizer, file)

    with open('science_dataset/objects/vectorized_queries.pkl', 'wb') as file:
        pickle.dump(vectorized_queries, file)
    

def importScienceOriginalDocuments():
    original_docs = pd.read_csv('science_dataset/original_documents.csv')
    original_docs = original_docs.fillna('')
    return original_docs

def importScienceOriginalQueries():
    # Load the vectorized_docs object
    queries = pd.read_csv("science_dataset/original_queries.csv")
    queries = queries.fillna('')
    return queries

def importScienceVectorizer():
    # Load the vectorizer object
    with open('science_dataset/objects/vectorizer.pkl', 'rb') as file:
        vectorizer = pickle.load(file)
    return vectorizer

def importScienceQueriesVectorizer():
    # Load the vectorizer object
    with open('science_dataset/objects/queries_vectorizer.pkl', 'rb') as file:
        queries_vectorier = pickle.load(file)
    return queries_vectorier

def importScienceVectorizedDocuments():
    # Load the vectorized_docs object
    with open('science_dataset/objects/vectorized_docs.pkl', 'rb') as file:
        vectorized_docs = pickle.load(file)
    return vectorized_docs

def importScienceVectorizedQueries():
    # Load the vectorized_docs object
    with open('science_dataset/objects/vectorized_queries.pkl', 'rb') as file:
        vectorized_queries = pickle.load(file)
    return vectorized_queries

def importScienceDataset():
    global OriginalDocuments
    global Queries
    global Vectorizer 
    global VectorizedDocuments
    global VectorizedQueries
    global QueriesVectorizer
    OriginalDocuments = importScienceOriginalDocuments()
    Queries = importScienceOriginalQueries()
    Vectorizer = importScienceVectorizer()
    VectorizedDocuments = importScienceVectorizedDocuments()
    VectorizedQueries =  importScienceVectorizedQueries()
    QueriesVectorizer = importScienceQueriesVectorizer()

# CLINICAL 

def createClinicalQueriesVector():
    queries = pd.read_csv("clinical_dataset/original_queries.csv")

    # Replace np.nan values with empty strings
    queries['disease'].fillna('', inplace=True)
    queries['gene'].fillna('', inplace=True)
    queries['demographic'].fillna('', inplace=True)
    queries['other'].fillna('', inplace=True)

    queries_vectorizer = TfidfVectorizer(stop_words='english')
    vectorized_queries = queries_vectorizer.fit_transform(queries['disease'] + ' ' + queries['gene'] + ' ' + queries['demographic'] + ' ' + queries['other'])

    with open('clinical_dataset/objects/queries_vectorizer.pkl', 'wb') as file:
        pickle.dump(queries_vectorizer, file)

    with open('clinical_dataset/objects/vectorized_queries.pkl', 'wb') as file:
        pickle.dump(vectorized_queries, file)
def importClinicalOriginalDocuments():
    original_docs = pd.read_csv('clinical_dataset/original_docs.csv')
    original_docs = original_docs.fillna('')
    return original_docs

def importClinicalOriginalQueries():
    # Load the vectorized_docs object
    queries = pd.read_csv("clinical_dataset/original_queries.csv")
    queries = queries.fillna('')
    return queries

def importClinicalVectorizer():
    # Load the vectorizer object
    with open('clinical_dataset/objects/vectorizer.pkl', 'rb') as file:
        vectorizer = pickle.load(file)
    return vectorizer

def importClinicalQueriesVectorizer():
    # Load the vectorizer object
    with open('clinical_dataset/objects/queries_vectorizer.pkl', 'rb') as file:
        queries_vectorier = pickle.load(file)
    return queries_vectorier

def importClinicalVectorizedDocuments():
    # Load the vectorized_docs object
    with open('clinical_dataset/objects/vectorized_docs.pkl', 'rb') as file:
        vectorized_docs = pickle.load(file)
    return vectorized_docs

def importClinicalVectorizedQueries():
    # Load the vectorized_docs object
    with open('clinical_dataset/objects/vectorized_queries.pkl', 'rb') as file:
        vectorized_queries = pickle.load(file)
    return vectorized_queries

def importClinicalDataset():
    global OriginalDocuments
    global Queries
    global Vectorizer 
    global VectorizedDocuments
    global VectorizedQueries
    global QueriesVectorizer
    OriginalDocuments = importClinicalOriginalDocuments()
    Queries = importClinicalOriginalQueries()
    Vectorizer = importClinicalVectorizer()
    VectorizedDocuments = importClinicalVectorizedDocuments()
    VectorizedQueries =  importClinicalVectorizedQueries()
    QueriesVectorizer = importClinicalQueriesVectorizer()

def dataProcessing(text):
    # To lower case
    text = text.lower()
    
    # Remove punctuation
    trans_table = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    text = text.translate(trans_table)
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    text = [word for word in text.split() if word not in stop_words]
    
    # Lemmatize text
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = []
    words = word_tokenize(str(text))
    for word in words:
        x = pos_tag([word])
        my_pos = wordnet.NOUN
        if x[0][1][0].lower() == 'v':
            my_pos = wordnet.VERB
        lemmatized_words.append(lemmatizer.lemmatize(word, pos=my_pos))
    text = ' '.join(lemmatized_words)
    
    # Remove Non-alphanumeric Characters
    text = re.compile('[^a-zA-Z0-9\s]').sub('', str(text))
    
    return text


def getQuerySuggestions(query, queries_vectorizer, vectorized_queries, queries):
    try:
        vectorized_query = queries_vectorizer.transform([query])
    except NotFittedError:
        print("Vectorize the queries first using 'createQueriesVector()' function.")
        return []
    similarity_scores = cosine_similarity(vectorized_query, vectorized_queries)
    results = list(enumerate(similarity_scores[0]))
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    result = []
    try:
        for res in sorted_results:
            result.append(queries['text'][res[0]])
    except Exception as e:
        for res in sorted_results:
            result.append(queries['disease'][res[0]] + ' ' + queries['gene'][res[0]] + ' ' + queries['demographic'][res[0]] + ' ' + queries['other'][res[0]])
    return result[:5]


def getFirstTenDocument(sorted_results):
    result = []
    i = 0
    try:
        for res in sorted_results:
            if i < 10:
                doc_id= int(OriginalDocuments['doc_id'][res[0]])  # Convert to Python int
                # doc_id= original_docs['doc_id'][res[0]]  
                content = OriginalDocuments['text'][res[0]]
                result.append({"doc_id": doc_id, "content": content})
                i += 1
            else:
                break
    except Exception as e:
        for res in sorted_results:
            if i < 10:
                doc_id = str(OriginalDocuments['doc_id'][res[0]])  # Convert to string
                content = (
                    "title: "
                    + str(OriginalDocuments['title'][res[0]])
                    + ', summary: '
                    + str(OriginalDocuments['summary'][res[0]])
                    + ', detailed_description: '
                    + str(OriginalDocuments['detailed_description'][res[0]])
                )
                result.append({"doc_id": doc_id, "content": content})
                i += 1
            else:
                break
    
    return result

# Create an instance of the FastAPI app
app = FastAPI()

class Query(BaseModel):
    query: str

class QueryVector(BaseModel):
    query_vector: List[List[float]]

@app.get("/", response_class=HTMLResponse)
def read_index():
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(index_path, "r") as f:
        return f.read()

@app.get("/search-result.html", response_class=HTMLResponse)
def read_search_result():
    index_path = os.path.join(os.path.dirname(__file__), "search-result.html")
    with open(index_path, "r") as f:
        return f.read()


def matchingAndRanking(queryVector,documentsVector):
    # Calculate cosine similarity for vectorized_query and vectorized_docs
    similarity_scores = cosine_similarity(queryVector, documentsVector)
    # This result contains objects like this (3796, 0.9705645380366313) first attribute (index in related docs)
    # second attribute (similarity score)
    results = list(enumerate(similarity_scores[0]))
    # Sort results by score (descending) from higher score to lower score 
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    return sorted_results

def queryIndexing(processedQuery, vectorizer):
    vectorized_query = vectorizer.transform([processedQuery])
    return vectorized_query.toarray().tolist()

# end_point 1
@app.post("/api/preprocess-query")
def preprocess_query(query: Query):
    processedQuery = dataProcessing(query.query)
    return {'result' : processedQuery}

# end_point 2
@app.post("/api/indexing-query")
def indexing_query(query: Query):
    query_index = queryIndexing(query.query,Vectorizer)
    return {'result' : query_index}

# end_point 3
@app.post("/api/matching-and-ranking")
def matching_and_ranking(query: QueryVector):
    query_vector_dense = np.array(query.query_vector)
    query_vector_sparse = csr_matrix(query_vector_dense)
    sorted_results = matchingAndRanking(query_vector_sparse,VectorizedDocuments)
    documents = getFirstTenDocument(sorted_results)
    return {'result' : documents}


@app.post("/api/suggestion")
def suggest_query(query: Query):
    if query.query.strip():  # Check if query has non-space characters
        suggested_queries = getQuerySuggestions(query.query, QueriesVectorizer, VectorizedQueries, Queries)
        return {"result": suggested_queries}
    else:
        return {"result": []}
    
@app.get("/import_dataset/{value}")
def import_dataset(value: int):
    if value == 0:
        importScienceDataset()
        return {"message": "Science dataset imported successfully."}
    elif value == 1:
        importClinicalDataset()
        return {"message": "Clinical dataset imported successfully."}
    else:
        return {"message": "Invalid value. Please provide 0 or 1."}

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # print(os.getcwd())
    nest_asyncio.apply()
    uvicorn.run(app=app, host="localhost", port=8000, log_level="info")

if __name__ == "__main__":
    main()