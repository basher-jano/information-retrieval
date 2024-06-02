

# Search Engine for Clinical and Science Datasets

## Overview

This project implements a search engine that processes, indexes, and retrieves documents from two datasets: clinical and science. The search engine is designed using the principles of Service-Oriented Architecture (SOA) to ensure modularity, scalability, and maintainability.

## Datasets

We are working with two Information Retrieval (IR) datasets:
1. **Clinical Dataset**
2. **Science Dataset**

## Project Structure

The project directory contains the following files:

- **clinical_dataset_proccisng.ipynb**: Contains the methods used for cleaning the documents in the clinical dataset.
- **clinical_dataset_vector_store.ipynb**: Contains the modified code for storing the vectors of the clinical dataset using Pinecone.
- **clinical_dataset_word_embedding.ipynb**: Contains the modified code for word embedding with Word2Vec model for the clinical dataset.
- **clinical_dataset_evaluation.ipynb**: Contains the Mean Average Precision (MAP) and Mean Reciprocal Rank (MRR) results after running all the queries in the clinical dataset.
- **science_dataset_processing.ipynb**: Contains the methods used for cleaning the documents in the science dataset.
- **science_dataset_vector_store.ipynb**: Contains the modified code for storing the vectors of the science dataset using Pinecone.
- **science_dataset_word_embedding.ipynb**: Contains the modified code for word embedding with Word2Vec model for the science dataset.
- **science_dataset_evaluation.ipynb**: Contains the MAP and MRR results after running all the queries in the science dataset.
- **index.html**: The main HTML page for the search engine.
- **search-result.html**: The HTML page for displaying the search results.
- **search_engin.py**: The backend Python file containing the API endpoints and search engine logic.

## Features

### Data Processing

- **Cleaning and Preprocessing**: The processing files contain methods for cleaning and preprocessing the documents in both datasets. The preprocessing methods are similar for each dataset but are organized into separate files for better organization.

### Vector Storage

- **Pinecone Indexes**: We use Pinecone for vector storage, creating two separate indexes for the clinical and science datasets.

### Word Embedding

- **Word2Vec**: The word embedding files contain modified code for generating word embeddings using the Word2Vec model.

### Evaluation

- **MAP and MRR**: The evaluation files contain the MAP and MRR results, which evaluate the performance of the search engine by running all queries in the dataset and using cosine similarity for matching and ranking.

### Search Engine

The search engine is implemented with the following features:

- **Service-Oriented Architecture (SOA)**: The search engine is designed using SOA principles, with separate services for query preprocessing, indexing, and matching & ranking.
- **Query Suggestion**: Implements query suggestion by comparing the user query vector with the query vectors of the dataset and using cosine similarity to suggest the five most related queries to the user query.

## API Endpoints

1. **Preprocess Query**:
   - Endpoint: `/api/preprocess-query`
   - Method: `POST`
   - Description: Processes the raw user query.

2. **Indexing Query**:
   - Endpoint: `/api/indexing-query`
   - Method: `POST`
   - Description: Converts the processed query into a vector.

3. **Matching and Ranking**:
   - Endpoint: `/api/matching-and-ranking`
   - Method: `POST`
   - Description: Matches the query vector with document vectors and ranks the results, returning the top documents.

## Getting Started

### Prerequisites

- Python
- Jupyter Notebook
- Pinecone Account
- Word2Vec Model

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/search-engine.git
   ```
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up Pinecone indexes for clinical and science datasets.

### Running the Project

1. Run the Jupyter notebooks to preprocess the datasets, generate word embeddings, store vectors, and evaluate the datasets.
2. Start the backend server:
   ```bash
   uvicorn search_engin:app --reload
   ```
3. Open `index.html` in your browser to access the search engine.

## Usage

1. Enter a search query in the search box on the main page.
2. The query will be processed, indexed, and matched against the documents in the datasets.
3. The top-ranked documents will be displayed on the search results page.
4. The search engine will also suggest related queries based on cosine similarity.

## Conclusion

This project demonstrates a modular and scalable approach to building a search engine using SOA principles. By separating concerns into different services, we ensure that each part of the application can be developed, tested, and scaled independently. The addition of query suggestions enhances the user experience by providing relevant query recommendations.


