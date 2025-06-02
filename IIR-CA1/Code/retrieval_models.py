# retrieval_models.py
from collections import defaultdict
import math

def bm25_score(query, inverted_index, k=1.5, b=0.75):
    """
    Compute BM25 scores for a given query against all documents in the inverted index.

    Parameters:
    - query: Query object
    - inverted_index: InvertedIndex object
    - k: BM25 k parameter (default=1.5)
    - b: BM25 b parameter (default=0.75)

    Returns:
    - scores: Dictionary mapping doc_id to BM25 score
    """
    scores = defaultdict(float)
    for term in query.tokens:
        idf = inverted_index.compute_idf(term)
        postings = inverted_index.get_postings(term)
        query_weight = query.term_weights.get(term, 1.0)
        for doc_id, freq in postings:
            doc_len = inverted_index.doc_lengths[doc_id]
            avg_doc_len = inverted_index.avg_doc_len
            numerator = freq * (k + 1)
            denominator = freq + k * (1 - b + b * (doc_len / avg_doc_len))
            score = idf * (numerator / denominator)
            scores[doc_id] += query_weight * score  # Incorporate term weight
    return scores

#YOUR CODE HERE (implement the retrieval models)

def model1(query, inverted_index):
   
    scores = defaultdict(float)
    for term in query.tokens:
        idf = inverted_index.compute_idf(term)
        postings = inverted_index.get_postings(term)
        for doc_id, freq in postings:
            
            scores[doc_id] += idf 
    return scores

from collections import defaultdict

def model2(query, inverted_index, k=1.5):
   
    scores = defaultdict(float)
    for term in query.tokens:
        postings = inverted_index.get_postings(term)
        query_weight = query.term_weights.get(term, 1.0)
        for doc_id, freq in postings:
            
            score = (freq * (k + 1)) / (freq + k)
            scores[doc_id] += query_weight*score 
    return scores

def model3(query, inverted_index, k=1.5):
    scores = defaultdict(float)
    for term in query.tokens:
        idf = inverted_index.compute_idf(term)
        postings = inverted_index.get_postings(term)
        query_weight =  query.term_weights.get(term, 1.0)
        for doc_id, freq in postings:
            doc_len = inverted_index.doc_lengths[doc_id]
            avg_doc_len = inverted_index.avg_doc_len
            numerator = freq * (k + 1)
            denominator = k*(doc_len/avg_doc_len) + freq
            score = idf * (numerator/denominator)
            scores[doc_id] +=query_weight * score
    return(scores)
            
def model4(query, inverted_index):
    scores = defaultdict(float)
    for term in query.tokens:
        postings = inverted_index.get_postings(term)
        query_weight =  query.term_weights.get(term, 1.0)
        for doc_id, freq in postings:
            if freq > 0:
                scores[doc_id] += 1
                
    return scores

def model5(query, inverted_index, k, b, d):
    scores = defaultdict(float)
    for term in query.tokens:
        idf = inverted_index.compute_idf(term)
        postings = inverted_index.get_postings(term)
        query_weight = query.term_weights.get(term, 1.0)
        for doc_id, freq in postings:
            doc_len = inverted_index.doc_lengths[doc_id]
            avg_doc_len = inverted_index.avg_doc_len
            numerator = (k+1)*((freq/(1-b+(b*doc_len/avg_doc_len)))+d)
            denominator = k + (freq/(1-b+(b*doc_len/avg_doc_len))+d)
            score = idf * (numerator / denominator)
            scores[doc_id] += query_weight * score  
    return scores

def model6(query, inverted_index, N):
    scores = defaultdict(float)
    for term in query.tokens:
        idf = inverted_index.compute_idf(term)
        posting = inverted_index.get_postings(term)
        query_weight = query.term_weights.get(term, 1.0)
        for doc_id, freq in posting:
            doc_len = inverted_index.doc_lengths[doc_id]
            avg_doc_len = inverted_index.avg_doc_len
            score = freq * math.log(1+(avg_doc_len/doc_len), 2) * math.log(N/idf, 2)
            scores[doc_id] += query_weight * score
    return scores

def model7(query, inverted_index, k, b):
    scores = defaultdict(float)
    for term in query.tokens:
        idf = inverted_index.compute_idf(term)
        postings = inverted_index.get_postings(term)
        query_weight = query.term_weights.get(term, 1.0)
        for doc_id, freq in postings:
            doc_len = inverted_index.doc_lengths[doc_id]
            avg_doc_len = inverted_index.avg_doc_len
            numerator = freq
            denominator = freq + k*((1-b)+b*((doc_len)/avg_doc_len))
            score = idf*idf * (numerator / denominator)
            scores[doc_id] += query_weight * score  
    return scores

def model8(query, inverted_index, k=1.5, b=0.75):
    scores = defaultdict(float)
    for term in query.tokens:
        idf = inverted_index.compute_idf(term)
        postings = inverted_index.get_postings(term)
        query_weight = query.term_weights.get(term, 1.0)
        for doc_id, freq in postings:
            doc_len = inverted_index.doc_lengths[doc_id]
            avg_doc_len = inverted_index.avg_doc_len
            numerator = freq * freq
            denominator = freq*freq + k*((1-b)+b*((doc_len)/avg_doc_len))
            score = idf * (numerator / denominator)
            scores[doc_id] += query_weight * score 
    return scores

def model9(query, inverted_index, k=1.5, b=0.75, d=0.5):
    scores = defaultdict(float)
    for term in query.tokens:
        idf = inverted_index.compute_idf(term)
        postings = inverted_index.get_postings(term)
        query_weight = query.term_weights.get(term, 1.0)
        for doc_id, freq in postings:
            doc_len = inverted_index.doc_lengths[doc_id]
            avg_doc_len = inverted_index.avg_doc_len
            numerator = freq * (k+1)
            denominator = freq + k*((1-b)+b*((doc_len)/avg_doc_len))
            score = idf * (numerator / denominator)
            scores[doc_id] += query_weight * score 
    return scores
#مدل پیشنهادی

def model10(query, inverted_index, alpha=0.5):
    scores = defaultdict(float)
    M = inverted_index.doc_count 
    for term in query.tokens:
        idf = inverted_index.compute_idf(term)
        postings = inverted_index.get_postings(term)
        query_weight = query.term_weights.get(term, 1.0)
        df = inverted_index.term_doc_freq.get(term, 0) 
        for doc_id, freq in postings:
            score = idf * ((1 + (df / M)) ** alpha)
            scores[doc_id] += query_weight * score  
    return scores



#for question number 2:
#مدل اصلی
def model11(query, inverted_index, b=0.75):
    scores = defaultdict(float)
    M = inverted_index.doc_count 
    for term in query.tokens:
        df = inverted_index.term_doc_freq.get(term, 0)  
        postings = inverted_index.get_postings(term)
        query_weight = query.term_weights.get(term, 1.0)

        for doc_id, freq in postings:
            doc_len = inverted_index.doc_lengths[doc_id]
            avg_doc_len = inverted_index.avg_doc_len
            numerator = query_weight * (math.log(1 + math.log(1 + freq)))* math.log((M + 1) / (df + 1))
            denominator = (1 - b + b * (doc_len / avg_doc_len))
            score = numerator / denominator
            scores[doc_id] += query_weight * score  

    return scores
#مدل بدون مولقه تو در تو
def model12(query, inverted_index, b):
    scores = defaultdict(float)
    M = inverted_index.doc_count  
    for term in query.tokens:
        df = len(inverted_index.get_postings(term)) 
        postings = inverted_index.get_postings(term)
        query_weight = query.term_weights.get(term, 1.0)
        for doc_id, freq in postings:
            doc_len = inverted_index.doc_lengths[doc_id]
            idf = inverted_index.compute_idf(term)
            avg_doc_len = inverted_index.avg_doc_len
            numarator = query_weight*(math.log(1+freq))*(math.log((M+1)/df))
            denominator = 1 - b + b*(doc_len/avg_doc_len)
            score = numarator / denominator
            scores[doc_id] += query_weight * score  

    return scores