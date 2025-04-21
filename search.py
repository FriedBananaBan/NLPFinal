import sys
import json
import heapq
import re
import math
import spacy

nlp = spacy.load("en_core_web_sm")
# Used to get nested nps
def extract_sub_phrases(chunk):
    words = chunk.text.lower().split()
    n = len(words)
    sub_phrases = set()
    for i in range(n):
        for j in range(i + 1, n + 1):
            phrase = ' '.join(words[i:j])
            if len(phrase.split()) > 1:  
                sub_phrases.add(phrase)
    return sub_phrases

file = sys.argv[1]
matches_count = int(sys.argv[4])

with open(sys.argv[2], "r") as idf:
    single_word_idf = json.load(idf)

with open(sys.argv[3], "r") as idf:
    np_idf = json.load(idf)

query = input("Input your query: ")
tokens = re.findall(r"\w+", query.lower())

query_tf = {}
query_idf = {}

query_np_tf = {}
query_np_idf = {}

single_word_scores = []
np_scores = []

with open(file, 'r') as f:
    article_count = sum(1 for _ in f)

# Calculate single_word_TFIDF for query
for word in tokens:
    query_tf[word] = query_tf.get(word, 0) + 1
    query_idf[word] = math.log(article_count/(single_word_idf.get(word, 0)+1))

for word in query_tf:
    query_tf[word] /= len(tokens)

query_vec = {}
for word in query_tf:
    query_vec[word] = query_tf[word] * query_idf[word]
query_norm = math.sqrt(sum(val**2 for val in query_vec.values()))

# Calculate np_TFIDF for query
np_count = 0
doc = nlp(query)
for chunk in doc.noun_chunks:
    np = chunk.text.lower()
    np_count+=1
    query_np_tf[np] = query_np_tf.get(np, 0) + 1
    query_np_idf[np] = math.log(article_count/(np_idf.get(np, 0)+1))

    for sub_np in extract_sub_phrases(chunk):
        np_count+=1
        query_np_tf[sub_np] = query_np_tf.get(sub_np, 0) + 1
        query_np_idf[sub_np] = math.log(article_count/(np_idf.get(sub_np, 0)+1))

for np in query_np_tf:
    count = query_np_tf[np]
    query_np_tf[np] = count/np_count

query_np_vec = {}
for np in query_np_tf:
    query_np_vec[np] = query_np_tf[np] * query_np_idf[np]

query_np_norm = math.sqrt(sum(val**2 for val in query_np_vec.values()))

# Tiebreakers incase of equal cosine similarity
i = 0
j = 0
with open(file, "r") as data:
    for news in data:
        cur = json.loads(news)
        total_words = cur['total_words']
        total_noun_phrases = cur['total_noun_phrases']
        single_word_TF = cur['single_word_tf']
        noun_phrase_TF = cur['noun_phrase_tf']

        doc_vec = {}
        for word in tokens:
            if word in single_word_TF:
                tf = single_word_TF[word]/total_words
                idf = math.log(article_count/(single_word_idf.get(word, 0)+1))
                doc_vec[word] = tf * idf

        doc_np_vec = {}
        for np in query_np_tf:
            if np in noun_phrase_TF:
                tf = noun_phrase_TF[np]/total_noun_phrases
                idf = math.log(article_count/(np_idf.get(np, 0)+1))
                doc_np_vec[np] = tf * idf


        dot_product = sum(query_vec[w] * doc_vec.get(w, 0.0) for w in query_vec)
        doc_norm = math.sqrt(sum(val**2 for val in doc_vec.values()))

        np_dot_product = sum(query_np_vec[w] * doc_np_vec.get(w, 0.0) for w in query_np_vec)
        doc_np_norm = math.sqrt(sum(val**2 for val in doc_np_vec.values()))

        if query_norm > 0 and doc_norm > 0:
            cosine_similarity = dot_product / (query_norm * doc_norm)
            heapq.heappush(single_word_scores, (cosine_similarity, i, cur))
            i+=1
            if len(single_word_scores) > matches_count:
                heapq.heappop(single_word_scores)

        if query_np_norm > 0 and doc_np_norm > 0:
            np_cosine_similarity = np_dot_product / (query_np_norm * doc_np_norm)
            heapq.heappush(np_scores, (np_cosine_similarity, j, cur))
            j+=1
            if len(np_scores) > matches_count:
                heapq.heappop(np_scores)

print("\nResults from Single Word TFIDF\n")
for score, index, article in sorted(single_word_scores, reverse=True):
    print(f"{score:.4f} — {article['headline']}\n{article['link']}")

print("\nResults from Noun Phrase TFIDF\n")
for score, index, article in sorted(np_scores, reverse=True):
    print(f"{score:.4f} — {article['headline']}\n{article['link']}")