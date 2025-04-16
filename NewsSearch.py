import sys
import json
import re
import spacy
from newspaper import Article

first = True
count = 0
nlp = spacy.load("en_core_web_sm")

file = sys.argv[2]
article_count = 0
if len(sys.argv) == 3:
    article_count = int(sys.argv[3])

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

with open(file, 'r') as f:
    with open('Parsed_News.json', 'w') as out:
        IDF = {}
        noun_phrase_IDF = {}

        for line in f:
            if article_count != 0 and count >= article_count + 1:
                break 
            if count % 100 == 0:
                print(f"currently at {count}")
                # Check point to store idf every 100 articles
                with open('single_word_idf.json', 'w') as idf:
                    json.dump(IDF, idf)
                with open('np_idf.json', 'w') as np_idf:
                    json.dump(noun_phrase_IDF, np_idf)
            count+=1
            # Get text from article link
            cur = json.loads(line)
            try:
                article = Article(cur['link'])
                article.download()
                article.parse()
                text = article.text
            except Exception as e:
                print(f"Failed at {cur['link']} due to: {e}")
                continue

            # Find TF and update IDF
            single_word_TF = {}
            total_words = 0
            tokens = re.findall(r"\w+", text.lower())
            word_set = set()
            for word in tokens:
                total_words += 1
                single_word_TF[word] = single_word_TF.get(word, 0) + 1
                word_set.add(word)
            for word in word_set:
                IDF[word] = IDF.get(word, 0) + 1

            # Now for Noun Phrases
            noun_phrase_TF = {}
            total_noun_phrases = 0 
            doc = nlp(text)
            noun_phrase_set = set()
            for chunk in doc.noun_chunks:
                full_np = chunk.text.lower()
                total_noun_phrases += 1
                noun_phrase_TF[full_np] = noun_phrase_TF.get(full_np, 0) + 1
                noun_phrase_set.add(full_np)

                for sub_np in extract_sub_phrases(chunk):
                    total_noun_phrases += 1
                    noun_phrase_TF[sub_np] = noun_phrase_TF.get(sub_np, 0) + 1
                    noun_phrase_set.add(sub_np)

            for noun_phrase in noun_phrase_set:
                noun_phrase_IDF[noun_phrase] = noun_phrase_IDF.get(noun_phrase, 0) + 1

            cur['total_words'] = total_words
            cur['total_noun_phrases'] = total_noun_phrases
            cur['single_word_tf'] = single_word_TF
            cur['noun_phrase_tf'] = noun_phrase_TF
            out.write(json.dumps(cur)+'\n')

        # Keep track of both idfs
        with open('single_word_idf.json', 'w') as idf:
            json.dump(IDF, idf)
        with open('np_idf.json', 'w') as np_idf:
            json.dump(noun_phrase_IDF, np_idf)


                