from hazm import word_tokenize, Postagger, Normalizer
from hazm import SentimentAnalyzer
from spellchecker import SpellChecker
from langdetect import detect
from gensim import corpora
from gensim.models import LdaModel
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from transformers import pipeline
import textstat

def preprocess_text(text):
    normalizer = Normalizer()
    normalized_text = normalizer.normalize(text)
    return word_tokenize(normalized_text)

def analyze_text(text):
    words = preprocess_text(text)

    word_count = len(words)

    sentiment_analyzer = pipeline('sentiment-analysis', model='HooshvareLab/bert-base-parsbert-uncased')
    sentiment = sentiment_analyzer(text)[0]['label']

    summarizer = pipeline('summarization', model='HooshvareLab/bert-base-parsbert-uncased')
    summary = summarizer(text, max_length=100, min_length=30, do_sample=False)
    summary_text = summary[0]['summary_text']

    spell = SpellChecker(language='fa')
    corrected_words = [spell.candidates(word).pop() if word not in spell else word for word in words]
    corrected_text = " ".join(corrected_words)

    texts = [words]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda_model = LdaModel(corpus, num_topics=2, id2word=dictionary, passes=15)
    topics = lda_model.print_topics(num_words=4)

    language = detect(text)

    G = nx.Graph()
    for word1 in words:
        for word2 in words:
            if word1 != word2:
                G.add_edge(word1, word2)

    plt.figure(figsize=(10, 10))
    nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=10)
    plt.show()

    return {
        'word_count': word_count,
        'sentiment': sentiment,
        'summary': summary_text,
        'corrected_text': corrected_text,
        'topics': topics,
        'language': language
    }

def visualize_data(result):
    fig, ax = plt.subplots(2, 1, figsize=(12, 10))
    
    bigram_df = pd.DataFrame(result.get('bigram_freq', {}).items(), columns=['Bigram', 'Frequency'])
    sns.barplot(x='Bigram', y='Frequency', data=bigram_df, ax=ax[0])
    ax[0].set_title('Bigram Frequencies')
    ax[0].tick_params(axis='x', rotation=90)
    
    trigram_df = pd.DataFrame(result.get('trigram_freq', {}).items(), columns=['Trigram', 'Frequency'])
    sns.barplot(x='Trigram', y='Frequency', data=trigram_df, ax=ax[1])
    ax[1].set_title('Trigram Frequencies')
    ax[1].tick_params(axis='x', rotation=90)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    text = "علی و مریم به پارک رفتند. مریم از دیدن گل‌ها بسیار خوشحال شد."

    result = analyze_text(text)

    print("Word Count:", result['word_count'])
    print("Sentiment:", result['sentiment'])
    print("Summary:", result['summary'])
    print("Corrected Text:", result['corrected_text'])
    print("Topics:", result['topics'])
    print("Language:", result['language'])

    visualize_data(result)