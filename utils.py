import spacy
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import streamlit as st



def preprocess_reviews(reviews):
    combined = ' '.join(reviews)
    doc = nlp(combined)

    filtered_words = [
        token.lemma_.lower()
        for token in doc
        if token.pos_ in ["NOUN", "ADJ"]
        and not token.is_stop
        and token.is_alpha
        and len(token.text) > 2
    ]

    return filtered_words

def plot_wordcloud(words):
    wc = WordCloud(
        width=600, height=400,
        background_color='white',
        max_words=100
    ).generate(" ".join(words))
    fig, ax = plt.subplots()
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    return fig

def plot_keyword_bar(words):
    top_keywords = Counter(words).most_common(10)
    keywords, counts = zip(*top_keywords)
    fig, ax = plt.subplots()
    ax.barh(keywords[::-1], counts[::-1], color='steelblue')
    ax.set_title("Top Keywords")
    ax.set_xlabel("Frequency")
    return fig

def display_wordcloud_and_barchart(df):
    words = preprocess_reviews(df['review'].tolist())
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("Word Cloud")
        st.pyplot(plot_wordcloud(words))
    with col2:
        st.markdown("Top Keywords")
        st.pyplot(plot_keyword_bar(words))


def plot_sentiment_chart(df):
    fig, ax = plt.subplots()
    df['sentiment'].value_counts().plot(kind='bar', ax=ax, color='orchid')
    ax.set_title("Sentiment Breakdown")
    ax.set_ylabel("Count")
    ax.set_xlabel("Sentiment")
    return fig
