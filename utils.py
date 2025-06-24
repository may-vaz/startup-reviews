
import re
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import streamlit as st

# Remove spaCy dependency
def preprocess_reviews(reviews):
    combined = ' '.join(reviews).lower()
    # Simple text cleaning
    words = re.findall(r'\b[a-zA-Z]{3,}\b', combined)
    
    # Basic stop words list
    stop_words = {'the', 'and','not','keeps','getting', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'}
    
    filtered_words = [word for word in words if word not in stop_words]
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