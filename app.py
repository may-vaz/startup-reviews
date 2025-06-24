import streamlit as st
import pandas as pd
from sentiment import analyze_sentiment
from utils import display_wordcloud_and_barchart, plot_sentiment_chart



st.set_page_config(page_title="Feedback Analyzer", layout="wide")
st.markdown("<h1 style='text-align: center;'>📊 Feedback Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Understand what users really feel about your product</p>", unsafe_allow_html=True)
st.markdown("---") 

option = st.radio("Choose Input Method", ["Paste Reviews", "Upload CSV"])

reviews = None

if option == "Paste Reviews":
    reviews_text = st.text_area("Paste feedback (one per line):", height=200)
    if st.button("Analyze") and reviews_text.strip():
        reviews = [line.strip() for line in reviews_text.split('\n') if line.strip()]
elif option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload CSV with 'review' column", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if 'review' not in df.columns:
            st.error("CSV must contain a column named 'review'")
        else:
            reviews = df['review'].dropna().tolist()

if reviews:
    st.subheader("📌 Sentiment Analysis")
    sentiment_df = analyze_sentiment(reviews)
    st.dataframe(sentiment_df)
    st.markdown("---") 
    st.pyplot(plot_sentiment_chart(sentiment_df))

    st.markdown("---") 


    st.subheader("📌 Word Cloud + Keyword Bar Chart")
    display_wordcloud_and_barchart(sentiment_df)


