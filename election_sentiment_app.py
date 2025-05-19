import pandas as pd
import numpy as np
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from io import StringIO
import time
import emoji
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
from langdetect import detect, DetectorFactory
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import os
import langid
from googletrans import Translator
# Initialize NLTK with comprehensive error handling
def initialize_nltk():
    try:
        # Set NLTK data path if needed
        nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
        if not os.path.exists(nltk_data_path):
            os.makedirs(nltk_data_path)
        nltk.data.path.append(nltk_data_path)
        
        # Download all required resources with verification
        required_nltk = [
            ('punkt', 'tokenizers/punkt'),
            ('wordnet', 'corpora/wordnet'),
            ('stopwords', 'corpora/stopwords'),
            ('omw-1.4', 'corpora/omw-1.4'),
            ('averaged_perceptron_tagger', 'taggers/averaged_perceptron_tagger')
        ]
        
        for resource, path in required_nltk:
            try:
                nltk.data.find(path)
            except LookupError:
                nltk.download(resource, quiet=True)
                
        # Verify punkt_tab specifically
        try:
            nltk.data.find('tokenizers/punkt_tab/english')
        except LookupError:
            # Fallback to standard punkt if punkt_tab isn't available
            nltk.download('punkt', quiet=True)
            
    except Exception as e:
        st.error(f"Failed to initialize NLTK: {str(e)}")
        st.stop()

# Initialize NLTK before anything else
initialize_nltk()

# Set deterministic language detection
DetectorFactory.seed = 0

# Initialize components with error handling
try:
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    vader_analyzer = SentimentIntensityAnalyzer()
except Exception as e:
    st.error(f"Failed to initialize NLP components: {str(e)}")
    st.stop()

# Translate text to English based on language detection
def translate_to_english(text):
    lang, confidence = langid.classify(text)

    if lang == 'ta':
        # Translate pure Tamil to English
        translator = Translator()
        translation = translator.translate(text, src='ta', dest='en')
        return translation.text
    elif lang == 'en':
        # Translate English-written Tamil (Thanglish) to English
        # This assumes that Thanglish contains Tamil characters in an English script
        return text
    else:
        # Leave as is for other languages (including English)
        return text

# Set page config
st.set_page_config(
    page_title="Election Sentiment Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS Styling
st.markdown("""
<style>
    :root {
        --primary-dark: #1a237e;
        --primary-light: #3949ab;
        --secondary: #00acc1;
        --accent: #ff7043;
        --text-primary: #e8eaf6;
        --text-secondary: #c5cae9;
        --success: #4caf50;
        --danger: #f44336;
        --warning: #ff9800;
        --info: #2196f3;
    }
    
    * {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        transition: all 0.3s ease;
    }
    
    .header {
        background: linear-gradient(135deg, var(--primary-dark), var(--primary-light));
        color: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        margin-bottom: 2rem;
        text-align: center;
        animation: fadeIn 1s ease-out;
        border-bottom: 4px solid var(--secondary);
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .party-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border-left: 4px solid var(--secondary);
        transition: all 0.3s ease;
        color: black;  
    }
    
    .party-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.12);
    }
    
    .result-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        animation: slideUp 0.5s ease-out;
        color: black;  
    }
    
    .winner-card {
        background: linear-gradient(135deg, var(--success), #66bb6a);
        color: white ;
        border-left: 6px solid white;
        animation: pulse 2s infinite;
    }
    .winner-card .positive {
        color: white !important;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.01); }
        100% { transform: scale(1); }
    }
    
    .positive { color: var(--success); font-weight: 600; }
    .negative { color: var(--danger); font-weight: 600; }
    .neutral { color: var(--warning); font-weight: 600; }
    
    .footer {
        background: linear-gradient(135deg, var(--primary-dark), var(--primary-light));
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-top: 3rem;
    }
    
    .footer-steps {
        display: flex;
        justify-content: space-between;
        margin-top: 1.5rem;
    }
    
    .step {
        background: rgba(255,255,255,0.1);
        padding: 1.2rem;
        border-radius: 8px;
        width: 30%;
        transition: all 0.3s ease;
    }
    
    .step:hover {
        transform: translateY(-5px);
        background: rgba(255,255,255,0.15);
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--secondary), var(--accent));
    }
    
    .stSelectbox, .stSlider, .stCheckbox {
        margin-bottom: 1rem;
    }
    
    .analysis-method {
        background: rgba(0,0,0,0.03);
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header" style = "background:white">
    <h1 style="margin:0;font-size:2.5rem;color:black">📊 Election Sentiment Analysis Dashboard</h1>
    <p style="margin:0.5rem 0 0;font-size:1.1rem;color:black">Advanced political sentiment analysis with hybrid NLP approaches</p>
</div>
""", unsafe_allow_html=True)

@lru_cache(maxsize=10000)
def detect_language_cached(text):
    try:
        return detect(text)
    except:
        return 'en'

def advanced_preprocess_text(text):
    # First translate to English if needed
    text = translate_to_english(text)
    
    # Convert emojis to text
    text = emoji.demojize(text)
    
    # Remove URLs, mentions, hashtags
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'RT[\s]+', '', text)
    text = re.sub(r'\n', ' ', text)
    
    # Expand contractions
    contractions = {
        "won't": "will not", "can't": "cannot", "n't": " not",
        "'re": " are", "'s": " is", "'d": " would",
        "'ll": " will", "'t": " not", "'ve": " have", "'m": " am"
    }
    for cont, exp in contractions.items():
        text = text.replace(cont, exp)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^A-Za-z0-9\s.,!?]', '', text)
    
    # Tokenize and lemmatize
    try:
        tokens = word_tokenize(text.lower())
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and len(token) > 2]
        return ' '.join(tokens)
    except Exception as e:
        st.error(f"Text processing error: {str(e)}")
        return text.lower()  # Fallback to simple lowercase

@lru_cache(maxsize=10000)
def cached_sentiment_analysis(text, method):
    if method == 'textblob':
        return TextBlob(text).sentiment.polarity
    elif method == 'vader':
        return vader_analyzer.polarity_scores(text)['compound']
    else:  # hybrid
        return (TextBlob(text).sentiment.polarity +
                vader_analyzer.polarity_scores(text)['compound']) / 2

def categorize_sentiment(score):
    if score > 0.05: return 'positive'
    elif score < -0.05: return 'negative'
    else: return 'neutral'

def process_tweet_batch(tweets, analysis_method):
    processed_tweets = []
    sentiments = []
    scores = []
    
    for tweet in tweets:
        try:
            processed_text = advanced_preprocess_text(tweet)
            score = cached_sentiment_analysis(processed_text, analysis_method)
            sentiment = categorize_sentiment(score)
            processed_tweets.append(processed_text)
            sentiments.append(sentiment)
            scores.append(score)
        except Exception as e:
            st.warning(f"Error processing tweet: {str(e)}")
            continue
    
    return processed_tweets, sentiments, scores

def create_sentiment_distribution_plot(party_name, positive, negative, neutral):
    fig, ax = plt.subplots(figsize=(7, 7))
    labels = ['Positive', 'Negative', 'Neutral']
    sizes = [positive, negative, neutral]
    colors = ['#4CAF50', '#F44336', '#FFC107']
    explode = (0.05, 0.05, 0.05)  # to slightly separate each slice
    ax.pie(
        sizes,
        explode=explode,
        labels=labels,
        autopct='%1.1f%%',
        startangle=140,
        colors=colors,
        shadow=True
    )
    ax.set_title(f"Sentiment Distribution for {party_name}", fontsize=16)
    ax.axis('equal')  # Equal aspect ratio ensures the pie is a circle.
    return fig

def create_sentiment_trend_plot(df, party_name):
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Resample by week for smoother trend
    weekly_sentiment = df.resample('W', on='date')['sentiment_score'].mean()
    
    # Plot trend line
    weekly_sentiment.plot(ax=ax, color='#3949ab', linewidth=2.5, marker='o')
    
    # Add threshold lines
    ax.axhline(0.05, color='#4CAF50', linestyle='--', alpha=0.7, label='Positive Threshold')
    ax.axhline(-0.05, color='#F44336', linestyle='--', alpha=0.7, label='Negative Threshold')
    
    ax.set_title(f'Sentiment Trend Over Time - {party_name}', pad=15)
    ax.set_ylabel('Average Sentiment Score')
    ax.set_xlabel('Date')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_comparison_chart(results):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    parties = [r['party_name'] for r in results]
    positives = [r['positive'] for r in results]
    negatives = [r['negative'] for r in results]
    neutrals = [r['neutral'] for r in results]
    
    x = np.arange(len(parties))
    width = 0.25
    
    # Create stacked bars
    bar1 = ax.bar(x - width, positives, width, label='Positive', color='#4CAF50')
    bar2 = ax.bar(x, negatives, width, label='Negative', color='#F44336')
    bar3 = ax.bar(x + width, neutrals, width, label='Neutral', color='#FFC107')
    
    ax.set_xticks(x)
    ax.set_xticklabels(parties)
    ax.set_ylabel('Percentage')
    ax.set_title('Sentiment Comparison Across Parties', pad=20)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add value labels
    for bars in [bar1, bar2, bar3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

@st.cache_data
def analyze_party_sentiment(uploaded_file, party_name, analysis_method='hybrid', max_tweets=2000):
    try:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        df = pd.read_csv(stringio)
        
        # Sample data if too large
        if len(df) > max_tweets:
            df = df.sample(max_tweets, random_state=42)
        
        tweets = df['text'].astype(str).tolist()
        
        # Process tweets in batches
        batch_size = 500
        batches = [tweets[i:i + batch_size] for i in range(0, len(tweets), batch_size)]
        
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(
                lambda batch: process_tweet_batch(batch, analysis_method),
                batches
            ))
        
        # Combine results
        processed_tweets = []
        sentiments = []
        scores = []
        for batch_result in results:
            processed_tweets.extend(batch_result[0])
            sentiments.extend(batch_result[1])
            scores.extend(batch_result[2])
        
        # Calculate percentages
        total = len(sentiments)
        positive = sentiments.count('positive') / total * 100
        negative = sentiments.count('negative') / total * 100
        neutral = sentiments.count('neutral') / total * 100
        
        # Create visualizations
        dist_fig = create_sentiment_distribution_plot(party_name, positive, negative, neutral)
        
        # Create trend plot if date column exists
        trend_fig = None
        if 'date' in df.columns:
            try:
                df['date'] = pd.to_datetime(df['date'])
                df['sentiment_score'] = scores
                df['sentiment'] = sentiments
                trend_fig = create_sentiment_trend_plot(df, party_name)
            except Exception as e:
                st.warning(f"Could not create time trend: {str(e)}")
        
        return {
            'party_name': party_name,
            'positive': positive,
            'negative': negative,
            'neutral': neutral,
            'sentiment_distribution': dist_fig,
            'sentiment_trend': trend_fig,
            'processed_tweets': processed_tweets,
            'sentiments': sentiments,
            'raw_data': df
        }
    except Exception as e:
        st.error(f"Error analyzing party sentiment: {str(e)}")
        return None

def main():
    # Initialize session state
    if 'party_count' not in st.session_state:
        st.session_state.party_count = 2
    if 'results' not in st.session_state:
        st.session_state.results = []
    if 'analyzed' not in st.session_state:
        st.session_state.analyzed = False
    
    # Sidebar configuration
    st.sidebar.markdown("### Analysis Configuration")
    
    with st.sidebar.expander("Analysis Settings", expanded=True):
        analysis_method = st.selectbox(
            "Sentiment Analysis Method",
            ["Hybrid (TextBlob + VADER)", "TextBlob Only", "VADER Only"],
            index=0
        )
        
        max_tweets = st.slider(
            "Maximum tweets per party",
            min_value=100,
            max_value=5000,
            value=2000,
            step=100
        )
    
    # Party upload sections
    st.markdown("""
    <div style="text-align:center;margin-bottom:2rem;">
        <h2>Upload Party Data</h2>
        <p>CSV files should contain at least a 'text' column</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create party cards
    cols_per_row = 3
    party_files = {}
    
    # Calculate number of rows needed
    rows = (st.session_state.party_count + cols_per_row - 1) // cols_per_row
    
    for row in range(rows):
        cols = st.columns(cols_per_row)
        for col_idx in range(cols_per_row):
            party_idx = row * cols_per_row + col_idx
            if party_idx < st.session_state.party_count:
                with cols[col_idx]:
                    st.markdown(f"""
                    <div class="party-card">
                        <h3>Party {party_idx + 1}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    party_name = st.text_input(
                        "Enter party name",
                        key=f"party_name_{party_idx}",
                        value=f"Party {party_idx + 1}",
                        label_visibility="collapsed"
                    )
                    
                    uploaded_file = st.file_uploader(
                        "Upload CSV file",
                        type=['csv'],
                        key=f"uploader_{party_idx}",
                        label_visibility="collapsed"
                    )
                    
                    if uploaded_file:
                        party_files[party_name] = uploaded_file
    
    # Add party and analyze buttons
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if st.button("＋ Add Party", key="add_party"):
            st.session_state.party_count += 1
            st.rerun()
    
    with col2:
        analyze_clicked = st.button("🚀 Analyze Sentiment", key="analyze_btn", type="primary")
    
    # Analysis logic
    if analyze_clicked:
        valid_parties = sum(1 for i in range(st.session_state.party_count)
                          if st.session_state.get(f"uploader_{i}") is not None)
        
        if valid_parties < 2:
            st.error("Please upload CSV files for at least two parties.")
            st.stop()
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text('Starting analysis...')
        
        st.session_state.results = []
        st.session_state.analyzed = True
        
        method_map = {
            "Hybrid (TextBlob + VADER)": "hybrid",
            "TextBlob Only": "textblob",
            "VADER Only": "vader"
        }
        
        # Analyze each party
        for i in range(st.session_state.party_count):
            party_name = st.session_state.get(f"party_name_{i}", f"Party {i + 1}")
            uploaded_file = st.session_state.get(f"uploader_{i}")
            
            if uploaded_file:
                status_text.text(f'Analyzing {party_name}...')
                result = analyze_party_sentiment(
                    uploaded_file,
                    party_name,
                    method_map[analysis_method],
                    max_tweets
                )
                if result:  # Only append if analysis succeeded
                    st.session_state.results.append(result)
                
                progress = (i + 1) / st.session_state.party_count
                progress_bar.progress(progress)
        
        progress_bar.progress(1.0)
        status_text.text('Analysis complete!')
        time.sleep(0.5)
        status_text.empty()
        progress_bar.empty()
    
    # Results display
    if st.session_state.get('analyzed', False) and st.session_state.results:
        st.markdown("""
        <div style="text-align:center;margin:3rem 0 1rem;">
            <h2>Analysis Results</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Find winning party
        winning_party = max(st.session_state.results, key=lambda x: x['positive'])
        
        st.markdown(f"""
        <div class="result-card winner-card">
            <h3 style="margin-top:0;">🏆 Predicted Winning Party</h3>
            <p style="font-size:1.2rem;margin-bottom:0.5rem;">
                <strong>{winning_party['party_name']}</strong> has the highest positive sentiment at
                <span class="positive">{winning_party['positive']:.1f}%</span>
            </p>
            <p style="font-size:0.9rem;margin-bottom:0;">
                Analysis Method: {analysis_method}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display party results
        st.markdown("### Party-wise Sentiment Analysis")
        
        tabs = st.tabs([result['party_name'] for result in st.session_state.results])
        
        for idx, tab in enumerate(tabs):
            with tab:
                result = st.session_state.results[idx]
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown(f"""
                    <div class="result-card">
                        <h3>{result['party_name']}</h3>
                        <p>Positive: <span class="positive">{result['positive']:.1f}%</span></p>
                        <p>Negative: <span class="negative">{result['negative']:.1f}%</span></p>
                        <p>Neutral: <span class="neutral">{result['neutral']:.1f}%</span></p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.pyplot(result['sentiment_distribution'])
                
                if result['sentiment_trend'] is not None:
                    st.markdown("#### Sentiment Trend Over Time")
                    st.pyplot(result['sentiment_trend'])
        
        # Comparative analysis
        st.markdown("### Comparative Analysis")
        comparison_fig = create_comparison_chart(st.session_state.results)
        st.pyplot(comparison_fig)
        
        # Footer
        st.markdown("""
        <div class="footer">
            <h3 style="margin-top:0;">Analysis Methodology</h3>
            <div class="footer-steps">
                <div class="step">
                    <h4>1. Data Collection</h4>
                    <p>Twitter data collected using API with relevant political keywords</p>
                </div>
                <div class="step">
                    <h4>2. Sentiment Analysis</h4>
                    <p>Hybrid approach combining lexicon-based and machine learning methods</p>
                </div>
                <div class="step">
                    <h4>3. Visualization</h4>
                    <p>Interactive dashboards showing sentiment distribution and trends</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()