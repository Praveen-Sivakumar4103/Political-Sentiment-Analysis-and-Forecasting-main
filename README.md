# Political-Sentiment-Analysis-and-Forecasting
Analyzes political sentiment from Tamil tweets using Python. Tweets are scraped, translated, and processed using VADER and TextBlob for hybrid sentiment analysis. Optimized with preprocessing, multithreading, and caching to deliver accurate and efficient public opinion insights.

## Key Features 

### 📊Hybrid Sentimental Analysis
Combines TextBlob (rule-based) and VADER (social-media-optimized) models for higher accuracy in classifying sentiments as Positive, Negative, or Neutral.

### 🔍Multi-Party Comparison
Compare sentiment trends across multiple political parties simultaneously, with visualizations highlighting which party has the most favorable public perception.

### 📈 Interactive Visualizations
Pie charts for sentiment distribution
Trend graphs showing sentiment over time
Comparative bar charts for side-by-side analysis

### 🌐Multilingual Support 
Processes text in English, Tamil, and Thanglish (Tamil written in English script) by automatically detecting and translating non-English content.



### ⚡Real-Time Processing
Analyzes up to 5,000 tweets per party with a progress-tracking dashboard for large datasets.


## Installation & Setup

1.Install dependencies 
```bash
pip install -r requirements.txt
```

2.Download NLTK datasets (required for text processing)

3.Run the app 
```bash
streamlit run election_sentiment_app.py
```

4.Access the dashboard at http://localhost:8501.


## How It Works

### 1.Upload CSV files containing political party data (must include a text column).

### 2.Configure analysis:
  Choose between Hybrid, TextBlob-only, or VADER-only sentiment analysis.
  Adjust the maximum number of tweets to process.
  
### 3.View results:
  Winning party prediction (highest positive sentiment).
  Party-wise sentiment breakdown.
  Time-based trends (if date column is provided).

## 📸 Screenshots & Walkthrough

### 1. **Dashboard Landing Page**
![Dashboard Interface](./assets/Screenshot%202025-05-21%20211641.png)  
*The clean interface showing analysis configuration options and file upload areas for multiple parties.*

---

### 2. **Data Upload Process**
![File Upload](./assets/Screenshot%202025-05-21%20211751.png)  
*Drag-and-drop CSV upload for 3 political parties. Supports files up to 200MB each.*

---

### 3. **Analysis Configuration**
![Settings Panel](./assets/Screenshot%202025-05-21%20211700.png)  
*Configure hybrid sentiment analysis (TextBlob + VADER) and set tweet processing limits (100-5000 tweets/party).*

---

### 4. **Real-Time Results**
![Analysis Output](./assets/Screenshot%202025-05-21%20211834.png)  
*Automatically detects winning party (Party 3 with 46.5% positive sentiment) and shows individual party breakdowns.*

---

### 5. **Comparative Visualization**
![Multi-Party Comparison](./assets/Screenshot%202025-05-21%20211915.png)  
*Interactive bar chart comparing sentiment distribution across all uploaded parties.*



## 👥 Authors/Acknowledgments



- Iniya Vasanthan V M 
- Praveen S
- Kishore S
 
