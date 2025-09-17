"""
AI-Powered Stakeholder Comment Analysis System
Prototype for Smart India Hackathon

This prototype demonstrates automated analysis of stakeholder comments on draft legislation
for the Ministry of Corporate Affairs (MCA).

Features:
- Sentiment Analysis (VADER)
- Text Summarization
- Word Cloud Generation
- Regional Analysis
- Topic Modeling
- Interactive Dashboard
"""

import nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re
from datetime import datetime, timedelta
import random
import numpy as np

# Download required NLTK data
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

class StakeholderAnalysisSystem:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess_text(self, text):
        """Clean and preprocess text for analysis"""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Tokenize
        tokens = text.split()
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]
        return ' '.join(tokens)
    
    def analyze_sentiment(self, text):
        """Perform sentiment analysis using VADER"""
        scores = self.analyzer.polarity_scores(text)
        
        # Classify sentiment
        if scores['compound'] >= 0.05:
            sentiment = "Positive"
        elif scores['compound'] <= -0.05:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
            
        return {
            'sentiment': sentiment,
            'compound': scores['compound'],
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu']
        }
    
    def extract_key_topics(self, texts, n_topics=5):
        """Extract key topics using LDA"""
        # Preprocess texts
        preprocessed = [self.preprocess_text(text) for text in texts]
        
        # Vectorize
        vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
        doc_term_matrix = vectorizer.fit_transform(preprocessed)
        
        # LDA
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda.fit(doc_term_matrix)
        
        # Extract topics
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            top_words = [feature_names[i] for i in topic.argsort()[-10:]]
            topics.append(f"Topic {topic_idx + 1}: {', '.join(top_words[-5:])}")
        
        return topics
    
    def generate_summary(self, text, max_sentences=2):
        """Generate extractive summary"""
        sentences = text.split('.')
        if len(sentences) <= max_sentences:
            return text
        
        # Simple extractive summarization based on sentence length and position
        sentence_scores = []
        for i, sentence in enumerate(sentences):
            score = len(sentence.split()) * (1 + 1/(i+1))  # Length + position weight
            sentence_scores.append((score, sentence.strip()))
        
        # Get top sentences
        top_sentences = sorted(sentence_scores, reverse=True)[:max_sentences]
        summary = '. '.join([sent[1] for sent in top_sentences if sent[1]])
        return summary + '.' if summary else text
    
    def create_wordcloud(self, texts):
        """Generate word cloud from texts"""
        combined_text = ' '.join([self.preprocess_text(text) for text in texts])
        
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            colormap='viridis',
            max_words=100
        ).generate(combined_text)
        
        return wordcloud

def generate_sample_data():
    """Generate 1000 realistic sample stakeholder comments"""
    import random
    from datetime import timedelta
    
    # Base comment templates
    base_comments = [
        {
            'comment': "The proposed amendments to corporate governance are excellent and will improve transparency significantly. However, the compliance burden on small companies needs reconsideration.",
            'region': 'Maharashtra',
            'age_group': '35-45',
            'stakeholder_type': 'Corporate Executive'
        },
        {
            'comment': "This draft is completely unrealistic and will destroy small businesses. The government doesn't understand ground realities at all.",
            'region': 'Gujarat',
            'age_group': '45-55',
            'stakeholder_type': 'Small Business Owner'
        },
        {
            'comment': "Good initiative overall. The environmental compliance section is particularly well-drafted. Some clarity needed on implementation timelines.",
            'region': 'Karnataka',
            'age_group': '25-35',
            'stakeholder_type': 'Legal Professional'
        },
        {
            'comment': "Absolutely fantastic work by the ministry! This will revolutionize corporate accountability in India. Fully support all provisions.",
            'region': 'Delhi',
            'age_group': '55-65',
            'stakeholder_type': 'Policy Expert'
        },
        {
            'comment': "The taxation clauses are confusing and contradictory. Need major revisions before implementation. Current form is not acceptable.",
            'region': 'Tamil Nadu',
            'age_group': '35-45',
            'stakeholder_type': 'Tax Consultant'
        },
        {
            'comment': "Mixed feelings about this draft. Some sections are good but others need improvement. The audit requirements seem excessive.",
            'region': 'West Bengal',
            'age_group': '45-55',
            'stakeholder_type': 'Auditor'
        },
        {
            'comment': "This is a step in the right direction. The corporate social responsibility provisions are well thought out and practical.",
            'region': 'Rajasthan',
            'age_group': '25-35',
            'stakeholder_type': 'NGO Representative'
        },
        {
            'comment': "Terrible draft! Will increase bureaucracy and corruption. The ministry should withdraw this immediately and start fresh.",
            'region': 'Punjab',
            'age_group': '55-65',
            'stakeholder_type': 'Industry Association'
        },
        {
            'comment': "The provisions for startups are encouraging. However, more flexibility is needed for emerging technology companies.",
            'region': 'Haryana',
            'age_group': '25-35',
            'stakeholder_type': 'Startup Founder'
        },
        {
            'comment': "Well-balanced approach to corporate regulation. The enforcement mechanisms are strong and the penalties are appropriate.",
            'region': 'Telangana',
            'age_group': '35-45',
            'stakeholder_type': 'Academic'
        }
    ]
    
    # Additional regions and stakeholder types for variety
    regions = ['Maharashtra', 'Gujarat', 'Karnataka', 'Delhi', 'Tamil Nadu', 'West Bengal', 
               'Rajasthan', 'Punjab', 'Haryana', 'Telangana', 'Uttar Pradesh', 'Madhya Pradesh',
               'Andhra Pradesh', 'Odisha', 'Kerala', 'Assam', 'Bihar', 'Jharkhand', 'Uttarakhand', 'Goa']
    
    age_groups = ['18-25', '25-35', '35-45', '45-55', '55-65', '65+']
    
    stakeholder_types = ['Corporate Executive', 'Small Business Owner', 'Legal Professional', 
                        'Policy Expert', 'Tax Consultant', 'Auditor', 'NGO Representative',
                        'Industry Association', 'Startup Founder', 'Academic', 'Investor',
                        'Financial Analyst', 'Compliance Officer', 'Trade Union Leader',
                        'Consumer Rights Activist', 'Government Official', 'Chartered Accountant']
    
    # Generate 1000 comments
    sample_comments = []
    base_date = datetime.now() - timedelta(days=30)
    
    for i in range(1000):
        # Pick a random base comment
        base = random.choice(base_comments)
        comment_text = base['comment']
        
        # Add variations to make comments unique
        variations = [
            " This needs immediate attention.",
            " We strongly support this initiative.", 
            " More consultation is needed with stakeholders.",
            " The implementation timeline seems unrealistic.",
            " This will benefit the industry significantly.",
            " Concerns about practical implementation remain.",
            " The ministry should consider feedback carefully.",
            " This represents a positive step forward.",
            " Additional clarity on compliance is required.",
            " We appreciate the government's efforts on this matter."
        ]
        
        if random.random() < 0.3:  # 30% chance to add variation
            comment_text += random.choice(variations)
        
        # Occasionally modify key words for variety
        if random.random() < 0.1:
            comment_text = comment_text.replace("excellent", "very good")
        if random.random() < 0.1:
            comment_text = comment_text.replace("terrible", "concerning")
        if random.random() < 0.1:
            comment_text = comment_text.replace("fantastic", "impressive")
        
        # Random timestamp within the last 30 days
        timestamp = base_date + timedelta(
            days=random.randint(0, 30),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )
        
        sample_comments.append({
            'comment': comment_text,
            'region': random.choice(regions),
            'age_group': random.choice(age_groups),
            'stakeholder_type': random.choice(stakeholder_types),
            'timestamp': timestamp,
            'id': i + 1
        })
    
    return sample_comments

def main():
    st.set_page_config(
        page_title="MCA Stakeholder Analysis System",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("🏛️ MCA Stakeholder Comment Analysis System")
    st.markdown("### AI-Powered Analysis of Draft Legislation Feedback")
    
    # Initialize the analysis system
    analysis_system = StakeholderAnalysisSystem()
    
    # Generate sample data
    comments_data = generate_sample_data()
    
    # Process all comments
    results = []
    for comment_data in comments_data:
        sentiment_result = analysis_system.analyze_sentiment(comment_data['comment'])
        summary = analysis_system.generate_summary(comment_data['comment'])
        
        results.append({
            **comment_data,
            **sentiment_result,
            'summary': summary
        })
    
    df = pd.DataFrame(results)
    
    # Sidebar for filters
    st.sidebar.header("📋 Filters")
    
    selected_regions = st.sidebar.multiselect(
        "Select Regions",
        options=df['region'].unique(),
        default=df['region'].unique()
    )
    
    selected_sentiments = st.sidebar.multiselect(
        "Select Sentiments",
        options=df['sentiment'].unique(),
        default=df['sentiment'].unique()
    )
    
    # Filter data
    filtered_df = df[
        (df['region'].isin(selected_regions)) & 
        (df['sentiment'].isin(selected_sentiments))
    ]
    
    # Main dashboard
    if filtered_df.empty:
        st.warning("No data available for the selected filters. Please adjust your region or sentiment selections in the sidebar.")
        return

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Comments", len(filtered_df))

    with col2:
        positive_pct = (filtered_df['sentiment'] == 'Positive').mean() * 100
        st.metric("Positive Sentiment", f"{positive_pct:.1f}%")

    with col3:
        negative_pct = (filtered_df['sentiment'] == 'Negative').mean() * 100
        st.metric("Negative Sentiment", f"{negative_pct:.1f}%")

    with col4:
        avg_compound = filtered_df['compound'].mean()
        st.metric("Avg. Sentiment Score", f"{avg_compound:.2f}")

    # Visualizations
    st.header("📈 Analytics Dashboard")

    # Sentiment Distribution
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sentiment Distribution")
        sentiment_counts = filtered_df['sentiment'].value_counts()
        fig_pie = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            color_discrete_map={
                'Positive': '#2E8B57',
                'Negative': '#DC143C',
                'Neutral': '#4682B4'
            }
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.subheader("Regional Sentiment Analysis")
        regional_sentiment = filtered_df.groupby(['region', 'sentiment']).size().unstack(fill_value=0)
        fig_regional = px.bar(
            regional_sentiment,
            title="Sentiment by Region",
            color_discrete_map={
                'Positive': '#2E8B57',
                'Negative': '#DC143C',
                'Neutral': '#4682B4'
            }
        )
        fig_regional.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_regional, use_container_width=True)

    # Age Group Analysis
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sentiment by Age Group")
        age_sentiment = filtered_df.groupby(['age_group', 'sentiment']).size().unstack(fill_value=0)
        fig_age = px.bar(
            age_sentiment,
            title="Sentiment Distribution by Age Group",
            color_discrete_map={
                'Positive': '#2E8B57',
                'Negative': '#DC143C',
                'Neutral': '#4682B4'
            }
        )
        st.plotly_chart(fig_age, use_container_width=True)

    with col2:
        st.subheader("Stakeholder Type Analysis")
        stakeholder_sentiment = filtered_df.groupby(['stakeholder_type', 'sentiment']).size().unstack(fill_value=0)
        fig_stakeholder = px.bar(
            stakeholder_sentiment,
            title="Sentiment by Stakeholder Type",
            color_discrete_map={
                'Positive': '#2E8B57',
                'Negative': '#DC143C',
                'Neutral': '#4682B4'
            }
        )
        fig_stakeholder.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_stakeholder, use_container_width=True)

    # Word Cloud
    st.subheader("💭 Word Cloud of Comments")
    wordcloud = analysis_system.create_wordcloud(filtered_df['comment'].tolist())

    fig_wc, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig_wc)

    # Topic Analysis
    st.subheader("🎯 Key Topics Identified")
    topics = analysis_system.extract_key_topics(filtered_df['comment'].tolist())
    for i, topic in enumerate(topics, 1):
        st.write(f"**{topic}**")

    # Comments Table
    st.subheader("💬 Detailed Comments Analysis")

    # Display comments with sentiment and summary
    for _, row in filtered_df.iterrows():
        with st.expander(f"Comment {row['id']} - {row['sentiment']} ({row['region']})"):
            st.write(f"**Original Comment:** {row['comment']}")
            st.write(f"**Summary:** {row['summary']}")
            st.write(f"**Sentiment Score:** {row['compound']:.3f}")
            st.write(f"**Stakeholder:** {row['stakeholder_type']} | **Age Group:** {row['age_group']}")

    # Export functionality
    st.header("📤 Export Results")

    if st.button("Generate Analysis Report"):
        report_data = {
            'Total Comments': len(filtered_df),
            'Positive Comments': len(filtered_df[filtered_df['sentiment'] == 'Positive']),
            'Negative Comments': len(filtered_df[filtered_df['sentiment'] == 'Negative']),
            'Neutral Comments': len(filtered_df[filtered_df['sentiment'] == 'Neutral']),
            'Average Sentiment Score': filtered_df['compound'].mean(),
            'Most Active Region': filtered_df['region'].mode()[0],
            'Key Topics': topics
        }
        st.json(report_data)
        st.success("Report generated successfully!")

if __name__ == "__main__":
    # Check if running in Streamlit
    try:
        main()
    except Exception as e:
        # If not running in Streamlit or error occurred, run a basic demo
        print("MCA Stakeholder Analysis System - Basic Demo")
        print("=" * 50)
        print(f"Error in Streamlit: {e}")
        print("Running basic demo instead...")
        print()
        
        try:
            system = StakeholderAnalysisSystem()
            sample_data = generate_sample_data()
            
            print(f"Analyzing {len(sample_data)} sample comments...")
            print()
            
            for i, comment_data in enumerate(sample_data  , 1):
                print(f"Comment {i}:")
                print(f"Text: {comment_data['comment']}")
                
                sentiment = system.analyze_sentiment(comment_data['comment'])
                print(f"Sentiment: {sentiment['sentiment']} (Score: {sentiment['compound']:.3f})")
                
                summary = system.generate_summary(comment_data['comment'])
                print(f"Summary: {summary}")
                print(f"Region: {comment_data['region']}, Age: {comment_data['age_group']}")
                print("-" * 50)
        except Exception as demo_error:
            print(f"Error in demo: {demo_error}")
            print("Please install missing packages with:")
            print("python3.12 -m pip install nltk pandas matplotlib wordcloud plotly streamlit scikit-learn")