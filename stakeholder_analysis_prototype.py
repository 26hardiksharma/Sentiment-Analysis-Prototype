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
    """Generate 10,000 realistic sample stakeholder comments with diverse patterns"""
    import random
    from datetime import timedelta
    
    # Comprehensive base comment templates covering various aspects
    positive_templates = [
        "The proposed amendments to corporate governance are excellent and will improve transparency significantly.",
        "This is a commendable initiative that will strengthen corporate accountability in India.",
        "Fantastic work by the ministry! This legislation addresses long-standing concerns effectively.",
        "The provisions for {topic} are well-drafted and practical for implementation.",
        "This represents a significant step forward for corporate regulation in our country.",
        "Excellent balance between regulation and business flexibility in this draft.",
        "The {topic} section demonstrates thorough research and stakeholder consultation.",
        "This legislation will significantly benefit {beneficiary} and improve market confidence.",
        "Comprehensive and well-structured approach to addressing corporate governance issues.",
        "The enforcement mechanisms proposed are robust and will ensure effective compliance.",
        "This draft successfully addresses the gaps in existing corporate legislation.",
        "The clarity in language and structure makes this legislation highly implementable.",
        "Strong provisions for {topic} that align with international best practices.",
        "This initiative will enhance India's position in global corporate governance rankings.",
        "The phased implementation approach is pragmatic and business-friendly.",
        "Excellent integration of stakeholder feedback in the final draft.",
        "This legislation strikes the right balance between regulation and innovation.",
        "The provisions for digital compliance are forward-thinking and relevant.",
        "This will significantly reduce regulatory ambiguity for businesses.",
        "Comprehensive coverage of {topic} with clear implementation guidelines."
    ]
    
    negative_templates = [
        "This draft is completely unrealistic and will destroy small businesses in India.",
        "The proposed amendments are too burdensome for companies to implement effectively.",
        "This legislation will increase bureaucracy and corruption without solving real problems.",
        "The compliance costs associated with this draft are prohibitively expensive.",
        "This proposal lacks understanding of ground realities faced by businesses.",
        "The {topic} provisions are impractical and will harm economic growth.",
        "This draft will create unnecessary regulatory burden on {affected_group}.",
        "The implementation timeline is too aggressive and unrealistic for most companies.",
        "This legislation will drive businesses away from India to other jurisdictions.",
        "The penalties proposed are disproportionate and will harm business sentiment.",
        "This draft ignores the concerns raised by industry stakeholders.",
        "The {topic} section is confusing and contradictory in its requirements.",
        "This proposal will significantly increase operational costs for businesses.",
        "The lack of clarity in implementation will lead to regulatory uncertainty.",
        "This legislation is too complex and will create compliance nightmares.",
        "The provisions for {topic} are overly restrictive and innovation-killing.",
        "This draft will particularly harm startups and small enterprises.",
        "The regulatory framework proposed is outdated and not future-ready.",
        "This legislation will create more problems than it solves.",
        "The impact assessment clearly shows negative consequences for the economy."
    ]
    
    neutral_templates = [
        "The draft has both positive and negative aspects that need careful consideration.",
        "Some sections are well-drafted while others require significant improvement.",
        "Mixed feelings about this proposal - good intentions but implementation concerns.",
        "The {topic} provisions are reasonable but need more stakeholder consultation.",
        "This draft requires more detailed impact assessment before finalization.",
        "Some clarity needed on the practical implementation of these provisions.",
        "The legislation addresses important issues but the approach needs refinement.",
        "Moderate support for the objectives but concerns about the execution method.",
        "The {topic} section needs more detailed guidelines for effective implementation.",
        "This proposal is a step in the right direction but requires further work.",
        "The draft shows promise but needs significant amendments before passage.",
        "Balanced approach attempted but more stakeholder input is essential.",
        "The provisions for {topic} are adequate but could be more comprehensive.",
        "This legislation needs more time for proper consultation and refinement.",
        "Some good elements present but overall structure needs improvement.",
        "The intent is commendable but practical challenges need addressing.",
        "Requires careful balance between regulatory needs and business flexibility.",
        "The {topic} framework is sound but implementation details are lacking.",
        "This draft needs more empirical data to support its provisions.",
        "Generally supportive but several sections need clarification and improvement."
    ]
    
    # Topics to be substituted in templates
    topics = [
        "corporate social responsibility", "audit requirements", "board composition",
        "financial reporting", "environmental compliance", "digital governance",
        "stakeholder engagement", "risk management", "internal controls",
        "executive compensation", "related party transactions", "insider trading",
        "whistleblower protection", "data privacy", "cybersecurity measures",
        "subsidiary governance", "foreign investment", "merger and acquisition",
        "corporate restructuring", "taxation compliance", "labor relations",
        "intellectual property", "supply chain management", "sustainability reporting",
        "corporate ethics", "anti-corruption measures", "transparency requirements"
    ]
    
    # Beneficiaries and affected groups
    beneficiaries = ["investors", "shareholders", "employees", "consumers", "small businesses",
                    "the public", "regulatory authorities", "market participants", "stakeholders"]
    
    affected_groups = ["small enterprises", "startups", "multinational corporations", "family businesses",
                      "listed companies", "private companies", "financial institutions", "service providers"]
    
    # Comprehensive geographical coverage - All Indian states and union territories
    regions = [
        'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh', 'Goa', 'Gujarat',
        'Haryana', 'Himachal Pradesh', 'Jharkhand', 'Karnataka', 'Kerala', 'Madhya Pradesh',
        'Maharashtra', 'Manipur', 'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Punjab',
        'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana', 'Tripura', 'Uttar Pradesh',
        'Uttarakhand', 'West Bengal', 'Delhi', 'Jammu and Kashmir', 'Ladakh', 'Chandigarh',
        'Dadra and Nagar Haveli', 'Daman and Diu', 'Lakshadweep', 'Puducherry', 'Andaman and Nicobar'
    ]
    
    # Diverse age groups with realistic distribution
    age_groups = ['18-25', '25-35', '35-45', '45-55', '55-65', '65+']
    age_weights = [0.15, 0.30, 0.25, 0.20, 0.08, 0.02]  # Realistic age distribution
    
    # Comprehensive stakeholder types
    stakeholder_types = [
        'Corporate Executive', 'Small Business Owner', 'Legal Professional', 'Policy Expert',
        'Tax Consultant', 'Auditor', 'NGO Representative', 'Industry Association',
        'Startup Founder', 'Academic', 'Investor', 'Financial Analyst', 'Compliance Officer',
        'Trade Union Leader', 'Consumer Rights Activist', 'Government Official',
        'Chartered Accountant', 'Investment Banker', 'Management Consultant', 'Company Secretary',
        'Risk Manager', 'Internal Auditor', 'Corporate Lawyer', 'Regulatory Expert',
        'Business Journalist', 'Economic Researcher', 'Pension Fund Manager', 'Retail Investor',
        'Institutional Investor', 'Credit Rating Analyst', 'Banking Professional',
        'Insurance Executive', 'Mutual Fund Manager', 'Private Equity Professional',
        'Venture Capitalist', 'Financial Advisor', 'Tax Expert', 'Environmental Consultant',
        'Technology Consultant', 'HR Professional', 'Supply Chain Expert', 'Procurement Specialist'
    ]
    
    # Sentiment modifiers and variations
    positive_modifiers = ["excellent", "fantastic", "outstanding", "commendable", "impressive",
                         "remarkable", "exceptional", "superb", "brilliant", "wonderful"]
    
    negative_modifiers = ["terrible", "awful", "disastrous", "concerning", "problematic",
                         "troubling", "alarming", "disappointing", "inadequate", "unacceptable"]
    
    neutral_modifiers = ["reasonable", "adequate", "moderate", "balanced", "fair",
                        "acceptable", "decent", "satisfactory", "average", "standard"]
    
    # Additional sentence variations
    positive_additions = [
        "This will significantly benefit the business ecosystem.",
        "The implementation framework is well thought out.",
        "This addresses long-standing industry concerns effectively.",
        "The consultation process has been comprehensive and inclusive.",
        "This will enhance India's regulatory framework substantially.",
        "The provisions align well with international best practices.",
        "This legislation will boost investor confidence significantly.",
        "The phased approach allows for smooth implementation.",
        "This will reduce regulatory uncertainty for businesses.",
        "The enforcement mechanisms are robust and fair."
    ]
    
    negative_additions = [
        "This will create unnecessary compliance burden on businesses.",
        "The cost of implementation will be prohibitively high.",
        "This lacks practical understanding of business operations.",
        "The timeline for implementation is unreasonably tight.",
        "This will harm the ease of doing business in India.",
        "The provisions are too rigid and lack flexibility.",
        "This will disproportionately affect small businesses.",
        "The regulatory complexity will increase manifold.",
        "This needs major revision before implementation.",
        "The industry feedback has not been adequately considered."
    ]
    
    neutral_additions = [
        "More stakeholder consultation would be beneficial.",
        "The implementation guidelines need further clarification.",
        "Some provisions require additional refinement.",
        "The impact assessment should be more comprehensive.",
        "More time is needed for proper evaluation.",
        "Additional clarity on compliance requirements is needed.",
        "The enforcement mechanism needs strengthening.",
        "More detailed operational guidelines would help.",
        "Further consultation with industry experts is recommended.",
        "The provisions need to be more specific and actionable."
    ]
    
    # Generate 10,000 comments
    sample_comments = []
    base_date = datetime.now() - timedelta(days=90)  # Spread over 3 months
    
    print("Generating 10,000 sample comments... This may take a moment.")
    
    for i in range(10000):
        if i % 1000 == 0:
            print(f"Generated {i} comments...")
        
        # Determine sentiment distribution (35% positive, 40% negative, 25% neutral)
        sentiment_rand = random.random()
        if sentiment_rand < 0.35:
            template = random.choice(positive_templates)
            additions = positive_additions
            modifiers = positive_modifiers
        elif sentiment_rand < 0.75:
            template = random.choice(negative_templates)
            additions = negative_additions
            modifiers = negative_modifiers
        else:
            template = random.choice(neutral_templates)
            additions = neutral_additions
            modifiers = neutral_modifiers
        
        # Substitute placeholders in template
        comment_text = template
        if '{topic}' in comment_text:
            comment_text = comment_text.replace('{topic}', random.choice(topics))
        if '{beneficiary}' in comment_text:
            comment_text = comment_text.replace('{beneficiary}', random.choice(beneficiaries))
        if '{affected_group}' in comment_text:
            comment_text = comment_text.replace('{affected_group}', random.choice(affected_groups))
        
        # Add variations to make comments unique
        if random.random() < 0.4:  # 40% chance to add additional sentence
            comment_text += " " + random.choice(additions)
        
        if random.random() < 0.3:  # 30% chance to add a concluding thought
            conclusions = [
                "Hope the ministry considers this feedback seriously.",
                "Looking forward to seeing the final implementation.",
                "This requires immediate attention from policymakers.",
                "The industry is watching this development closely.",
                "Public consultation on this matter is essential.",
                "This will have long-term implications for the economy.",
                "Proper implementation is crucial for success.",
                "The stakeholder community needs more clarity.",
                "This deserves careful consideration before finalization.",
                "The government should act on this feedback promptly."
            ]
            comment_text += " " + random.choice(conclusions)
        
        # Occasionally modify adjectives for variety
        if random.random() < 0.15:
            for original, replacement in zip(
                ["excellent", "terrible", "good", "bad", "great", "poor"],
                [random.choice(positive_modifiers), random.choice(negative_modifiers), 
                 "decent", "concerning", "impressive", "inadequate"]
            ):
                comment_text = comment_text.replace(original, replacement)
        
        # Random timestamp within the last 90 days with realistic distribution
        # More recent comments are more likely
        days_back = int(random.expovariate(1/20))  # Exponential distribution
        if days_back > 90:
            days_back = random.randint(0, 90)
        
        timestamp = base_date + timedelta(
            days=days_back,
            hours=random.randint(9, 18),  # Business hours bias
            minutes=random.randint(0, 59)
        )
        
        # Select demographics with realistic distribution
        selected_age = random.choices(age_groups, weights=age_weights)[0]
        
        sample_comments.append({
            'comment': comment_text,
            'region': random.choice(regions),
            'age_group': selected_age,
            'stakeholder_type': random.choice(stakeholder_types),
            'timestamp': timestamp,
            'id': i + 1
        })
    
    print("Generated 10,000 comments successfully!")
    return sample_comments

def main():
    st.set_page_config(
        page_title="MCA Stakeholder Analysis System",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("🏛️ MCA Stakeholder Comment Analysis System")
    st.markdown("### AI-Powered Analysis of Draft Legislation Feedback")
    
    # Initialize session state for pagination
    if 'page_num' not in st.session_state:
        st.session_state.page_num = 1
    if 'comments_per_page' not in st.session_state:
        st.session_state.comments_per_page = 100
    
    # Initialize the analysis system
    analysis_system = StakeholderAnalysisSystem()
    
    # Add caching for data generation and processing
    @st.cache_data
    def get_processed_comments():
        """Generate and process comments with caching for performance"""
        # Generate sample data
        with st.spinner("Generating 10,000 sample comments... This may take a moment."):
            comments_data = generate_sample_data()
        
        # Process all comments with progress bar
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_comments = len(comments_data)
        batch_size = 1000  # Process in batches for better performance
        
        for i in range(0, total_comments, batch_size):
            batch = comments_data[i:i + batch_size]
            batch_results = []
            
            for comment_data in batch:
                sentiment_result = analysis_system.analyze_sentiment(comment_data['comment'])
                summary = analysis_system.generate_summary(comment_data['comment'])
                
                batch_results.append({
                    **comment_data,
                    **sentiment_result,
                    'summary': summary
                })
            
            results.extend(batch_results)
            
            # Update progress
            progress = min((i + batch_size) / total_comments, 1.0)
            progress_bar.progress(progress)
            status_text.text(f"Processing comments: {min(i + batch_size, total_comments):,} / {total_comments:,}")
        
        progress_bar.empty()
        status_text.empty()
        
        return results
    
    # Get processed data (cached for performance)
    with st.spinner("Loading comments data..."):
        results = get_processed_comments()
    
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
        st.metric("Total Comments", f"{len(filtered_df):,}")

    with col2:
        positive_pct = (filtered_df['sentiment'] == 'Positive').mean() * 100
        st.metric("Positive Sentiment", f"{positive_pct:.1f}%")

    with col3:
        negative_pct = (filtered_df['sentiment'] == 'Negative').mean() * 100
        st.metric("Negative Sentiment", f"{negative_pct:.1f}%")

    with col4:
        avg_compound = filtered_df['compound'].mean()
        st.metric("Avg. Sentiment Score", f"{avg_compound:.2f}")
    
    # Data scale information
    st.info(f"📊 **Large Scale Analysis**: Currently analyzing {len(df):,} total comments across {len(df['region'].unique())} regions and {len(df['stakeholder_type'].unique())} stakeholder types. Data spans {(df['timestamp'].max() - df['timestamp'].min()).days} days of feedback collection.")

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

    # Word Cloud (optimized for large datasets)
    st.subheader("💭 Word Cloud of Comments")
    
    # For performance, use a sample if dataset is very large
    wordcloud_sample_size = min(5000, len(filtered_df))
    if len(filtered_df) > 5000:
        st.info(f"💡 Using a representative sample of {wordcloud_sample_size:,} comments for word cloud generation (from {len(filtered_df):,} total filtered comments)")
        wordcloud_data = filtered_df.sample(n=wordcloud_sample_size, random_state=42)['comment'].tolist()
    else:
        wordcloud_data = filtered_df['comment'].tolist()
    
    with st.spinner("Generating word cloud..."):
        wordcloud = analysis_system.create_wordcloud(wordcloud_data)

    fig_wc, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig_wc)

    # Topic Analysis (optimized for large datasets)
    st.subheader("🎯 Key Topics Identified")
    
    # For performance, use a sample if dataset is very large
    topic_sample_size = min(10000, len(filtered_df))
    if len(filtered_df) > 10000:
        st.info(f"💡 Analyzing topics from a representative sample of {topic_sample_size:,} comments (from {len(filtered_df):,} total filtered comments)")
        topic_data = filtered_df.sample(n=topic_sample_size, random_state=42)['comment'].tolist()
    else:
        topic_data = filtered_df['comment'].tolist()
    
    with st.spinner("Extracting key topics..."):
        topics = analysis_system.extract_key_topics(topic_data)
    
    for i, topic in enumerate(topics, 1):
        st.write(f"**{topic}**")

    # Comments Table with Pagination
    st.subheader("💬 Detailed Comments Analysis")
    
    # Calculate pagination
    total_comments = len(filtered_df)
    total_pages = (total_comments - 1) // st.session_state.comments_per_page + 1 if total_comments > 0 else 1
    
    # Ensure page number is within valid range
    if st.session_state.page_num > total_pages:
        st.session_state.page_num = total_pages
    if st.session_state.page_num < 1:
        st.session_state.page_num = 1
    
    # Calculate start and end indices for current page
    start_idx = (st.session_state.page_num - 1) * st.session_state.comments_per_page
    end_idx = min(start_idx + st.session_state.comments_per_page, total_comments)
    
    # Display page information
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("⬅️ Previous Page", disabled=(st.session_state.page_num <= 1)):
            st.session_state.page_num -= 1
            st.rerun()
    
    with col2:
        st.markdown(f"<div style='text-align: center'><strong>Page {st.session_state.page_num} of {total_pages}</strong><br>Showing comments {start_idx + 1}-{end_idx} of {total_comments}</div>", unsafe_allow_html=True)
    
    with col3:
        if st.button("Next Page ➡️", disabled=(st.session_state.page_num >= total_pages)):
            st.session_state.page_num += 1
            st.rerun()
    
    st.markdown("---")
    
    # Display comments for current page only
    if total_comments > 0:
        current_page_df = filtered_df.iloc[start_idx:end_idx]
        
        for _, row in current_page_df.iterrows():
            with st.expander(f"Comment {row['id']} - {row['sentiment']} ({row['region']})"):
                st.write(f"**Original Comment:** {row['comment']}")
                st.write(f"**Summary:** {row['summary']}")
                st.write(f"**Sentiment Score:** {row['compound']:.3f}")
                st.write(f"**Stakeholder:** {row['stakeholder_type']} | **Age Group:** {row['age_group']}")
    else:
        st.info("No comments match the selected filters.")
    
    # Add pagination controls at the bottom as well
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("⬅️ Previous", key="prev_bottom", disabled=(st.session_state.page_num <= 1)):
            st.session_state.page_num -= 1
            st.rerun()
    
    with col2:
        # Page number selector
        page_options = list(range(1, total_pages + 1))
        selected_page = st.selectbox(
            "Jump to page:",
            options=page_options,
            index=st.session_state.page_num - 1,
            key="page_selector"
        )
        if selected_page != st.session_state.page_num:
            st.session_state.page_num = selected_page
            st.rerun()
    
    with col3:
        if st.button("Next ➡️", key="next_bottom", disabled=(st.session_state.page_num >= total_pages)):
            st.session_state.page_num += 1
            st.rerun()

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