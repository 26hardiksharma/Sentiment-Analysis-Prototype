"""
Simple Command-Line Demo for MCA Stakeholder Analysis System
This is a lightweight version that doesn't require Streamlit for quick testing
"""

import json
from datetime import datetime
import os

# Simple sentiment analysis without external dependencies
class SimpleSentimentAnalyzer:
    def __init__(self):
        # Simple word-based sentiment scoring
        self.positive_words = {
            'excellent', 'good', 'great', 'fantastic', 'wonderful', 'amazing', 
            'perfect', 'outstanding', 'superb', 'brilliant', 'effective',
            'support', 'approve', 'beneficial', 'helpful', 'positive'
        }
        
        self.negative_words = {
            'terrible', 'bad', 'awful', 'horrible', 'disaster', 'worst',
            'hate', 'destroy', 'unrealistic', 'confusing', 'excessive',
            'contradictory', 'unacceptable', 'bureaucracy', 'corruption'
        }
    
    def analyze_sentiment(self, text):
        words = text.lower().split()
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        
        if positive_count > negative_count:
            sentiment = "Positive"
            score = 0.1 + (positive_count - negative_count) * 0.1
        elif negative_count > positive_count:
            sentiment = "Negative" 
            score = -0.1 - (negative_count - positive_count) * 0.1
        else:
            sentiment = "Neutral"
            score = 0.0
            
        return {
            'sentiment': sentiment,
            'compound': max(-1.0, min(1.0, score)),  # Clamp to [-1, 1]
            'positive': positive_count / len(words) if words else 0,
            'negative': negative_count / len(words) if words else 0
        }

def generate_sample_comments():
    """Generate 1000 sample stakeholder comments by randomizing templates"""
    import random
    from datetime import timedelta
    templates = [
        {
            'comment': "The proposed amendments to corporate governance are excellent and will improve transparency significantly. However, the compliance burden on small companies needs reconsideration.",
            'region': 'Maharashtra',
            'age_group': '35-45',
            'stakeholder_type': 'Corporate Executive',
        },
        {
            'comment': "This draft is completely unrealistic and will destroy small businesses. The government doesn't understand ground realities at all.",
            'region': 'Gujarat',
            'age_group': '45-55',
            'stakeholder_type': 'Small Business Owner',
        },
        {
            'comment': "Good initiative overall. The environmental compliance section is particularly well-drafted. Some clarity needed on implementation timelines.",
            'region': 'Karnataka',
            'age_group': '25-35',
            'stakeholder_type': 'Legal Professional',
        },
        {
            'comment': "Absolutely fantastic work by the ministry! This will revolutionize corporate accountability in India. Fully support all provisions.",
            'region': 'Delhi',
            'age_group': '55-65',
            'stakeholder_type': 'Policy Expert',
        },
        {
            'comment': "The taxation clauses are confusing and contradictory. Need major revisions before implementation. Current form is not acceptable.",
            'region': 'Tamil Nadu',
            'age_group': '35-45',
            'stakeholder_type': 'Tax Consultant',
        }
    ]
    base_date = datetime(2025, 9, 10, 8, 0, 0)
    comments = []
    for i in range(1000):
        t = random.choice(templates)
        # Slightly vary the comment for realism
        comment_text = t['comment']
        if random.random() < 0.2:
            comment_text += " (Submitted via eConsultation)"
        if random.random() < 0.1:
            comment_text = comment_text.replace("excellent", "very good")
        if random.random() < 0.1:
            comment_text = comment_text.replace("unrealistic", "not practical")
        # Randomize timestamp
        timestamp = base_date + timedelta(days=random.randint(0, 14), hours=random.randint(0, 23), minutes=random.randint(0, 59))
        comments.append({
            'id': i + 1,
            'comment': comment_text,
            'region': t['region'],
            'age_group': t['age_group'],
            'stakeholder_type': t['stakeholder_type'],
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S')
        })
    return comments

def simple_summary(text, max_sentences=2):
    """Simple extractive summarization"""
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    if len(sentences) <= max_sentences:
        return text
    
    # Return first and last sentences for simplicity
    if len(sentences) >= 2:
        return f"{sentences[0]}. {sentences[-1]}."
    return sentences[0] + "."

def analyze_comments():
    """Main analysis function"""
    print("🏛️  MCA STAKEHOLDER COMMENT ANALYSIS SYSTEM")
    print("=" * 60)
    print("AI-Powered Analysis of Draft Legislation Feedback")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = SimpleSentimentAnalyzer()
    
    # Get sample data
    comments = generate_sample_comments()
    
    # Process each comment
    results = []
    for comment_data in comments:
        sentiment_result = analyzer.analyze_sentiment(comment_data['comment'])
        summary = simple_summary(comment_data['comment'])
        
        result = {
            **comment_data,
            **sentiment_result,
            'summary': summary
        }
        results.append(result)
    
    # Display individual results
    print(f"\n📊 INDIVIDUAL COMMENT ANALYSIS")
    print("-" * 60)
    
    for i, result in enumerate(results, 1):
        print(f"\n🔍 COMMENT {i}")
        print(f"Region: {result['region']} | Age: {result['age_group']}")
        print(f"Stakeholder: {result['stakeholder_type']}")
        print(f"Timestamp: {result['timestamp']}")
        print(f"\nOriginal Comment:")
        print(f"  {result['comment']}")
        print(f"\nSentiment: {result['sentiment']} (Score: {result['compound']:.3f})")
        print(f"Summary: {result['summary']}")
        print("-" * 60)
    
    # Generate overall statistics
    print(f"\n📈 OVERALL ANALYSIS SUMMARY")
    print("=" * 60)
    
    total_comments = len(results)
    positive_count = sum(1 for r in results if r['sentiment'] == 'Positive')
    negative_count = sum(1 for r in results if r['sentiment'] == 'Negative')
    neutral_count = sum(1 for r in results if r['sentiment'] == 'Neutral')
    
    avg_sentiment = sum(r['compound'] for r in results) / total_comments
    
    print(f"Total Comments Analyzed: {total_comments}")
    print(f"Positive Comments: {positive_count} ({positive_count/total_comments*100:.1f}%)")
    print(f"Negative Comments: {negative_count} ({negative_count/total_comments*100:.1f}%)")
    print(f"Neutral Comments: {neutral_count} ({neutral_count/total_comments*100:.1f}%)")
    print(f"Average Sentiment Score: {avg_sentiment:.3f}")
    
    # Regional analysis
    print(f"\n🗺️  REGIONAL BREAKDOWN")
    print("-" * 30)
    regions = {}
    for result in results:
        region = result['region']
        if region not in regions:
            regions[region] = {'positive': 0, 'negative': 0, 'neutral': 0, 'total': 0}
        regions[region][result['sentiment'].lower()] += 1
        regions[region]['total'] += 1
    
    for region, stats in regions.items():
        pos_pct = stats['positive'] / stats['total'] * 100
        neg_pct = stats['negative'] / stats['total'] * 100
        print(f"{region}: {stats['total']} comments (↑{pos_pct:.0f}% ↓{neg_pct:.0f}%)")
    
    # Age group analysis
    print(f"\n👥 AGE GROUP BREAKDOWN")
    print("-" * 30)
    age_groups = {}
    for result in results:
        age = result['age_group']
        if age not in age_groups:
            age_groups[age] = {'positive': 0, 'negative': 0, 'neutral': 0, 'total': 0}
        age_groups[age][result['sentiment'].lower()] += 1
        age_groups[age]['total'] += 1
    
    for age, stats in age_groups.items():
        pos_pct = stats['positive'] / stats['total'] * 100
        neg_pct = stats['negative'] / stats['total'] * 100
        print(f"{age} years: {stats['total']} comments (↑{pos_pct:.0f}% ↓{neg_pct:.0f}%)")
    
    # Key insights
    print(f"\n💡 KEY INSIGHTS")
    print("-" * 30)
    
    most_positive_region = max(regions.items(), key=lambda x: x[1]['positive'])
    most_negative_region = max(regions.items(), key=lambda x: x[1]['negative'])
    
    print(f"• Most positive region: {most_positive_region[0]}")
    print(f"• Most critical region: {most_negative_region[0]}")
    print(f"• Overall sentiment: {'Positive' if avg_sentiment > 0.05 else 'Negative' if avg_sentiment < -0.05 else 'Neutral'}")
    
    # Export results
    export_data = {
        'analysis_date': datetime.now().isoformat(),
        'total_comments': total_comments,
        'sentiment_distribution': {
            'positive': positive_count,
            'negative': negative_count,
            'neutral': neutral_count
        },
        'average_sentiment_score': avg_sentiment,
        'regional_analysis': regions,
        'age_group_analysis': age_groups,
        'detailed_results': results
    }
    
    # Save to JSON file
    with open('analysis_results.json', 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results exported to 'analysis_results.json'")
    print("=" * 60)
    
    return results

if __name__ == "__main__":
    analyze_comments()
    
    print(f"\n🚀 HACKATHON DEMO COMPLETED!")
    print("To run the full interactive dashboard:")
    print("1. Install requirements: pip install -r requirements.txt")
    print("2. Run dashboard: streamlit run stakeholder_analysis_prototype.py")
    print("3. Open browser: http://localhost:8501")