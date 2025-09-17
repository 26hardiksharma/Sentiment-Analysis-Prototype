"""
Advanced Features Demo for MCA Stakeholder Analysis System
Demonstrates innovative features like anomaly detection, multilingual support simulation,
and automated response generation
"""

import re
import random
from datetime import datetime, timedelta
from collections import Counter
import json

class AdvancedAnalysisSystem:
    def __init__(self):
        self.positive_words = {
            'excellent', 'good', 'great', 'fantastic', 'wonderful', 'amazing', 
            'perfect', 'outstanding', 'superb', 'brilliant', 'effective',
            'support', 'approve', 'beneficial', 'helpful', 'positive', 'love'
        }
        
        self.negative_words = {
            'terrible', 'bad', 'awful', 'horrible', 'disaster', 'worst',
            'hate', 'destroy', 'unrealistic', 'confusing', 'excessive',
            'contradictory', 'unacceptable', 'bureaucracy', 'corruption'
        }
        
        # Sarcasm indicators
        self.sarcasm_patterns = [
            r'oh.*great',
            r'just what we needed',
            r'brilliant idea',
            r'perfect timing',
            r'exactly what.*wanted'
        ]
        
        # Spam indicators
        self.spam_patterns = [
            r'click here',
            r'amazing offer',
            r'call now',
            r'guaranteed',
            r'free money'
        ]
    
    def detect_sarcasm(self, text):
        """Detect potential sarcasm in text"""
        text_lower = text.lower()
        
        for pattern in self.sarcasm_patterns:
            if re.search(pattern, text_lower):
                return True
        
        # Check for exclamation marks with negative context
        if '!' in text and any(word in text_lower for word in ['not', 'never', 'no']):
            return True
            
        return False
    
    def detect_spam(self, text):
        """Detect potential spam content"""
        text_lower = text.lower()
        
        # Check spam patterns
        for pattern in self.spam_patterns:
            if re.search(pattern, text_lower):
                return True
        
        # Check for excessive capitalization
        caps_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        if caps_ratio > 0.3:
            return True
        
        # Check for excessive punctuation
        punct_ratio = sum(1 for c in text if c in '!?') / len(text) if text else 0
        if punct_ratio > 0.1:
            return True
            
        return False
    
    def detect_anomaly(self, comment, avg_length=100, avg_sentiment=0.0):
        """Detect anomalous comments"""
        anomalies = []
        
        # Length anomaly
        if len(comment) > avg_length * 3:
            anomalies.append("Unusually long comment")
        elif len(comment) < avg_length * 0.2:
            anomalies.append("Unusually short comment")
        
        # Sentiment analysis
        sentiment_result = self.analyze_sentiment_basic(comment)
        
        # Extreme sentiment
        if abs(sentiment_result['compound']) > 0.8:
            anomalies.append("Extreme sentiment detected")
        
        # Sarcasm detection
        if self.detect_sarcasm(comment):
            anomalies.append("Potential sarcasm detected")
        
        # Spam detection
        if self.detect_spam(comment):
            anomalies.append("Potential spam content")
        
        return anomalies
    
    def analyze_sentiment_basic(self, text):
        """Basic sentiment analysis"""
        words = text.lower().split()
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        
        # Check for sarcasm and invert if detected
        is_sarcastic = self.detect_sarcasm(text)
        if is_sarcastic:
            positive_count, negative_count = negative_count, positive_count
        
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
            'compound': max(-1.0, min(1.0, score)),
            'is_sarcastic': is_sarcastic
        }
    
    def simulate_language_detection(self, text):
        """Simulate language detection (normally would use real API)"""
        # Simple heuristic for demo purposes
        hindi_words = ['और', 'है', 'में', 'को', 'का', 'से', 'पर', 'यह', 'वह', 'जो']
        tamil_words = ['இது', 'அது', 'என்று', 'உள்ள', 'ஒரு', 'மற்றும்', 'இல்', 'அல்ல']
        
        if any(word in text for word in hindi_words):
            return "Hindi"
        elif any(word in text for word in tamil_words):
            return "Tamil"
        else:
            return "English"
    
    def generate_auto_response(self, comment, sentiment, stakeholder_type):
        """Generate automated response suggestions"""
        
        if sentiment == "Positive":
            responses = [
                f"Thank you for your positive feedback. We appreciate your support for the proposed amendments.",
                f"We are pleased that you find the draft beneficial. Your endorsement strengthens our confidence in these provisions.",
                f"Your positive input is valuable to us. We will continue to work towards implementing these improvements."
            ]
        elif sentiment == "Negative":
            responses = [
                f"Thank you for raising your concerns. We will carefully review the issues you've highlighted and consider necessary adjustments.",
                f"We appreciate your feedback and understand your concerns. The implementation committee will address these points in detail.",
                f"Your critical input helps us improve the draft. We are committed to addressing the challenges you've identified."
            ]
        else:  # Neutral
            responses = [
                f"Thank you for your balanced feedback. We value your suggestions for improvement.",
                f"We appreciate your constructive input. Your suggestions will be reviewed by the relevant committee.",
                f"Thank you for your thoughtful analysis. We will consider your recommendations during the revision process."
            ]
        
        # Customize based on stakeholder type
        if stakeholder_type == "Small Business Owner":
            responses = [r + " We understand the unique challenges faced by small businesses." for r in responses]
        elif stakeholder_type == "Legal Professional":
            responses = [r + " We value your legal expertise in this matter." for r in responses]
        
        return random.choice(responses)
    
    def extract_key_phrases(self, text):
        """Extract key phrases from text"""
        # Simple phrase extraction (in real implementation, would use NER)
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Remove common words
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'this', 'that', 'is', 'are', 'was', 'were'}
        meaningful_words = [word for word in words if word not in common_words and len(word) > 3]
        
        # Get most frequent phrases
        word_freq = Counter(meaningful_words)
        return word_freq.most_common(5)

def generate_advanced_sample_data():
    """Generate sample data with various edge cases for advanced analysis"""
    return [
        {
            'id': 1,
            'comment': "The proposed amendments to corporate governance are excellent and will improve transparency significantly. However, the compliance burden on small companies needs reconsideration. We support the overall direction but request flexibility for MSMEs.",
            'region': 'Maharashtra',
            'age_group': '35-45',
            'stakeholder_type': 'Corporate Executive',
            'language': 'English'
        },
        {
            'id': 2,
            'comment': "Oh great, another 50-page compliance form! Just what small businesses needed during these tough times. Brilliant idea to add more bureaucracy!",
            'region': 'Gujarat',
            'age_group': '45-55',
            'stakeholder_type': 'Small Business Owner',
            'language': 'English'
        },
        {
            'id': 3,
            'comment': "AMAZING OFFER!!! CLICK HERE FOR FREE BUSINESS REGISTRATION!!! GUARANTEED SUCCESS!!!",
            'region': 'Unknown',
            'age_group': 'Unknown',
            'stakeholder_type': 'Spam',
            'language': 'English'
        },
        {
            'id': 4,
            'comment': "The draft legislation demonstrates a comprehensive understanding of contemporary corporate governance challenges. The provisions relating to audit committee independence are particularly well-crafted, though I would recommend additional clarity regarding the transition period for existing non-compliant structures. The penalty framework appears proportionate and should serve as an effective deterrent against non-compliance.",
            'region': 'Delhi',
            'age_group': '55-65',
            'stakeholder_type': 'Legal Professional',
            'language': 'English'
        },
        {
            'id': 5,
            'comment': "Good",
            'region': 'Punjab',
            'age_group': '25-35',
            'stakeholder_type': 'Unknown',
            'language': 'English'
        },
        {
            'id': 6,
            'comment': "यह मसौदा बहुत अच्छा है लेकिन छोटे व्यवसायों के लिए कुछ समस्याएं हैं। अधिक स्पष्टता की जरूरत है।",
            'region': 'Uttar Pradesh',
            'age_group': '35-45',
            'stakeholder_type': 'Business Owner',
            'language': 'Hindi'
        }
    ]

def run_advanced_analysis():
    """Run comprehensive advanced analysis demo"""
    print("🚀 ADVANCED MCA STAKEHOLDER ANALYSIS SYSTEM")
    print("=" * 70)
    print("Innovative Features Demo")
    print("=" * 70)
    
    # Initialize system
    system = AdvancedAnalysisSystem()
    
    # Get advanced sample data
    comments = generate_advanced_sample_data()
    
    # Calculate baseline metrics
    avg_length = sum(len(c['comment']) for c in comments) / len(comments)
    
    print(f"\n📊 ADVANCED ANALYSIS RESULTS")
    print("-" * 70)
    
    for i, comment_data in enumerate(comments, 1):
        comment = comment_data['comment']
        
        print(f"\n🔍 COMMENT {i}")
        print(f"Language: {comment_data['language']} | Region: {comment_data['region']}")
        print(f"Stakeholder: {comment_data['stakeholder_type']}")
        print(f"\nText: {comment[:100]}{'...' if len(comment) > 100 else ''}")
        
        # Sentiment analysis
        sentiment_result = system.analyze_sentiment_basic(comment)
        print(f"Sentiment: {sentiment_result['sentiment']} (Score: {sentiment_result['compound']:.3f})")
        
        if sentiment_result['is_sarcastic']:
            print("⚠️  Sarcasm detected - sentiment adjusted")
        
        # Anomaly detection
        anomalies = system.detect_anomaly(comment, avg_length)
        if anomalies:
            print(f"🚨 Anomalies: {', '.join(anomalies)}")
        
        # Language detection simulation
        detected_lang = system.simulate_language_detection(comment)
        if detected_lang != 'English':
            print(f"🌐 Non-English content detected: {detected_lang}")
        
        # Key phrase extraction
        key_phrases = system.extract_key_phrases(comment)
        if key_phrases:
            phrases = [f"{word}({count})" for word, count in key_phrases[:3]]
            print(f"🔑 Key phrases: {', '.join(phrases)}")
        
        # Auto-response generation (skip for spam)
        if comment_data['stakeholder_type'] != 'Spam':
            auto_response = system.generate_auto_response(
                comment, 
                sentiment_result['sentiment'], 
                comment_data['stakeholder_type']
            )
            print(f"💬 Suggested response: {auto_response[:80]}...")
        
        print("-" * 70)
    
    # Generate advanced insights
    print(f"\n🧠 ADVANCED INSIGHTS")
    print("=" * 70)
    
    # Anomaly summary
    total_anomalies = 0
    anomaly_types = Counter()
    
    for comment_data in comments:
        anomalies = system.detect_anomaly(comment_data['comment'], avg_length)
        total_anomalies += len(anomalies)
        for anomaly in anomalies:
            anomaly_types[anomaly] += 1
    
    print(f"Total anomalies detected: {total_anomalies}")
    if anomaly_types:
        print("Most common anomaly types:")
        for anomaly_type, count in anomaly_types.most_common():
            print(f"  • {anomaly_type}: {count}")
    
    # Language distribution
    languages = Counter(c['language'] for c in comments)
    print(f"\nLanguage distribution:")
    for lang, count in languages.items():
        print(f"  • {lang}: {count} comments")
    
    # Quality score
    quality_comments = [c for c in comments if c['stakeholder_type'] != 'Spam']
    quality_score = len(quality_comments) / len(comments) * 100
    print(f"\nComment quality score: {quality_score:.1f}%")
    
    print("\n🎯 INNOVATION HIGHLIGHTS:")
    print("✅ Sarcasm and irony detection")
    print("✅ Spam and anomaly filtering")
    print("✅ Multilingual content simulation")
    print("✅ Automated response generation")
    print("✅ Advanced phrase extraction")
    print("✅ Quality assessment metrics")
    
    # Export advanced results
    export_data = {
        'analysis_timestamp': datetime.now().isoformat(),
        'total_comments': len(comments),
        'quality_score': quality_score,
        'anomaly_summary': dict(anomaly_types),
        'language_distribution': dict(languages),
        'advanced_features_demonstrated': [
            'sarcasm_detection',
            'spam_filtering',
            'anomaly_detection',
            'multilingual_support',
            'auto_response_generation',
            'phrase_extraction'
        ]
    }
    
    with open('advanced_analysis_results.json', 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Advanced results exported to 'advanced_analysis_results.json'")
    print("=" * 70)

if __name__ == "__main__":
    run_advanced_analysis()
    
    print(f"\n🏆 HACKATHON INNOVATION DEMO COMPLETED!")
    print("Advanced features showcased:")
    print("• Intelligent anomaly detection")
    print("• Sarcasm and context awareness")
    print("• Multilingual processing simulation")
    print("• Automated government response generation")
    print("• Quality assessment and filtering")