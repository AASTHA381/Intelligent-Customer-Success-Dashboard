import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
from datetime import datetime, timedelta

class ChurnPredictor:
    """Machine Learning model for predicting customer churn"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = [
            'tenure_months', 'monthly_revenue', 'total_interactions',
            'support_tickets', 'last_login_days', 'feature_usage_score'
        ]
        self.load_or_train_model()
    
    def prepare_features(self, customer_data):
        """Prepare features for prediction"""
        features = {}
        
        # Handle different data types
        if isinstance(customer_data, dict):
            features['tenure_months'] = customer_data.get('tenure_months', 0)
            features['monthly_revenue'] = customer_data.get('monthly_revenue', 0)
            features['total_interactions'] = customer_data.get('total_interactions', 0)
            features['support_tickets'] = customer_data.get('support_tickets', 0)
            features['last_login_days'] = customer_data.get('last_login_days', 0)
            features['feature_usage_score'] = customer_data.get('feature_usage_score', 0)
        else:
            # Handle pandas Series or other formats
            for col in self.feature_columns:
                features[col] = getattr(customer_data, col, 0)
        
        return pd.DataFrame([features])
    
    def train_model(self, data_path='../data/customers.csv'):
        """Train the churn prediction model"""
        try:
            # Load training data
            df = pd.read_csv(data_path)
            
            # Prepare features and target
            X = df[self.feature_columns]
            y = (df['churn_risk'] > 0.5).astype(int)  # Convert to binary
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test_scaled)
            print("Churn Prediction Model Performance:")
            print(classification_report(y_test, y_pred))
            
            # Save model
            os.makedirs('../models', exist_ok=True)
            joblib.dump(self.model, '../models/churn_model.pkl')
            joblib.dump(self.scaler, '../models/churn_scaler.pkl')
            
            return True
        except Exception as e:
            print(f"Error training churn model: {e}")
            return False
    
    def load_or_train_model(self):
        """Load existing model or train new one"""
        try:
            self.model = joblib.load('../models/churn_model.pkl')
            self.scaler = joblib.load('../models/churn_scaler.pkl')
            print("Loaded existing churn prediction model")
        except:
            print("Training new churn prediction model...")
            if not self.train_model():
                # Fallback to simple rule-based model
                self.model = None
                print("Using fallback rule-based churn prediction")
    
    def predict_churn(self, customer_data):
        """Predict churn probability for a customer"""
        try:
            if self.model is not None:
                features_df = self.prepare_features(customer_data)
                features_scaled = self.scaler.transform(features_df)
                churn_prob = self.model.predict_proba(features_scaled)[0][1]
                return churn_prob
            else:
                # Fallback rule-based prediction
                return self._rule_based_churn_prediction(customer_data)
        except Exception as e:
            print(f"Error predicting churn: {e}")
            return self._rule_based_churn_prediction(customer_data)
    
    def _rule_based_churn_prediction(self, customer_data):
        """Fallback rule-based churn prediction"""
        risk_score = 0.0
        
        # Low tenure increases risk
        tenure = customer_data.get('tenure_months', 0)
        if tenure < 3:
            risk_score += 0.3
        elif tenure < 12:
            risk_score += 0.1
        
        # Low revenue increases risk
        revenue = customer_data.get('monthly_revenue', 0)
        if revenue < 50:
            risk_score += 0.2
        elif revenue < 100:
            risk_score += 0.1
        
        # Few interactions increase risk
        interactions = customer_data.get('total_interactions', 0)
        if interactions < 5:
            risk_score += 0.2
        
        # Recent login activity
        last_login = customer_data.get('last_login_days', 0)
        if last_login > 30:
            risk_score += 0.3
        elif last_login > 7:
            risk_score += 0.1
        
        return min(risk_score, 1.0)

class HealthScorer:
    """Customer health scoring system"""
    
    def __init__(self):
        self.weights = {
            'engagement': 0.3,
            'usage': 0.25,
            'satisfaction': 0.2,
            'financial': 0.15,
            'support': 0.1
        }
    
    def calculate_health_score(self, customer_data):
        """Calculate overall customer health score (0-100)"""
        scores = {}
        
        # Engagement score
        scores['engagement'] = self._calculate_engagement_score(customer_data)
        
        # Usage score
        scores['usage'] = self._calculate_usage_score(customer_data)
        
        # Satisfaction score
        scores['satisfaction'] = self._calculate_satisfaction_score(customer_data)
        
        # Financial score
        scores['financial'] = self._calculate_financial_score(customer_data)
        
        # Support score
        scores['support'] = self._calculate_support_score(customer_data)
        
        # Calculate weighted average
        health_score = sum(scores[key] * self.weights[key] for key in scores.keys())
        
        return min(max(health_score, 0), 100)  # Ensure score is between 0-100
    
    def _calculate_engagement_score(self, customer_data):
        """Calculate engagement score based on interactions and login frequency"""
        interactions = customer_data.get('total_interactions', 0)
        last_login_days = customer_data.get('last_login_days', 30)
        
        # More interactions = higher score
        interaction_score = min(interactions * 2, 50)
        
        # Recent login = higher score
        if last_login_days <= 1:
            login_score = 50
        elif last_login_days <= 7:
            login_score = 40
        elif last_login_days <= 30:
            login_score = 20
        else:
            login_score = 0
        
        return interaction_score + login_score
    
    def _calculate_usage_score(self, customer_data):
        """Calculate usage score based on feature adoption"""
        feature_usage = customer_data.get('feature_usage_score', 0)
        return min(feature_usage * 20, 100)
    
    def _calculate_satisfaction_score(self, customer_data):
        """Calculate satisfaction score"""
        # Use support tickets as inverse satisfaction indicator
        support_tickets = customer_data.get('support_tickets', 0)
        
        if support_tickets == 0:
            return 100
        elif support_tickets <= 2:
            return 80
        elif support_tickets <= 5:
            return 60
        else:
            return 40
    
    def _calculate_financial_score(self, customer_data):
        """Calculate financial health score"""
        revenue = customer_data.get('monthly_revenue', 0)
        tenure = customer_data.get('tenure_months', 1)
        
        # Revenue per month of tenure
        revenue_score = min(revenue / 10, 70)  # Max 70 points for revenue
        
        # Tenure bonus
        tenure_bonus = min(tenure * 2, 30)  # Max 30 points for tenure
        
        return revenue_score + tenure_bonus
    
    def _calculate_support_score(self, customer_data):
        """Calculate support interaction score"""
        support_tickets = customer_data.get('support_tickets', 0)
        
        # Fewer support tickets = better health
        if support_tickets == 0:
            return 100
        elif support_tickets <= 1:
            return 90
        elif support_tickets <= 3:
            return 70
        elif support_tickets <= 5:
            return 50
        else:
            return 30
    
    def get_health_status(self, health_score):
        """Get health status label"""
        if health_score >= 90:
            return "Excellent"
        elif health_score >= 70:
            return "Good"
        elif health_score >= 50:
            return "Fair"
        else:
            return "Poor"

class InterventionRecommender:
    """AI-powered intervention strategy recommender"""
    
    def __init__(self):
        self.intervention_strategies = {
            'high_churn_low_engagement': [
                'Schedule personal check-in call',
                'Offer product training session',
                'Provide dedicated customer success manager',
                'Send personalized onboarding materials'
            ],
            'high_churn_low_usage': [
                'Offer feature demo session',
                'Provide use case examples',
                'Schedule product walkthrough',
                'Send tutorial videos'
            ],
            'high_churn_support_issues': [
                'Priority support queue assignment',
                'Technical expert consultation',
                'Product feedback session',
                'Escalate to development team'
            ],
            'low_revenue': [
                'Discuss upgrade opportunities',
                'Show ROI calculations',
                'Offer limited-time discount',
                'Highlight premium features'
            ],
            'engagement_drop': [
                'Send re-engagement email campaign',
                'Offer new feature preview',
                'Schedule product update call',
                'Provide industry insights'
            ]
        }
    
    def get_recommendations(self, customer_data):
        """Get personalized intervention recommendations"""
        recommendations = []
        
        churn_risk = customer_data.get('churn_risk', 0)
        if 'churn_probability' in customer_data:
            churn_risk = customer_data['churn_probability']
        
        health_score = customer_data.get('health_score', 100)
        
        # High churn risk customers
        if churn_risk > 0.7:
            engagement_score = self._get_engagement_level(customer_data)
            usage_score = customer_data.get('feature_usage_score', 0)
            support_tickets = customer_data.get('support_tickets', 0)
            
            if engagement_score < 50:
                recommendations.extend(self.intervention_strategies['high_churn_low_engagement'][:2])
            
            if usage_score < 3:
                recommendations.extend(self.intervention_strategies['high_churn_low_usage'][:2])
            
            if support_tickets > 3:
                recommendations.extend(self.intervention_strategies['high_churn_support_issues'][:1])
        
        # Medium churn risk
        elif churn_risk > 0.4:
            last_login = customer_data.get('last_login_days', 0)
            if last_login > 14:
                recommendations.extend(self.intervention_strategies['engagement_drop'][:2])
        
        # Low revenue customers
        revenue = customer_data.get('monthly_revenue', 0)
        if revenue < 100:
            recommendations.extend(self.intervention_strategies['low_revenue'][:1])
        
        # Remove duplicates and limit to top 5
        recommendations = list(dict.fromkeys(recommendations))[:5]
        
        # Add priority levels
        prioritized_recommendations = []
        for i, rec in enumerate(recommendations):
            priority = 'High' if i < 2 else 'Medium' if i < 4 else 'Low'
            prioritized_recommendations.append({
                'action': rec,
                'priority': priority,
                'category': self._categorize_recommendation(rec)
            })
        
        return prioritized_recommendations
    
    def _get_engagement_level(self, customer_data):
        """Calculate engagement level"""
        interactions = customer_data.get('total_interactions', 0)
        last_login = customer_data.get('last_login_days', 30)
        
        engagement_score = interactions * 10
        if last_login <= 7:
            engagement_score += 20
        elif last_login <= 30:
            engagement_score += 10
        
        return engagement_score
    
    def _categorize_recommendation(self, recommendation):
        """Categorize recommendation type"""
        if 'call' in recommendation.lower() or 'personal' in recommendation.lower():
            return 'Personal Outreach'
        elif 'training' in recommendation.lower() or 'demo' in recommendation.lower():
            return 'Product Education'
        elif 'support' in recommendation.lower() or 'technical' in recommendation.lower():
            return 'Support Enhancement'
        elif 'discount' in recommendation.lower() or 'upgrade' in recommendation.lower():
            return 'Commercial'
        else:
            return 'Engagement'