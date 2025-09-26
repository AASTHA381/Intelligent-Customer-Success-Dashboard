from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from ml_models import ChurnPredictor, HealthScorer, InterventionRecommender

app = Flask(__name__, 
           static_folder='../frontend',
           template_folder='../frontend')
CORS(app)

# Initialize models
churn_predictor = ChurnPredictor()
health_scorer = HealthScorer()
intervention_recommender = InterventionRecommender()

# Load and prepare data
def load_customer_data():
    """Load customer data from CSV files"""
    try:
        customers = pd.read_csv('../data/customers.csv')
        interactions = pd.read_csv('../data/customer_interactions.csv')
        return customers, interactions
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame(), pd.DataFrame()

@app.route('/')
def index():
    """Serve the main dashboard"""
    return render_template('index.html')

@app.route('/api/dashboard-stats')
def dashboard_stats():
    """Get overall dashboard statistics"""
    try:
        customers, interactions = load_customer_data()
        
        if customers.empty:
            return jsonify({'error': 'No customer data available'}), 500
        
        # Calculate key metrics
        total_customers = len(customers)
        active_customers = len(customers[customers['status'] == 'active'])
        churn_risk = len(customers[customers['churn_risk'] >= 0.7])
        avg_health_score = customers['health_score'].mean()
        
        # Recent interactions
        recent_interactions = len(interactions[
            pd.to_datetime(interactions['interaction_date']) >= 
            datetime.now() - timedelta(days=30)
        ])
        
        return jsonify({
            'total_customers': total_customers,
            'active_customers': active_customers,
            'churn_risk_customers': churn_risk,
            'avg_health_score': round(avg_health_score, 2),
            'recent_interactions': recent_interactions
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/customers')
def get_customers():
    """Get all customers with their health scores and churn predictions"""
    try:
        customers, _ = load_customer_data()
        
        if customers.empty:
            return jsonify([])
        
        # Add predictions and health scores
        customers_data = []
        for _, customer in customers.iterrows():
            customer_dict = customer.to_dict()
            
            # Get churn prediction
            churn_prob = churn_predictor.predict_churn(customer_dict)
            customer_dict['churn_probability'] = round(churn_prob, 3)
            
            # Get health score
            health_score = health_scorer.calculate_health_score(customer_dict)
            customer_dict['health_score'] = round(health_score, 2)
            
            # Get intervention recommendations
            if churn_prob > 0.5:
                recommendations = intervention_recommender.get_recommendations(customer_dict)
                customer_dict['recommendations'] = recommendations
            
            customers_data.append(customer_dict)
        
        return jsonify(customers_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/customer/<int:customer_id>')
def get_customer(customer_id):
    """Get detailed information for a specific customer"""
    try:
        customers, interactions = load_customer_data()
        
        customer = customers[customers['customer_id'] == customer_id]
        if customer.empty:
            return jsonify({'error': 'Customer not found'}), 404
        
        customer_data = customer.iloc[0].to_dict()
        
        # Get customer interactions
        customer_interactions = interactions[
            interactions['customer_id'] == customer_id
        ].to_dict('records')
        
        # Get predictions
        churn_prob = churn_predictor.predict_churn(customer_data)
        health_score = health_scorer.calculate_health_score(customer_data)
        recommendations = intervention_recommender.get_recommendations(customer_data)
        
        return jsonify({
            'customer': customer_data,
            'interactions': customer_interactions,
            'churn_probability': round(churn_prob, 3),
            'health_score': round(health_score, 2),
            'recommendations': recommendations
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict-churn', methods=['POST'])
def predict_churn():
    """Predict churn for given customer data"""
    try:
        customer_data = request.json
        churn_prob = churn_predictor.predict_churn(customer_data)
        
        return jsonify({
            'churn_probability': round(churn_prob, 3),
            'risk_level': 'High' if churn_prob > 0.7 else 'Medium' if churn_prob > 0.4 else 'Low'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health-score', methods=['POST'])
def calculate_health():
    """Calculate customer health score"""
    try:
        customer_data = request.json
        health_score = health_scorer.calculate_health_score(customer_data)
        
        return jsonify({
            'health_score': round(health_score, 2),
            'health_status': health_scorer.get_health_status(health_score)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    """Get intervention recommendations for a customer"""
    try:
        customer_data = request.json
        recommendations = intervention_recommender.get_recommendations(customer_data)
        
        return jsonify({'recommendations': recommendations})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analytics/churn-trend')
def churn_trend():
    """Get churn trend analytics"""
    try:
        customers, _ = load_customer_data()
        
        # Group by month and calculate churn rates
        customers['signup_date'] = pd.to_datetime(customers['signup_date'])
        monthly_data = customers.groupby(customers['signup_date'].dt.to_period('M')).agg({
            'customer_id': 'count',
            'churn_risk': 'mean'
        }).reset_index()
        
        monthly_data['month'] = monthly_data['signup_date'].astype(str)
        
        return jsonify({
            'months': monthly_data['month'].tolist(),
            'customer_count': monthly_data['customer_id'].tolist(),
            'churn_rate': (monthly_data['churn_risk'] * 100).round(2).tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analytics/health-distribution')
def health_distribution():
    """Get health score distribution"""
    try:
        customers, _ = load_customer_data()
        
        # Calculate health score distribution
        health_ranges = {
            'Excellent (90-100)': len(customers[customers['health_score'] >= 90]),
            'Good (70-89)': len(customers[(customers['health_score'] >= 70) & (customers['health_score'] < 90)]),
            'Fair (50-69)': len(customers[(customers['health_score'] >= 50) & (customers['health_score'] < 70)]),
            'Poor (0-49)': len(customers[customers['health_score'] < 50])
        }
        
        return jsonify(health_ranges)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Customer Success Dashboard...")
    print("Dashboard will be available at: http://localhost:5000")
    app.run(debug=True, port=5000)