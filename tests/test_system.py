import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

import unittest
import json
from app import app
from ml_models import ChurnPredictor, HealthScorer, InterventionRecommender

class TestCustomerSuccessDashboard(unittest.TestCase):
    
    def setUp(self):
        """Set up test client"""
        self.app = app.test_client()
        self.app.testing = True
        
        # Initialize models
        self.churn_predictor = ChurnPredictor()
        self.health_scorer = HealthScorer()
        self.intervention_recommender = InterventionRecommender()
    
    def test_dashboard_stats_endpoint(self):
        """Test dashboard stats API endpoint"""
        response = self.app.get('/api/dashboard-stats')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('total_customers', data)
        self.assertIn('active_customers', data)
        self.assertIn('churn_risk_customers', data)
        self.assertIn('avg_health_score', data)
        print("✓ Dashboard stats endpoint working correctly")
    
    def test_customers_endpoint(self):
        """Test customers API endpoint"""
        response = self.app.get('/api/customers')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIsInstance(data, list)
        if data:
            customer = data[0]
            self.assertIn('churn_probability', customer)
            self.assertIn('health_score', customer)
        print("✓ Customers endpoint working correctly")
    
    def test_churn_prediction_high_risk(self):
        """Test churn prediction for high-risk customer"""
        test_data = {
            "tenure_months": 3,
            "monthly_revenue": 45,
            "total_interactions": 8,
            "support_tickets": 6,
            "last_login_days": 21,
            "feature_usage_score": 1.2
        }
        
        response = self.app.post('/api/predict-churn',
                               data=json.dumps(test_data),
                               content_type='application/json')
        self.assertEqual(response.status_code, 200)
        
        result = json.loads(response.data)
        self.assertIn('churn_probability', result)
        self.assertIn('risk_level', result)
        self.assertGreater(result['churn_probability'], 0.5)  # Should be high risk
        print(f"✓ High-risk customer test: {result['churn_probability']:.3f} probability, {result['risk_level']} risk")
    
    def test_churn_prediction_healthy_customer(self):
        """Test churn prediction for healthy customer"""
        test_data = {
            "tenure_months": 24,
            "monthly_revenue": 350,
            "total_interactions": 85,
            "support_tickets": 1,
            "last_login_days": 1,
            "feature_usage_score": 4.8
        }
        
        response = self.app.post('/api/predict-churn',
                               data=json.dumps(test_data),
                               content_type='application/json')
        self.assertEqual(response.status_code, 200)
        
        result = json.loads(response.data)
        self.assertLess(result['churn_probability'], 0.3)  # Should be low risk
        print(f"✓ Healthy customer test: {result['churn_probability']:.3f} probability, {result['risk_level']} risk")
    
    def test_health_score_calculation(self):
        """Test health score calculation"""
        test_data = {
            "tenure_months": 12,
            "monthly_revenue": 200,
            "total_interactions": 45,
            "support_tickets": 2,
            "last_login_days": 3,
            "feature_usage_score": 3.8
        }
        
        response = self.app.post('/api/health-score',
                               data=json.dumps(test_data),
                               content_type='application/json')
        self.assertEqual(response.status_code, 200)
        
        result = json.loads(response.data)
        self.assertIn('health_score', result)
        self.assertIn('health_status', result)
        self.assertGreaterEqual(result['health_score'], 0)
        self.assertLessEqual(result['health_score'], 100)
        print(f"✓ Health score test: {result['health_score']:.1f}/100 ({result['health_status']})")
    
    def test_intervention_recommendations(self):
        """Test intervention recommendations"""
        test_data = {
            "tenure_months": 6,
            "monthly_revenue": 75,
            "total_interactions": 15,
            "support_tickets": 4,
            "last_login_days": 14,
            "feature_usage_score": 2.1,
            "churn_probability": 0.65
        }
        
        response = self.app.post('/api/recommendations',
                               data=json.dumps(test_data),
                               content_type='application/json')
        self.assertEqual(response.status_code, 200)
        
        result = json.loads(response.data)
        self.assertIn('recommendations', result)
        recommendations = result['recommendations']
        self.assertIsInstance(recommendations, list)
        
        if recommendations:
            rec = recommendations[0]
            self.assertIn('action', rec)
            self.assertIn('priority', rec)
            self.assertIn('category', rec)
        
        print(f"✓ Recommendations test: {len(recommendations)} recommendations generated")
        for i, rec in enumerate(recommendations[:3]):  # Show first 3
            print(f"  {i+1}. {rec['action']} ({rec['priority']} priority)")
    
    def test_ml_models_directly(self):
        """Test ML models directly"""
        test_customer = {
            "tenure_months": 8,
            "monthly_revenue": 120,
            "total_interactions": 25,
            "support_tickets": 3,
            "last_login_days": 5,
            "feature_usage_score": 3.2
        }
        
        # Test churn predictor
        churn_prob = self.churn_predictor.predict_churn(test_customer)
        self.assertIsInstance(churn_prob, float)
        self.assertGreaterEqual(churn_prob, 0)
        self.assertLessEqual(churn_prob, 1)
        print(f"✓ Direct churn prediction: {churn_prob:.3f}")
        
        # Test health scorer
        health_score = self.health_scorer.calculate_health_score(test_customer)
        self.assertIsInstance(health_score, float)
        self.assertGreaterEqual(health_score, 0)
        self.assertLessEqual(health_score, 100)
        print(f"✓ Direct health score: {health_score:.1f}")
        
        # Test intervention recommender
        recommendations = self.intervention_recommender.get_recommendations(test_customer)
        self.assertIsInstance(recommendations, list)
        print(f"✓ Direct recommendations: {len(recommendations)} suggestions")

def run_comprehensive_tests():
    """Run all tests and display results"""
    print("=" * 60)
    print("🧪 RUNNING COMPREHENSIVE TESTS")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCustomerSuccessDashboard)
    runner = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, 'w'))
    
    # Run tests
    result = runner.run(suite)
    
    # Display summary
    print(f"\n📊 TEST SUMMARY:")
    print(f"✅ Tests passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"❌ Tests failed: {len(result.failures)}")
    print(f"🔥 Test errors: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\n🔥 ERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\n🎯 Success Rate: {success_rate:.1f}%")
    
    if success_rate == 100:
        print("🎉 ALL TESTS PASSED! The system is working perfectly.")
    elif success_rate >= 80:
        print("✅ Most tests passed. System is mostly functional.")
    else:
        print("⚠️  Several tests failed. Please check the implementation.")
    
    print("=" * 60)

if __name__ == '__main__':
    # Change to the test directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Run comprehensive tests
    run_comprehensive_tests()