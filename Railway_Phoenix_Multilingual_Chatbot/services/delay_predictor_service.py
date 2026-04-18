"""
Delay Predictor Service
Interfaces with the trained ML model for train delay predictions
"""
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional
import sys
sys.path.append('/Workspace/Users/cse230001067@iiti.ac.in/Rail_Drishti_Multilingual_Chatbot')
import config

class DelayPredictorService:
    """Service for train delay predictions with dynamic updates"""
    
    def __init__(self):
        self.model = None
        self.train_data = None
        self.load_model()
        self.load_train_data()
        
    def load_model(self):
        """Load the trained model"""
        try:
            with open(config.DELAY_MODEL_PATH, 'rb') as f:
                self.model = pickle.load(f)
            print("✓ Delay prediction model loaded")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    def load_train_data(self):
        """Load train data for lookups"""
        try:
            self.train_data = pd.read_csv(config.TRAIN_DATA_PATH, low_memory=False)
            print(f"✓ Train data loaded ({len(self.train_data)} records)")
        except Exception as e:
            print(f"Error loading train data: {e}")
            self.train_data = None
    
    def get_train_info(self, train_number: str) -> Optional[Dict]:
        """Get train information by number"""
        if self.train_data is None:
            return None
        
        train_info = self.train_data[self.train_data['Train No'].astype(str) == str(train_number)]
        if train_info.empty:
            return None
        
        first_record = train_info.iloc[0]
        return {
            'train_no': train_number,
            'train_name': first_record['Train Name'],
            'source': first_record['Source Station Name'],
            'destination': first_record['Destination Station Name'],
            'train_type': first_record.get('Train Type', 'Unknown')
        }
    
    def get_route_stations(self, train_number: str) -> list:
        """Get list of stations for a train"""
        if self.train_data is None:
            return []
        
        route = self.train_data[self.train_data['Train No'].astype(str) == str(train_number)]
        if route.empty:
            return []
        
        stations = route[['Station Code', 'Station Name', 'Distance']].drop_duplicates()
        return stations.to_dict('records')
    
    def predict_delay(
        self,
        train_number: str,
        current_station: str,
        destination_station: str,
        weather: str = "Clear",
        congestion: str = "Low",
        day_of_week: Optional[str] = None,
        time_of_day: Optional[str] = None
    ) -> Dict:
        """
        Predict delay for a train journey
        
        Returns:
            Dict with predicted_delay, confidence, reason, and details
        """
        if self.model is None or self.train_data is None:
            return {
                'predicted_delay': 30,
                'confidence': 0.5,
                'reason': 'Model not available',
                'details': 'Using average delay estimate'
            }
        
        # Get current time info if not provided
        if day_of_week is None:
            day_of_week = datetime.now().strftime('%A')
        if time_of_day is None:
            hour = datetime.now().hour
            if 5 <= hour < 12:
                time_of_day = 'Morning'
            elif 12 <= hour < 17:
                time_of_day = 'Afternoon'
            elif 17 <= hour < 21:
                time_of_day = 'Evening'
            else:
                time_of_day = 'Night'
        
        # Find train info
        train_route = self.train_data[
            (self.train_data['Train No'].astype(str) == str(train_number))
        ]
        
        if train_route.empty:
            return {
                'predicted_delay': 25,
                'confidence': 0.4,
                'reason': 'Train not found in database',
                'details': 'Using default estimate'
            }
        
        # Get distance and train type
        sample_record = train_route.iloc[0]
        train_type = sample_record.get('Train Type', 'Express')
        
        # Calculate distance between current and destination
        current_dist = train_route[train_route['Station Code'] == current_station]
        dest_dist = train_route[train_route['Station Code'] == destination_station]
        
        if not current_dist.empty and not dest_dist.empty:
            distance = abs(float(dest_dist.iloc[0]['Distance']) - float(current_dist.iloc[0]['Distance']))
        else:
            distance = 200  # Default estimate
        
        # Prepare features for model (ONLY 3 features to match trained model)
        features = self._prepare_features(distance, weather, congestion)
        
        try:
            # Get prediction
            predicted_delay = self.model.predict([features])[0]
            predicted_delay = max(0, int(predicted_delay))  # Ensure non-negative
            
            # Estimate confidence based on data availability
            confidence = 0.75 if not train_route.empty else 0.6
            
            # Determine primary reason
            reason = self._determine_delay_reason(weather, congestion, predicted_delay)
            
            return {
                'predicted_delay': predicted_delay,
                'confidence': confidence,
                'reason': reason,
                'details': f"Based on {weather} weather and {congestion} congestion",
                'distance_km': distance,
                'train_type': train_type
            }
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return {
                'predicted_delay': 35,
                'confidence': 0.5,
                'reason': 'Prediction error',
                'details': str(e)
            }
    
    def _prepare_features(self, distance, weather, congestion) -> list:
        """
        Prepare feature vector for model
        
        Model expects exactly 3 features:
        1. Distance Between Stations (km)
        2. Weather Conditions (encoded: Clear=0, Foggy=1, Rainy=2)
        3. Route Congestion (encoded: High=0, Low=1, Medium=2)
        """
        # Match exact encoding from training data
        weather_map = {'Clear': 0, 'Foggy': 1, 'Rainy': 2, 'Cloudy': 0, 'Storm': 2}
        congestion_map = {'High': 0, 'Low': 1, 'Medium': 2}
        
        features = [
            distance,  # Feature 1: Distance
            weather_map.get(weather, 0),  # Feature 2: Weather encoded
            congestion_map.get(congestion, 1)  # Feature 3: Congestion encoded
        ]
        
        return features
    
    def _determine_delay_reason(self, weather: str, congestion: str, delay: int) -> str:
        """Determine primary reason for delay"""
        if delay < 10:
            return "On time"
        elif weather in ['Rainy', 'Foggy', 'Storm']:
            return config.DELAY_REASONS['weather']
        elif congestion in ['High', 'Medium']:
            return config.DELAY_REASONS['congestion']
        elif delay > 60:
            return config.DELAY_REASONS['technical']
        else:
            return config.DELAY_REASONS['operational']
    
    def update_prediction_with_actual(
        self,
        train_number: str,
        predicted_delay: int,
        actual_delay: int
    ) -> Dict:
        """
        Log the difference between predicted and actual delay
        This data can be used for model retraining
        """
        error = abs(predicted_delay - actual_delay)
        accuracy = max(0, 100 - (error / max(predicted_delay, 1) * 100))
        
        # Store in Delta table for future retraining
        # (Implementation depends on Delta table setup)
        
        return {
            'error_minutes': error,
            'accuracy_percent': accuracy,
            'needs_learning': error > 15
        }
