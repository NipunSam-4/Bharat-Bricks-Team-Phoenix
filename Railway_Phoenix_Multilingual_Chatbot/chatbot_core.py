"""
Rail Drishti Chatbot Core - Enhanced Version
Main chatbot logic with multilingual support and advanced query processing
"""
import re
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import sys
sys.path.append('/Workspace/Users/cse230001067@iiti.ac.in/Rail_Drishti_Multilingual_Chatbot')
from services.translation_service import TranslationService
from services.delay_predictor_service import DelayPredictorService
from services.query_processor_service import QueryProcessor
import config

class RailDrishtiChatbot:
    """Main chatbot class for multilingual train delay queries with dynamic predictions"""
    
    def __init__(self):
        self.translator = TranslationService()
        self.predictor = DelayPredictorService()
        self.query_processor = QueryProcessor()
        self.conversation_history = []
        self.user_language = config.DEFAULT_LANGUAGE
        self.current_train_context = None  # Store train context for follow-up questions
        
    def process_message(self, user_message: str, language: Optional[str] = None, 
                       weather: str = "Clear", congestion: str = "Low") -> Dict:
        """
        Process user message and return intelligent response
        
        Args:
            user_message: User's input text
            language: Optional language code override
            weather: Current weather condition
            congestion: Current congestion level
            
        Returns:
            Dict with response, language, metadata, and prediction details
        """
        # Detect or use provided language
        if language is None:
            detected_lang = self.translator.detect_language(user_message)
            self.user_language = detected_lang
        else:
            self.user_language = language
        
        # Translate to English for processing
        english_message = self.translator.translate_to_english(
            user_message, self.user_language
        )
        
        # Process query using advanced query processor
        query_analysis = self.query_processor.process_query(
            english_message, self.user_language
        )
        
        # Generate intelligent response
        english_response, metadata = self._generate_intelligent_response(
            query_analysis, weather, congestion
        )
        
        # Translate response back to user's language
        response = self.translator.translate_from_english(
            english_response, self.user_language
        )
        
        # Store in history
        self.conversation_history.append({
            'user': user_message,
            'bot': response,
            'language': self.user_language,
            'intent': query_analysis['intent'],
            'timestamp': datetime.now().isoformat()
        })
        
        # Trim history
        if len(self.conversation_history) > config.MAX_HISTORY_LENGTH:
            self.conversation_history = self.conversation_history[-config.MAX_HISTORY_LENGTH:]
        
        return {
            'response': response,
            'language': self.user_language,
            'language_name': self.translator.get_language_name(self.user_language),
            'intent': query_analysis['intent'],
            'confidence': query_analysis['confidence'],
            'train_number': query_analysis.get('train_number'),
            'metadata': metadata
        }
    
    def _generate_intelligent_response(self, query_analysis: Dict, 
                                      weather: str, congestion: str) -> tuple:
        """
        Generate intelligent response based on query analysis
        
        Returns:
            Tuple of (response_text, metadata_dict)
        """
        intent = query_analysis['intent']
        train_number = query_analysis['train_number']
        entities = query_analysis['entities']
        
        metadata = {
            'prediction': None,
            'train_info': None,
            'delay_minutes': None
        }
        
        # Handle different intents
        if intent == 'greeting':
            return self._handle_greeting(), metadata
        
        elif intent == 'general_query':
            return self._handle_help(), metadata
        
        elif intent == 'delay_query':
            return self._handle_delay_query(
                train_number, entities, weather, congestion, metadata
            )
        
        elif intent == 'status_query':
            return self._handle_status_query(train_number, entities, metadata)
        
        elif intent == 'reason_query':
            return self._handle_reason_query(train_number, weather, congestion, metadata)
        
        elif intent == 'schedule_query':
            return self._handle_schedule_query(train_number, metadata)
        
        else:
            return self._handle_help(), metadata
    
    def _handle_greeting(self) -> str:
        """Handle greeting intent"""
        greetings = {
            'en': "Hello! 👋 Welcome to Rail Drishti, your intelligent train delay assistant. I can help you check real-time train delays, understand delay reasons, and provide updates in your regional language. Please provide your train number (e.g., 12345) to get started.",
            'hi': "नमस्ते! 👋 रेल दृष्टि में आपका स्वागत है। मैं आपको ट्रेन विलंब, कारण और वास्तविक समय अपडेट में मदद कर सकता हूं। कृपया अपना ट्रेन नंबर बताएं।"
        }
        return greetings.get(self.user_language, greetings['en'])
    
    def _handle_help(self) -> str:
        """Handle help/general intent"""
        help_text = {
            'en': ("🚂 I can assist you with:\n"
                   "1️⃣ Real-time train delay predictions\n"
                   "2️⃣ Train status and current location\n"
                   "3️⃣ Delay reasons and explanations\n"
                   "4️⃣ Train schedule information\n\n"
                   "Simply provide your train number (5 digits) and I'll help you!"),
            'hi': ("🚂 मैं आपकी इनमें मदद कर सकता हूं:\n"
                   "1️⃣ वास्तविक समय ट्रेन विलंब भविष्यवाणी\n"
                   "2️⃣ ट्रेन स्थिति और वर्तमान स्थान\n"
                   "3️⃣ विलंब के कारण\n"
                   "4️⃣ ट्रेन समय-सारणी\n\n"
                   "बस अपना ट्रेन नंबर बताएं!")
        }
        return help_text.get(self.user_language, help_text['en'])
    
    def _handle_delay_query(self, train_number: str, entities: Dict, 
                           weather: str, congestion: str, metadata: Dict) -> tuple:
        """Handle delay prediction query"""
        if not train_number:
            # Check if we have context from previous conversation
            if self.current_train_context:
                train_number = self.current_train_context
            else:
                return ("Please provide your train number to check delay information. "
                       "Example: 'Check delay for train 12345'"), metadata
        
        # Store train context
        self.current_train_context = train_number
        
        # Get train info first
        train_info = self.predictor.get_train_info(train_number)
        if not train_info:
            return (f"Sorry, I couldn't find information for train number {train_number}. "
                   "Please check the number and try again."), metadata
        
        # Extract stations from entities or use defaults
        origin_stations = [s[1] for s in entities['stations'] if s[0] == 'origin']
        dest_stations = [s[1] for s in entities['stations'] if s[0] == 'destination']
        
        current_station = origin_stations[0] if origin_stations else train_info.get('source', 'NDLS')
        destination = dest_stations[0] if dest_stations else train_info.get('destination', 'MMCT')
        
        # Get delay prediction
        prediction = self.predictor.predict_delay(
            train_number=train_number,
            current_station=current_station,
            destination_station=destination,
            weather=weather,
            congestion=congestion
        )
        
        # Store metadata
        metadata['prediction'] = prediction
        metadata['train_info'] = train_info
        metadata['delay_minutes'] = prediction['predicted_delay']
        
        # Format response
        delay_mins = prediction['predicted_delay']
        confidence = prediction['confidence'] * 100
        reason = prediction['reason']
        distance = prediction.get('distance_km', 0)
        
        # Calculate ETA
        eta = datetime.now() + timedelta(minutes=delay_mins + 120)  # Base travel time estimate
        eta_str = eta.strftime("%I:%M %p")
        
        if delay_mins < 10:
            status_emoji = "✅"
            status_msg = "Great news!"
        elif delay_mins < 30:
            status_emoji = "⚠️"
            status_msg = "Slight delay expected"
        else:
            status_emoji = "🔴"
            status_msg = "Significant delay"
        
        response = (
            f"{status_emoji} {status_msg}\n\n"
            f"🚂 Train: {train_info['train_name']} (#{train_number})\n"
            f"📍 Route: {current_station} → {destination} ({distance:.0f} km)\n"
            f"⏱️ Expected Delay: {delay_mins} minutes\n"
            f"🎯 Confidence: {confidence:.0f}%\n"
            f"🕐 Estimated Arrival: {eta_str}\n"
            f"📋 Primary Reason: {reason}\n\n"
            f"💡 Tip: This prediction updates dynamically as the train moves. "
            f"Ask me again for latest updates!"
        )
        
        return response, metadata
    
    def _handle_status_query(self, train_number: str, entities: Dict, metadata: Dict) -> tuple:
        """Handle train status query"""
        if not train_number and self.current_train_context:
            train_number = self.current_train_context
        
        if not train_number:
            return "Please provide a train number to check status.", metadata
        
        train_info = self.predictor.get_train_info(train_number)
        if not train_info:
            return f"Train {train_number} not found in database.", metadata
        
        metadata['train_info'] = train_info
        
        # Get route stations
        stations = self.predictor.get_route_stations(train_number)
        station_count = len(stations) if stations else 0
        
        response = (
            f"🚂 Train Information\n\n"
            f"Train Number: {train_number}\n"
            f"Name: {train_info['train_name']}\n"
            f"Type: {train_info.get('train_type', 'Express')}\n"
            f"Origin: {train_info['source']}\n"
            f"Destination: {train_info['destination']}\n"
            f"Stops: {station_count} stations\n\n"
            f"Would you like to check delay predictions for this train?"
        )
        
        return response, metadata
    
    def _handle_reason_query(self, train_number: str, weather: str, 
                            congestion: str, metadata: Dict) -> tuple:
        """Handle delay reason query"""
        # Get general delay reasons or specific for a train
        if train_number and self.current_train_context:
            # Get prediction to explain current delay
            prediction = self.predictor.predict_delay(
                train_number=train_number,
                current_station="NDLS",
                destination_station="MMCT",
                weather=weather,
                congestion=congestion
            )
            
            reason = prediction['reason']
            details = prediction['details']
            
            response = (
                f"🔍 Delay Analysis for Train {train_number}:\n\n"
                f"Primary Cause: {reason}\n"
                f"Details: {details}\n\n"
                f"📊 Current Conditions:\n"
                f"☁️ Weather: {weather}\n"
                f"🚦 Congestion: {congestion}\n\n"
                f"The system learns from actual arrival times to improve predictions!"
            )
        else:
            response = (
                "🔍 Common causes of train delays:\n\n"
                "1️⃣ Weather Conditions\n"
                "   • Rain, fog, storms affect visibility and speed\n\n"
                "2️⃣ Track Congestion\n"
                "   • High traffic on routes\n"
                "   • Platform availability issues\n\n"
                "3️⃣ Technical Issues\n"
                "   • Engine problems\n"
                "   • Signaling issues\n\n"
                "4️⃣ Operational Factors\n"
                "   • Crew changes\n"
                "   • Priority train crossings\n\n"
                "Provide a train number for specific delay analysis!"
            )
        
        return response, metadata
    
    def _handle_schedule_query(self, train_number: str, metadata: Dict) -> tuple:
        """Handle schedule information query"""
        if not train_number and self.current_train_context:
            train_number = self.current_train_context
        
        if not train_number:
            return "Please provide a train number to check schedule.", metadata
        
        train_info = self.predictor.get_train_info(train_number)
        if not train_info:
            return f"Schedule information not available for train {train_number}.", metadata
        
        response = (
            f"📅 Schedule Information\n\n"
            f"Train: {train_info['train_name']} (#{train_number})\n"
            f"Route: {train_info['source']} → {train_info['destination']}\n\n"
            f"Note: For real-time delay predictions, ask me about current delays!"
        )
        
        return response, metadata
    
    def reset_conversation(self):
        """Reset conversation history and context"""
        self.conversation_history = []
        self.current_train_context = None
        self.user_language = config.DEFAULT_LANGUAGE
    
    def get_conversation_summary(self) -> Dict:
        """Get summary of current conversation"""
        return {
            'message_count': len(self.conversation_history),
            'current_train': self.current_train_context,
            'user_language': self.user_language,
            'recent_intents': [msg['intent'] for msg in self.conversation_history[-5:]]
        }
