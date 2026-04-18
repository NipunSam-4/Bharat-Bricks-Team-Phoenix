"""
Query Processor Service
Parses user queries, extracts train information, and determines intent
"""
import re
from typing import Dict, Optional, List, Tuple
from datetime import datetime

class QueryProcessor:
    """Processes natural language queries about trains"""
    
    def __init__(self):
        # Intent keywords for different query types
        self.intent_patterns = {
            'delay_query': [
                'delay', 'late', 'delayed', 'when will', 'reach', 'arrive',
                'विलंब', 'देरी', 'कब पहुंचेगी', 'कब आएगी'  # Hindi
            ],
            'status_query': [
                'status', 'where', 'location', 'current', 'position',
                'स्थिति', 'कहां है', 'स्थान'  # Hindi
            ],
            'reason_query': [
                'why', 'reason', 'cause', 'because',
                'क्यों', 'कारण'  # Hindi
            ],
            'schedule_query': [
                'schedule', 'time', 'timing', 'timetable', 'departure', 'arrival',
                'समय', 'समय-सारणी'  # Hindi
            ]
        }
        
        # Common train number patterns
        self.train_number_pattern = r'\b(\d{5})\b'
        
    def process_query(self, query: str, user_language: str = 'en') -> Dict:
        """
        Process a user query and extract structured information
        
        Returns:
            Dict with: intent, train_number, entities, confidence
        """
        query_lower = query.lower()
        
        # Extract train number
        train_number = self._extract_train_number(query)
        
        # Determine intent
        intent = self._determine_intent(query_lower)
        
        # Extract entities (stations, dates, etc.)
        entities = self._extract_entities(query_lower)
        
        # Calculate confidence
        confidence = self._calculate_confidence(train_number, intent, entities)
        
        return {
            'intent': intent,
            'train_number': train_number,
            'entities': entities,
            'confidence': confidence,
            'original_query': query,
            'language': user_language
        }
    
    def _extract_train_number(self, text: str) -> Optional[str]:
        """Extract train number from text"""
        match = re.search(self.train_number_pattern, text)
        return match.group(1) if match else None
    
    def _determine_intent(self, query_lower: str) -> str:
        """Determine the primary intent of the query"""
        intent_scores = {}
        
        for intent, keywords in self.intent_patterns.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                intent_scores[intent] = score
        
        if not intent_scores:
            return 'general_query'
        
        # Return intent with highest score
        return max(intent_scores, key=intent_scores.get)
    
    def _extract_entities(self, query_lower: str) -> Dict:
        """Extract entities like station names, dates, etc."""
        entities = {
            'stations': [],
            'date': None,
            'time': None
        }
        
        # Extract common station patterns (simplified)
        # In production, use a station name database
        station_keywords = ['from', 'to', 'at', 'in']
        
        # Extract potential station names (words after "from"/"to")
        words = query_lower.split()
        for i, word in enumerate(words):
            if word in ['from', 'starting'] and i + 1 < len(words):
                entities['stations'].append(('origin', words[i + 1]))
            elif word in ['to', 'destination'] and i + 1 < len(words):
                entities['stations'].append(('destination', words[i + 1]))
        
        # Extract date references
        if 'today' in query_lower:
            entities['date'] = datetime.now().date()
        elif 'tomorrow' in query_lower:
            entities['date'] = (datetime.now().date().replace(day=datetime.now().day + 1))
        
        return entities
    
    def _calculate_confidence(self, train_number: Optional[str], 
                             intent: str, entities: Dict) -> float:
        """Calculate confidence score for the parsed query"""
        confidence = 0.0
        
        # Train number provides high confidence
        if train_number:
            confidence += 0.5
        
        # Clear intent adds confidence
        if intent != 'general_query':
            confidence += 0.3
        
        # Entities add confidence
        if entities['stations']:
            confidence += 0.1
        if entities['date']:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def format_response_template(self, intent: str, language: str = 'en') -> str:
        """Get response template based on intent and language"""
        templates = {
            'en': {
                'delay_query': "Train {train_name} (#{train_no}) is currently at {current_station}. Expected delay: {delay} minutes. Estimated arrival: {eta}.",
                'status_query': "Train {train_name} (#{train_no}) is currently at {current_station}, traveling towards {destination}.",
                'reason_query': "The delay is primarily due to: {reason}. {additional_details}",
                'schedule_query': "Train {train_name} is scheduled to depart {origin} at {departure_time} and arrive at {destination} at {arrival_time}.",
                'general_query': "I can help you with train delays, status, and schedules. Please provide a train number."
            },
            'hi': {
                'delay_query': "ट्रेन {train_name} (#{train_no}) वर्तमान में {current_station} पर है। अपेक्षित विलंब: {delay} मिनट। अनुमानित आगमन: {eta}।",
                'status_query': "ट्रेन {train_name} (#{train_no}) वर्तमान में {current_station} पर है, {destination} की ओर जा रही है।",
                'reason_query': "विलंब का मुख्य कारण है: {reason}। {additional_details}",
                'schedule_query': "ट्रेन {train_name} {origin} से {departure_time} बजे प्रस्थान करती है और {destination} पर {arrival_time} बजे पहुंचती है।",
                'general_query': "मैं ट्रेन विलंब, स्थिति और समय-सारणी में आपकी मदद कर सकता हूं। कृपया ट्रेन नंबर प्रदान करें।"
            }
        }
        
        lang_templates = templates.get(language, templates['en'])
        return lang_templates.get(intent, lang_templates['general_query'])
    
    def validate_train_number(self, train_number: str) -> bool:
        """Validate if train number format is correct"""
        if not train_number:
            return False
        return bool(re.match(r'^\d{5}$', train_number))
