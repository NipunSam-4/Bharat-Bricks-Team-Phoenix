"""
Translation Service using Indian Language Models
Supports CPU-only inference with IndicTrans2 or API fallback to Sarvam
"""
import requests
from typing import Dict, Optional
import config

class TranslationService:
    """Handles multilingual translation for Rail Drishti chatbot"""
    
    def __init__(self):
        self.supported_languages = config.SUPPORTED_LANGUAGES
        self.use_api = True  # Use API by default for CPU constraints
        
    def detect_language(self, text: str) -> str:
        """
        Detect language of input text
        Simple heuristic: check for Devanagari, Tamil, Telugu scripts
        """
        # Devanagari (Hindi, Marathi)
        if any('\u0900' <= char <= '\u097F' for char in text):
            return 'hi'  # Default to Hindi
        # Tamil
        elif any('\u0B80' <= char <= '\u0BFF' for char in text):
            return 'ta'
        # Telugu
        elif any('\u0C00' <= char <= '\u0C7F' for char in text):
            return 'te'
        # Bengali
        elif any('\u0980' <= char <= '\u09FF' for char in text):
            return 'bn'
        # Gujarati
        elif any('\u0A80' <= char <= '\u0AFF' for char in text):
            return 'gu'
        # Kannada
        elif any('\u0C80' <= char <= '\u0CFF' for char in text):
            return 'kn'
        # Malayalam
        elif any('\u0D00' <= char <= '\u0D7F' for char in text):
            return 'ml'
        else:
            return 'en'  # Default to English
    
    def translate_to_english(self, text: str, source_lang: Optional[str] = None) -> str:
        """Translate text to English"""
        if source_lang is None:
            source_lang = self.detect_language(text)
        
        if source_lang == 'en':
            return text
        
        # Use Sarvam API for translation
        if self.use_api and config.SARVAM_API_KEY:
            return self._translate_via_api(text, source_lang, 'en')
        
        # Fallback: return original text
        return text
    
    def translate_from_english(self, text: str, target_lang: str) -> str:
        """Translate English text to target language"""
        if target_lang == 'en':
            return text
        
        # Use Sarvam API for translation
        if self.use_api and config.SARVAM_API_KEY:
            return self._translate_via_api(text, 'en', target_lang)
        
        # Fallback: return original text
        return text
    
    def _translate_via_api(self, text: str, source_lang: str, target_lang: str) -> str:
        """Call Sarvam API for translation"""
        try:
            headers = {
                "Authorization": f"Bearer {config.SARVAM_API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "source_language": source_lang,
                "target_language": target_lang,
                "text": text
            }
            
            response = requests.post(
                config.SARVAM_API_URL,
                json=payload,
                headers=headers,
                timeout=config.PREDICTION_TIMEOUT
            )
            
            if response.status_code == 200:
                return response.json().get('translated_text', text)
            else:
                print(f"Translation API error: {response.status_code}")
                return text
                
        except Exception as e:
            print(f"Translation error: {e}")
            return text
    
    def get_language_name(self, lang_code: str) -> str:
        """Get full language name from code"""
        return self.supported_languages.get(lang_code, "English")
