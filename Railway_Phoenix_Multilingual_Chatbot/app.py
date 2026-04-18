#!/usr/bin/env python3
"""Railway Phoenix - Complete Multi-Language Support"""
import gradio as gr
import pandas as pd
import numpy as np
import pickle
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("🚂 Starting Railway Phoenix with Complete ML Translation...")

# Load model, encoders, and data
try:
    with open("models/chatbot_train_delay_predictor.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/label_encoders.pkl", "rb") as f:
        label_encoders = pickle.load(f)
    train_data = pd.read_csv("models/full_artificial_train_data.csv")
    print(f"✓ Loaded model, encoders, and {len(train_data)} train records")
    MODEL_OK = True
except Exception as e:
    print(f"⚠ Loading failed: {e}")
    MODEL_OK = False
    model, label_encoders, train_data = None, None, None

# Load multilingual translation engine
try:
    from chatbot_core import RailDrishtiChatbot
    chatbot_engine = RailDrishtiChatbot()
    translator = chatbot_engine.translator
    USE_ML_TRANSLATION = True
    print("✓ Loaded ML translation engine")
except Exception as e:
    print(f"⚠ ML translation not available: {e}")
    USE_ML_TRANSLATION = False
    translator = None

translation_cache = {}

LANGUAGES = {
    "English": "en",
    "हिंदी (Hindi)": "hi",
    "मराठी (Marathi)": "mr",
    "தமிழ் (Tamil)": "ta",
    "తెలుగు (Telugu)": "te",
    "বাংলা (Bengali)": "bn",
}

# COMPLETE fallback translations for ALL languages
TRANSLATIONS = {
    "hi": {
        # UI Elements
        "Railway Phoenix - Station-Specific Delay Predictor": "रेलवे फीनिक्स - स्टेशन-विशिष्ट विलंब भविष्यवक्ता",
        "Complete all steps below, then click Get Prediction!": "नीचे दिए सभी चरण पूरे करें, फिर भविष्यवाणी प्राप्त करें पर क्लिक करें!",
        "Step 1: Select Your Train": "चरण 1: अपनी ट्रेन चुनें",
        "Step 2: Select Your Station": "चरण 2: अपना स्टेशन चुनें",
        "Step 3: Set Current Conditions": "चरण 3: वर्तमान स्थितियां सेट करें",
        "Train Number": "ट्रेन नंबर",
        "Station": "स्टेशन",
        "Weather Conditions": "मौसम की स्थिति",
        "Time of Day": "दिन का समय",
        "Route Congestion": "मार्ग भीड़",
        "Get Prediction": "भविष्यवाणी प्राप्त करें",
        "Clear": "साफ़ करें",
        "Prediction Results": "भविष्यवाणी परिणाम",
        "stops": "स्टॉप",
        # Dropdown Options
        "Clear": "साफ़",
        "Rainy": "बरसात",
        "Foggy": "कोहरा",
        "Morning": "सुबह",
        "Afternoon": "दोपहर",
        "Evening": "शाम",
        "Night": "रात",
        "Low": "कम",
        "Medium": "मध्यम",
        "High": "उच्च",
        # Response Labels
        "Your Station": "आपका स्टेशन",
        "Route": "मार्ग",
        "Distance from Origin": "मूल से दूरी",
        "Type": "प्रकार",
        "Predicted Arrival Delay": "अनुमानित आगमन विलंब",
        "Scheduled Arrival": "निर्धारित आगमन",
        "Status": "स्थिति",
        "minutes": "मिनट",
        "Stop": "स्टॉप",
        "km": "किमी",
        "Conditions": "स्थितियां",
        "Weather": "मौसम",
        "Congestion": "भीड़",
        "Time": "समय",
        # Errors
        "Please select a train first": "कृपया पहले एक ट्रेन चुनें",
        "Please select a station first": "कृपया पहले एक स्टेशन चुनें",
        "Could not predict delay": "विलंब की भविष्यवाणी नहीं की जा सकी",
    },
    "ta": {
        # UI Elements
        "Railway Phoenix - Station-Specific Delay Predictor": "ரயில்வே ஃபீனிக்ஸ் - நிலைய சிறப்பு தாமத முன்னறிவிப்பு",
        "Complete all steps below, then click Get Prediction!": "கீழே உள்ள அனைத்து படிகளையும் முடிக்கவும், பின்னர் முன்னறிவிப்பைப் பெறுக!",
        "Step 1: Select Your Train": "படி 1: உங்கள் ரயிலைத் தேர்ந்தெடுக்கவும்",
        "Step 2: Select Your Station": "படி 2: உங்கள் நிலையத்தைத் தேர்ந்தெடுக்கவும்",
        "Step 3: Set Current Conditions": "படி 3: தற்போதைய நிலைமைகளை அமைக்கவும்",
        "Train Number": "ரயில் எண்",
        "Station": "நிலையம்",
        "Weather Conditions": "வானிலை நிலைமைகள்",
        "Time of Day": "நாளின் நேரம்",
        "Route Congestion": "பாதை நெரிசல்",
        "Get Prediction": "முன்னறிவிப்பைப் பெறுக",
        "Clear": "அழி",
        "Prediction Results": "முன்னறிவிப்பு முடிவுகள்",
        "stops": "நிறுத்தங்கள்",
        # Dropdown Options
        "Clear": "தெளிவு",
        "Rainy": "மழை",
        "Foggy": "மூடுபனி",
        "Morning": "காலை",
        "Afternoon": "பிற்பகல்",
        "Evening": "மாலை",
        "Night": "இரவு",
        "Low": "குறைவு",
        "Medium": "நடுத்தர",
        "High": "அதிக",
        # Response Labels
        "Your Station": "உங்கள் நிலையம்",
        "Route": "பாதை",
        "Distance from Origin": "தொடக்கத்திலிருந்து தூரம்",
        "Type": "வகை",
        "Predicted Arrival Delay": "முன்னறிவிக்கப்பட்ட வருகை தாமதம்",
        "Scheduled Arrival": "திட்டமிட்ட வருகை",
        "Status": "நிலை",
        "minutes": "நிமிடங்கள்",
        "Stop": "நிறுத்தம்",
        "km": "கி.மீ",
        "Conditions": "நிலைமைகள்",
        "Weather": "வானிலை",
        "Congestion": "நெரிசல்",
        "Time": "நேரம்",
        # Errors
        "Please select a train first": "தயவுசெய்து முதலில் ஒரு ரயிலைத் தேர்ந்தெடுக்கவும்",
        "Please select a station first": "தயவுசெய்து முதலில் ஒரு நிலையத்தைத் தேர்ந்தெடுக்கவும்",
        "Could not predict delay": "தாமதத்தை முன்னறிவிக்க முடியவில்லை",
    },
    "te": {
        # UI Elements
        "Railway Phoenix - Station-Specific Delay Predictor": "రైల్వే ఫీనిక్స్ - స్టేషన్-నిర్దిష్ట ఆలస్యం అంచనా",
        "Complete all steps below, then click Get Prediction!": "క్రింద ఉన్న అన్ని దశలను పూర్తి చేయండి, తర్వాత అంచనా పొందండి!",
        "Step 1: Select Your Train": "దశ 1: మీ రైలును ఎంచుకోండి",
        "Step 2: Select Your Station": "దశ 2: మీ స్టేషన్‌ను ఎంచుకోండి",
        "Step 3: Set Current Conditions": "దశ 3: ప్రస్తుత పరిస్థితులను సెట్ చేయండి",
        "Train Number": "రైలు సంఖ్య",
        "Station": "స్టేషన్",
        "Weather Conditions": "వాతావరణ పరిస్థితులు",
        "Time of Day": "రోజు సమయం",
        "Route Congestion": "మార్గం రద్దీ",
        "Get Prediction": "అంచనా పొందండి",
        "Clear": "క్లియర్",
        "Prediction Results": "అంచనా ఫలితాలు",
        "stops": "స్టాప్‌లు",
        # Dropdown Options
        "Clear": "స్పష్టం",
        "Rainy": "వర్షం",
        "Foggy": "పొగమంచు",
        "Morning": "ఉదయం",
        "Afternoon": "మధ్యాహ్నం",
        "Evening": "సాయంకాలం",
        "Night": "రాత్రి",
        "Low": "తక్కువ",
        "Medium": "మధ్యస్థం",
        "High": "ఎక్కువ",
        # Response Labels
        "Your Station": "మీ స్టేషన్",
        "Route": "మార్గం",
        "Distance from Origin": "మూలం నుండి దూరం",
        "Type": "రకం",
        "Predicted Arrival Delay": "అంచనా వేసిన రాక ఆలస్యం",
        "Scheduled Arrival": "షెడ్యూల్ చేసిన రాక",
        "Status": "స్థితి",
        "minutes": "నిమిషాలు",
        "Stop": "స్టాప్",
        "km": "కి.మీ",
        "Conditions": "పరిస్థితులు",
        "Weather": "వాతావరణం",
        "Congestion": "రద్దీ",
        "Time": "సమయం",
        # Errors
        "Please select a train first": "దయచేసి ముందుగా రైలును ఎంచుకోండి",
        "Please select a station first": "దయచేసి ముందుగా స్టేషన్‌ను ఎంచుకోండి",
        "Could not predict delay": "ఆలస్యం అంచనా వేయలేకపోయింది",
    },
    "bn": {
        # UI Elements
        "Railway Phoenix - Station-Specific Delay Predictor": "রেলওয়ে ফিনিক্স - স্টেশন-নির্দিষ্ট বিলম্ব পূর্বাভাসকারী",
        "Complete all steps below, then click Get Prediction!": "নীচের সমস্ত ধাপ সম্পূর্ণ করুন, তারপর পূর্বাভাস পান ক্লিক করুন!",
        "Step 1: Select Your Train": "ধাপ ১: আপনার ট্রেন নির্বাচন করুন",
        "Step 2: Select Your Station": "ধাপ ২: আপনার স্টেশন নির্বাচন করুন",
        "Step 3: Set Current Conditions": "ধাপ ৩: বর্তমান অবস্থা সেট করুন",
        "Train Number": "ট্রেন নম্বর",
        "Station": "স্টেশন",
        "Weather Conditions": "আবহাওয়া পরিস্থিতি",
        "Time of Day": "দিনের সময়",
        "Route Congestion": "রুট যানজট",
        "Get Prediction": "পূর্বাভাস পান",
        "Clear": "সাফ করুন",
        "Prediction Results": "পূর্বাভাস ফলাফল",
        "stops": "স্টপ",
        # Dropdown Options
        "Clear": "পরিষ্কার",
        "Rainy": "বৃষ্টি",
        "Foggy": "কুয়াশা",
        "Morning": "সকাল",
        "Afternoon": "দুপুর",
        "Evening": "সন্ধ্যা",
        "Night": "রাত",
        "Low": "কম",
        "Medium": "মাঝারি",
        "High": "উচ্চ",
        # Response Labels
        "Your Station": "আপনার স্টেশন",
        "Route": "রুট",
        "Distance from Origin": "উৎস থেকে দূরত্ব",
        "Type": "প্রকার",
        "Predicted Arrival Delay": "পূর্বাভাসিত আগমন বিলম্ব",
        "Scheduled Arrival": "নির্ধারিত আগমন",
        "Status": "অবস্থা",
        "minutes": "মিনিট",
        "Stop": "স্টপ",
        "km": "কি.মি",
        "Conditions": "অবস্থা",
        "Weather": "আবহাওয়া",
        "Congestion": "যানজট",
        "Time": "সময়",
        # Errors
        "Please select a train first": "দয়া করে প্রথমে একটি ট্রেন নির্বাচন করুন",
        "Please select a station first": "দয়া করে প্রথমে একটি স্টেশন নির্বাচন করুন",
        "Could not predict delay": "বিলম্বের পূর্বাভাস দিতে পারেনি",
    },
    "mr": {
        # UI Elements
        "Railway Phoenix - Station-Specific Delay Predictor": "रेल्वे फिनिक्स - स्टेशन-विशिष्ट विलंब अंदाजकर्ता",
        "Complete all steps below, then click Get Prediction!": "खाली सर्व चरण पूर्ण करा, नंतर अंदाज मिळवा क्लिक करा!",
        "Step 1: Select Your Train": "चरण 1: तुमची ट्रेन निवडा",
        "Step 2: Select Your Station": "चरण 2: तुमचे स्टेशन निवडा",
        "Step 3: Set Current Conditions": "चरण 3: सध्याच्या परिस्थिती सेट करा",
        "Train Number": "ट्रेन क्रमांक",
        "Station": "स्टेशन",
        "Weather Conditions": "हवामान परिस्थिती",
        "Time of Day": "दिवसाचा वेळ",
        "Route Congestion": "मार्ग गर्दी",
        "Get Prediction": "अंदाज मिळवा",
        "Clear": "साफ करा",
        "Prediction Results": "अंदाज निकाल",
        "stops": "थांबे",
        # Dropdown Options
        "Clear": "स्वच्छ",
        "Rainy": "पाऊस",
        "Foggy": "धुके",
        "Morning": "सकाळ",
        "Afternoon": "दुपार",
        "Evening": "संध्याकाळ",
        "Night": "रात्र",
        "Low": "कमी",
        "Medium": "मध्यम",
        "High": "उच्च",
        # Response Labels
        "Your Station": "तुमचे स्टेशन",
        "Route": "मार्ग",
        "Distance from Origin": "मूळ पासून अंतर",
        "Type": "प्रकार",
        "Predicted Arrival Delay": "अंदाजित आगमन विलंब",
        "Scheduled Arrival": "नियोजित आगमन",
        "Status": "स्थिती",
        "minutes": "मिनिटे",
        "Stop": "थांबा",
        "km": "किमी",
        "Conditions": "परिस्थिती",
        "Weather": "हवामान",
        "Congestion": "गर्दी",
        "Time": "वेळ",
        # Errors
        "Please select a train first": "कृपया प्रथम ट्रेन निवडा",
        "Please select a station first": "कृपया प्रथम स्टेशन निवडा",
        "Could not predict delay": "विलंबाचा अंदाज लावू शकत नाही",
    }
}

def translate(text, lang_code):
    """Translate with instant fallbacks"""
    if lang_code == "en":
        return text
    
    # Check cache
    cache_key = f"{text}|{lang_code}"
    if cache_key in translation_cache:
        return translation_cache[cache_key]
    
    # Use fallback translations (instant)
    if lang_code in TRANSLATIONS and text in TRANSLATIONS[lang_code]:
        translated = TRANSLATIONS[lang_code][text]
        translation_cache[cache_key] = translated
        return translated
    
    # Try ML API as backup
    if USE_ML_TRANSLATION:
        try:
            translated = translator.translate_from_english(text, lang_code)
            if translated and translated != text:
                translation_cache[cache_key] = translated
                return translated
        except Exception as e:
            print(f"API translation failed for '{text}': {e}")
    
    # Fallback to English
    return text

def get_all_trains():
    if not MODEL_OK:
        return []
    trains = train_data[["Train No", "Train Name"]].drop_duplicates().sort_values("Train No")
    return [f"{row['Train No']} - {row['Train Name']}" for _, row in trains.iterrows()]

def get_train_stations(train_selection):
    if not MODEL_OK or not train_selection:
        return []
    try:
        train_num = int(train_selection.split("-")[0].strip())
        stations = train_data[train_data["Train No"] == train_num].sort_values("SEQ")
        return [f"{row['Station Name']} ({row['Distance']} km)" for _, row in stations.iterrows()]
    except:
        return []

def update_all_ui(language):
    lang_code = LANGUAGES.get(language, "en")
    print(f"🌍 Switching to {language} - Clearing all inputs and predictions")
    
    # Translate all text
    title = translate("Railway Phoenix - Station-Specific Delay Predictor", lang_code)
    subtitle = translate("Complete all steps below, then click Get Prediction!", lang_code)
    step1 = translate("Step 1: Select Your Train", lang_code)
    step2 = translate("Step 2: Select Your Station", lang_code)
    step3 = translate("Step 3: Set Current Conditions", lang_code)
    
    train_label = translate("Train Number", lang_code)
    station_label = translate("Station", lang_code)
    weather_label = translate("Weather Conditions", lang_code)
    time_label = translate("Time of Day", lang_code)
    congestion_label = translate("Route Congestion", lang_code)
    
    get_pred = translate("Get Prediction", lang_code)
    clear_btn = translate("Clear", lang_code)
    results = translate("Prediction Results", lang_code)
    
    # Translate choices
    weather_choices = [translate("Clear", lang_code), translate("Rainy", lang_code), translate("Foggy", lang_code)]
    time_choices = [translate("Morning", lang_code), translate("Afternoon", lang_code), 
                   translate("Evening", lang_code), translate("Night", lang_code)]
    congestion_choices = [translate("Low", lang_code), translate("Medium", lang_code), translate("High", lang_code)]
    
    print(f"✓ UI updated to {language} - All selections cleared")
    
    return (
        gr.Markdown(value=f"# 🚂 {title}"),
        gr.Markdown(value=f"**{subtitle}**"),
        gr.Markdown(value=f"### 🚂 {step1}"),
        gr.Markdown(value=f"### 🚉 {step2}"),
        gr.Markdown(value=f"### 🌤️ {step3}"),
        gr.Dropdown(label=train_label, value=None),  # CLEAR train selection
        gr.Dropdown(label=station_label, choices=[], value=None, visible=False),  # CLEAR + HIDE station
        gr.Radio(label=f"☁️ {weather_label}", choices=weather_choices, value=weather_choices[0]),
        gr.Radio(label=f"🕐 {time_label}", choices=time_choices, value=time_choices[0]),
        gr.Radio(label=f"🚦 {congestion_label}", choices=congestion_choices, value=congestion_choices[0]),
        gr.Button(value=f"🔍 {get_pred}", variant="primary", size="lg"),
        gr.Button(value=f"🔄 {clear_btn}", variant="secondary"),
        gr.Markdown(value=f"### 💬 {results}"),
        ""  # CLEAR prediction display
    )

def update_station_dropdown(train_selection, language):
    lang_code = LANGUAGES.get(language, "en")
    station_label = translate("Station", lang_code)
    
    if not train_selection:
        return gr.Dropdown(choices=[], value=None, visible=False)
    
    stations = get_train_stations(train_selection)
    if not stations:
        return gr.Dropdown(choices=[], value=None, visible=False)
    
    stops_text = translate("stops", lang_code)
    return gr.Dropdown(choices=stations, value=stations[0], visible=True,
                      label=f"{station_label} ({len(stations)} {stops_text})", interactive=True)

def reverse_translate(text, lang_code):
    if lang_code == "en":
        return text
    if lang_code in TRANSLATIONS:
        for eng, trans in TRANSLATIONS[lang_code].items():
            if trans == text:
                return eng
    return text

def predict_station_delay(train_selection, station_name, weather, congestion):
    if not MODEL_OK:
        return None
    try:
        train_num = int(train_selection.split("-")[0].strip())
        if "(" in station_name:
            station_name = station_name.split("(")[0].strip()
        
        station_data = train_data[(train_data["Train No"] == train_num) & 
                                 (train_data["Station Name"] == station_name)]
        if len(station_data) == 0:
            return None
        
        station_row = station_data.iloc[0]
        distance = station_row["Distance"]
        
        weather_enc = label_encoders["Weather Conditions"].transform([weather])[0]
        congestion_enc = label_encoders["Route Congestion"].transform([congestion])[0]
        
        features = np.array([[distance, weather_enc, congestion_enc]])
        predicted_delay = model.predict(features)[0]
        predicted_delay = max(0, int(predicted_delay))
        
        return {
            "delay": predicted_delay, "train_num": train_num, "station": station_row["Station Name"],
            "distance": distance, "sequence": station_row["SEQ"], "arrival_time": station_row["Arrival time"],
            "train_name": station_row["Train Name"], "train_type": station_row["Train Type"],
            "source": station_row["Source Station Name"], "destination": station_row["Destination Station Name"]
        }
    except Exception as e:
        print(f"Prediction error: {e}")
        return None

def get_prediction(train_selection, selected_station, weather_sel, time_sel,
                  congestion_sel, language):
    try:
        lang_code = LANGUAGES.get(language, "en")
        
        print(f"🔮 Generating prediction in {language}")
        
        if not train_selection:
            error = translate("Please select a train first", lang_code)
            return f"❌ {error}"
        
        if not selected_station:
            error = translate("Please select a station first", lang_code)
            return f"❌ {error}"
        
        # Reverse translate
        weather = reverse_translate(weather_sel, lang_code)
        congestion = reverse_translate(congestion_sel, lang_code)
        
        print(f"  Reverse: {weather_sel} → {weather}, {congestion_sel} → {congestion}")
        
        result = predict_station_delay(train_selection, selected_station, weather, congestion)
        
        if not result:
            error = translate("Could not predict delay", lang_code)
            return f"❌ {error}"
        
        delay = result["delay"]
        emoji = "✅" if delay < 15 else "⚠️" if delay < 45 else "❌"
        status = "🟢" if delay < 15 else "🟡" if delay < 45 else "🔴"
        
        # Translate ALL labels
        your_station = translate("Your Station", lang_code)
        route = translate("Route", lang_code)
        distance = translate("Distance from Origin", lang_code)
        train_type = translate("Type", lang_code)
        pred_delay = translate("Predicted Arrival Delay", lang_code)
        scheduled = translate("Scheduled Arrival", lang_code)
        status_label = translate("Status", lang_code)
        minutes = translate("minutes", lang_code)
        stop = translate("Stop", lang_code)
        conditions = translate("Conditions", lang_code)
        weather_label = translate("Weather", lang_code)
        congestion_label = translate("Congestion", lang_code)
        time_label = translate("Time", lang_code)
        
        print(f"✓ Labels translated: {your_station}, {route}, {pred_delay}")
        
        # Create a prominent boxed display
        reply = f"""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 15px; color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">

<h3 style="color: white; margin-top: 0;">{emoji} <strong>{result['train_name']}</strong> (#{result['train_num']})</h3>

<div style="background: rgba(255,255,255,0.95); padding: 15px; border-radius: 10px; color: #333; margin: 10px 0;">
<table style="width: 100%; border-collapse: collapse;">
<tr><td style="padding: 8px; font-weight: bold;">🚉 {your_station}</td><td style="padding: 8px;">{result['station']} ({stop} #{result['sequence']})</td></tr>
<tr><td style="padding: 8px; font-weight: bold;">📍 {route}</td><td style="padding: 8px;">{result['source']} → {result['destination']}</td></tr>
<tr><td style="padding: 8px; font-weight: bold;">📏 {distance}</td><td style="padding: 8px;">{result['distance']} km</td></tr>
<tr><td style="padding: 8px; font-weight: bold;">🚄 {train_type}</td><td style="padding: 8px;">{result['train_type']}</td></tr>
</table>
</div>

<div style="background: rgba(255,255,255,0.95); padding: 15px; border-radius: 10px; color: #333; margin: 10px 0;">
<table style="width: 100%; border-collapse: collapse;">
<tr><td style="padding: 8px; font-weight: bold;">⏱️ {pred_delay}</td><td style="padding: 8px;"><span style="color: #e74c3c; font-size: 1.2em; font-weight: bold;">{delay} {minutes}</span></td></tr>
<tr><td style="padding: 8px; font-weight: bold;">⏰ {scheduled}</td><td style="padding: 8px;">{result['arrival_time']}</td></tr>
<tr><td style="padding: 8px; font-weight: bold;">📊 {status_label}</td><td style="padding: 8px;"><span style="font-weight: bold; color: #3498db;">{status}</span></td></tr>
</table>
</div>

<div style="background: rgba(255,255,255,0.95); padding: 15px; border-radius: 10px; color: #333; margin: 10px 0;">
<h4 style="margin-top: 0; color: #333;"><strong>{conditions}:</strong></h4>
<table style="width: 100%; border-collapse: collapse;">
<tr><td style="padding: 8px; font-weight: bold;">☁️ {weather_label}</td><td style="padding: 8px;">{weather_sel}</td></tr>
<tr><td style="padding: 8px; font-weight: bold;">🚦 {congestion_label}</td><td style="padding: 8px;">{congestion_sel}</td></tr>
<tr><td style="padding: 8px; font-weight: bold;">🕐 {time_label}</td><td style="padding: 8px;">{time_sel}</td></tr>
</table>
</div>

</div>
"""
        
        print(f"✓ Prediction generated successfully")
        return reply
        
    except Exception as e:
        print(f"❌ ERROR in get_prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"❌ **Error:** {str(e)}\n\nPlease try again or check the logs."

print("🎨 Creating complete multilingual interface...")


# Custom CSS for Railway Phoenix
custom_css = """
/* Global styling */
.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* Navbar styling */
.navbar-title {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    padding: 25px !important;
    text-align: center !important;
    border-radius: 15px !important;
    margin-bottom: 25px !important;
    box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4) !important;
}

.navbar-title h1 {
    color: white !important;
    margin: 0 !important;
    font-size: 2.2em !important;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
}

/* Ensure no overflow/scroll on any Gradio row */
.gradio-row {
    overflow: visible !important;
}

/* Text center utility */
.text-center {
    text-align: center;
}

/* Input box styling - REMOVED HOVER EFFECT to prevent flickering */
.input-box {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    padding: 25px;
    border-radius: 15px;
    margin: 20px 0;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    border: 3px solid #667eea;
}

.step-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white !important;
    padding: 12px 20px;
    border-radius: 10px;
    margin-bottom: 15px !important;
    font-weight: bold;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.step-header h3 {
    color: white !important;
    margin: 0 !important;
}

/* Button styling */
button.primary-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 18px 40px !important;
    font-size: 1.2em !important;
    font-weight: bold !important;
    box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4) !important;
    transition: all 0.3s ease !important;
    color: white !important;
}

button.primary-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 16px rgba(102, 126, 234, 0.5) !important;
}

button.secondary-btn {
    border-radius: 12px !important;
    padding: 18px 40px !important;
    font-size: 1.1em !important;
    transition: all 0.3s ease !important;
}

button.secondary-btn:hover {
    transform: translateY(-2px) !important;
}

/* Results box */
.results-box {
    margin-top: 20px;
    border-radius: 15px;
    padding: 25px;
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    border: 3px solid #667eea;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    min-height: 100px;
}

/* Dropdown styling */
.gradio-dropdown {
    border-radius: 10px !important;
}

/* Radio button group styling */

/* Prediction output styling */
.prediction-output {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    padding: 20px;
    border-radius: 15px;
    border: 3px solid #667eea;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    min-height: 100px;
    margin-top: 10px;
}
.gradio-radio {
    background: white;
    padding: 15px;
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}
"""


with gr.Blocks(title="Railway Phoenix", css=custom_css) as demo:
    # Navbar
    title_md = gr.Markdown("# 🚂 Railway Phoenix - Station-Specific Delay Predictor", elem_classes="navbar-title")
    
    subtitle_md = gr.Markdown("**Complete all steps below, then click Get Prediction!**", elem_classes="text-center")
    
    # Language Selector (outside boxes)
    language_selector = gr.Dropdown(
        choices=list(LANGUAGES.keys()), value="English",
        label="🌍 Select Language / भाषा / மொழி / భాష / ভাষা / भाषा",
        interactive=True
    )
    
    gr.Markdown("")  # Spacing
    
    # Step 1 Box
    with gr.Group(elem_classes="input-box"):
        step1_md = gr.Markdown("### 🚂 Step 1: Select Your Train", elem_classes="step-header")
        train_selector = gr.Dropdown(choices=get_all_trains(), label="Train Number", 
                                    interactive=True, filterable=True)
    
    # Step 2 Box
    with gr.Group(elem_classes="input-box"):
        step2_md = gr.Markdown("### 🚉 Step 2: Select Your Station", elem_classes="step-header")
        station_selector = gr.Dropdown(choices=[], label="Station", visible=False, interactive=True)
    
    # Step 3 Box
    with gr.Group(elem_classes="input-box"):
        step3_md = gr.Markdown("### 🌤️ Step 3: Set Current Conditions", elem_classes="step-header")
        
        with gr.Row():
            weather = gr.Radio(choices=["Clear", "Rainy", "Foggy"], value="Clear", 
                              label="☁️ Weather Conditions", scale=1)
            time_of_day = gr.Radio(choices=["Morning", "Afternoon", "Evening", "Night"], 
                                  value="Morning", label="🕐 Time of Day", scale=1, min_width=500, elem_classes="time-radio")
            congestion = gr.Radio(choices=["Low", "Medium", "High"], value="Low", 
                                 label="🚦 Route Congestion", scale=1)
    
    # Action Buttons
    gr.Markdown("")  # Spacing
    with gr.Row():
        send = gr.Button("🔍 Get Prediction", variant="primary", size="lg", elem_classes="primary-btn")
        clear = gr.Button("🔄 Clear", variant="secondary", elem_classes="secondary-btn")
    
    # Results Section
    gr.Markdown("")  # Spacing
    results_md = gr.Markdown("### 💬 Prediction Results")
    
    # Prediction display outside of Group to avoid DOM conflicts
    prediction_display = gr.HTML(value="", visible=True, elem_classes="prediction-output")
    
    # Footer
    gr.Markdown("")
    gr.Markdown("✨ **Complete Multilingual Support** - 6 languages with instant translations", elem_classes="text-center")
    
    # Events
    language_selector.change(update_all_ui, language_selector,
        [title_md, subtitle_md, step1_md, step2_md, step3_md, train_selector, 
         station_selector, weather, time_of_day, congestion, send, clear, results_md, prediction_display])
    
    train_selector.change(update_station_dropdown, [train_selector, language_selector], station_selector)
    
    send.click(get_prediction, [train_selector, station_selector, weather, time_of_day, 
                                congestion, language_selector], prediction_display)
    
    clear.click(lambda: (None, gr.Dropdown(choices=[], visible=False), ""), 
               None, [train_selector, station_selector, prediction_display])

if __name__ == "__main__":
    print("🚀 Launching on 0.0.0.0:8000...")
    print("✅ Complete translations for Hindi, Tamil, Telugu, Bengali, Marathi")
    demo.launch(server_name="0.0.0.0", server_port=8000, show_error=True)
