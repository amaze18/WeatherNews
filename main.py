# --- Import required libraries and modules ---
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Union
from news_weather_agent import (
    persona_response,
    generate_weekly_news_summary,
    check_and_alert_for_major_events,
    check_and_alert_for_weather_user,
    check_and_alert_for_weather_bot,
    check_and_alert_for_major_events_bot
)
from bot_prompt import get_bot_prompt
import time as pytime
import uvicorn
import os
from dotenv import load_dotenv
load_dotenv()
from supabase import Client, create_client
from datetime import datetime, timezone, time, timedelta
import logging
from fastapi_utils.tasks import repeat_every
from apscheduler.schedulers.background import BackgroundScheduler
import asyncio
import random
import hashlib
from fastapi.middleware.cors import CORSMiddleware

# --- Initialize FastAPI app ---
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Supabase connection details ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- Pydantic request model ---
class QuestionRequest(BaseModel):
    message: Union[str, None] = None
    bot_id: str = "delhi"
    custom_bot_name: str = ""
    user_name: str = ""
    user_gender: str = ""
    user_location: str = ""
    language: str = ""
    traits: str = ""
    previous_conversation: list = []
    email: str = ""
    request_time: str = ""
    platform: str = ""

# --- Helper functions ---
def get_today_user_bot_pairs():
    try:
        utc_now = datetime.now(timezone.utc)
        start = datetime.combine(utc_now.date(), time.min, tzinfo=timezone.utc).isoformat()
        end = datetime.combine(utc_now.date(), time.max, tzinfo=timezone.utc).isoformat()
        response = supabase.table("message_paritition") \
            .select("email, bot_id") \
            .gte("created_at", start) \
            .lte("created_at", end) \
            .execute()
        pairs = {(item["email"], item["bot_id"]) for item in response.data if item.get("email") and item.get("bot_id")}
        return list(pairs)
    except Exception as e:
        logging.error(f"Exception in get_today_user_bot_pairs: {e}")
        return []

def get_all_news_agent_params():
    pairs = get_today_user_bot_pairs()
    params_list = []
    for email, bot_id in pairs:
        user_details = supabase.table("user_details").select("*").eq("email", email).single().execute().data or {}
        bot_details = supabase.table("bot_personality_details").select("*").eq("bot_id", bot_id).single().execute().data or {}
        user_name = user_details.get("name", email.split("@")[0])
        user_gender = user_details.get("gender", "Other")
        user_city = user_details.get("city") or "India"
        custom_bot_name = bot_details.get("bot_name", bot_id)
        bot_city = bot_details.get("bot_city", "India")
        language = "English"
        traits = ""
        raw_bot_prompt = get_bot_prompt(bot_id)
        bot_prompt = raw_bot_prompt.format(
            custom_bot_name=custom_bot_name,
            traitsString=traits,
            userName=user_name,
            userGender=user_gender,
            languageString=language
        )
        params_list.append({
            "email": email,
            "bot_id": bot_id,
            "user_name": user_name,
            "user_gender": user_gender,
            "custom_bot_name": custom_bot_name,
            "language": language,
            "traits": traits,
            "bot_prompt": bot_prompt,
            "bot_city": bot_city,
            "user_location": user_city
        })
    return params_list

def insert_bot_message(email, bot_id, message, activity_name="bot_alert"):
    """
    Enhanced function to insert bot messages with activity tracking.
    """
    return log_activity_message_to_supabase(
        email=email,
        bot_id=bot_id,
        user_message="",
        bot_response=message,
        platform="weather_news",
        activity_name=activity_name
    )

def log_activity_message_to_supabase(email, bot_id, user_message, bot_response, platform="weather_news", activity_name=None):
    """
    Enhanced function to log activity messages to Supabase with platform and activity tracking.
    Based on the reference implementation from gaming agents project.
    """
    now = datetime.utcnow().isoformat()
    
    # Ensure bot_response is a string and handle Unicode properly
    if isinstance(bot_response, dict):
        bot_response = bot_response.get("raw") or bot_response.get("response") or str(bot_response)
    
    # Ensure bot_response is properly encoded as string
    if not isinstance(bot_response, str):
        bot_response = str(bot_response)
    
    # Handle potential Unicode issues by encoding/decoding
    try:
        bot_response = bot_response.encode('utf-8', errors='ignore').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        bot_response = str(bot_response)
    
    data = {
        "email": email,
        "bot_id": bot_id,
        "user_message": user_message,
        "bot_response": bot_response,
        "requested_time": now,
        "platform": platform
    }
    
    if activity_name:
        data["activity_name"] = activity_name
    
    print(f"[SUPABASE] Attempting to store message for {email} with bot {bot_id}")
    print(f"[SUPABASE] Platform: {platform}")
    print(f"[SUPABASE] Activity: {activity_name or 'N/A'}")
    print(f"[SUPABASE] User message: {user_message[:100]}{'...' if len(user_message) > 100 else ''}")
    print(f"[SUPABASE] Bot response: {bot_response[:100]}{'...' if len(bot_response) > 100 else ''}")
    
    try:
        response = supabase.table("message_paritition").insert(data).execute()
        
        if response.data and len(response.data) > 0:
            record_id = response.data[0].get('id', 'N/A')
            print(f"[SUPABASE] ✅ SUCCESS: Message stored with ID {record_id}")
            print(f"[SUPABASE] ✅ Email: {email}")
            print(f"[SUPABASE] ✅ Bot ID: {bot_id}")
            print(f"[SUPABASE] ✅ Platform: {platform}")
            print(f"[SUPABASE] ✅ Activity: {activity_name or 'N/A'}")
            print(f"[SUPABASE] ✅ Created at: {response.data[0].get('created_at', 'N/A')}")
            logging.info(f"Successfully stored message for {email} with ID {record_id}")
            return True
        else:
            print(f"[SUPABASE] ❌ ERROR: No data returned from insert operation")
            logging.error(f"Insert operation returned no data for {email}")
            return False
            
    except Exception as e:
        print(f"[SUPABASE] ❌ ERROR: Failed to insert message: {e}")
        print(f"[SUPABASE] ❌ Email: {email}")
        print(f"[SUPABASE] ❌ Bot ID: {bot_id}")
        print(f"[SUPABASE] ❌ Platform: {platform}")
        logging.error(f"Failed to insert message for {email}: {e}")
        return False

def determine_activity_type(user_message):
    """
    Determine the activity type based on the user's message content.
    Returns appropriate activity_name for weather news interactions.
    """
    if not user_message:
        return "general_chat"
    
    message_lower = user_message.lower()
    
    # Weather-related keywords
    weather_keywords = [
        "weather", "temperature", "rain", "sunny", "cloudy", "forecast", 
        "hot", "cold", "humidity", "wind", "storm", "snow", "climate"
    ]
    
    # News-related keywords
    news_keywords = [
        "news", "latest", "happening", "event", "incident", "breaking", 
        "update", "report", "story", "headline", "current"
    ]
    
    # Check for weather-related content
    if any(keyword in message_lower for keyword in weather_keywords):
        if "temperature" in message_lower or "temp" in message_lower:
            return "temperature_query"
        elif "forecast" in message_lower:
            return "weather_forecast"
        elif any(word in message_lower for word in ["rain", "storm", "snow", "wind"]):
            return "weather_alert"
        else:
            return "current_weather"
    
    # Check for news-related content
    elif any(keyword in message_lower for keyword in news_keywords):
        return "news_query"
    
    # Default to general chat
    else:
        return "general_chat"

def insert_user_message(email, bot_id, user_message, bot_response):
    """
    Legacy function for backward compatibility - now calls the enhanced logging function.
    """
    activity_name = determine_activity_type(user_message)
    return log_activity_message_to_supabase(
        email=email,
        bot_id=bot_id,
        user_message=user_message,
        bot_response=bot_response,
        platform="weather_news",
        activity_name=activity_name
    )

# --- Core logic for both endpoints ---
async def handle_news_weather_agent(request: QuestionRequest):
    raw_bot_prompt = get_bot_prompt(request.bot_id)
    bot_prompt = raw_bot_prompt.format(
        custom_bot_name=request.custom_bot_name,
        traitsString=request.traits,
        userName=request.user_name,
        userGender=request.user_gender,
        languageString=request.language
    )
    start = pytime.time()
    response = await persona_response(
        user_message=request.message,
        persona_prompt=bot_prompt,
        language=request.language,
        user_name=request.user_name,
        user_location=request.user_location
    )
    end = pytime.time()
    print(f"[DEBUG] Time taken for persona_response: {end - start} seconds")
    
    # Save the conversation to the database if email is provided
    print(f"[CHAT] Processing chat request from {request.email or 'NO EMAIL'}")
    print(f"[CHAT] Bot ID: {request.bot_id}")
    print(f"[CHAT] User message: {request.message or 'NO MESSAGE'}")
    
    if request.email and request.email.strip():
        print(f"[CHAT] ✅ Email provided: {request.email} - will save to Supabase")
        try:
            # Determine activity type based on user message
            activity_name = determine_activity_type(request.message)
            
            storage_success = log_activity_message_to_supabase(
                email=request.email,
                bot_id=request.bot_id,
                user_message=request.message or "",
                bot_response=response,
                platform="weather_news",
                activity_name=activity_name
            )
            if storage_success:
                print(f"[CHAT] ✅ Conversation successfully saved to Supabase for {request.email}")
                print(f"[CHAT] ✅ Activity type: {activity_name}")
            else:
                print(f"[CHAT] ❌ Failed to save conversation to Supabase for {request.email}")
        except Exception as e:
            print(f"[CHAT] ❌ Exception while saving conversation: {e}")
            logging.error(f"Failed to save conversation for {request.email}: {e}")
    else:
        print("[CHAT] ⚠️  WARNING: No email provided, conversation NOT saved to database")
        print("[CHAT] ⚠️  To save conversations, include 'email' field in your request")
    
    return {
        "response": response,
        "user_name": request.user_name,
        "language": request.language
    }

# --- Endpoints ---
@app.post("/news_weather_agent")
async def news_weather_agent(request: QuestionRequest):
    return await handle_news_weather_agent(request)

@app.post("/weather/news_weather_agent")
async def news_weather_agent_alias(request: QuestionRequest):
    return await handle_news_weather_agent(request)

# --- Scheduled tasks ---
@repeat_every(seconds=60*60*24*7)
def scheduled_weekly_news_summary():
    for params in get_all_news_agent_params():
        summary = generate_weekly_news_summary(
            params["bot_prompt"],
            params["user_name"],
            params["language"],
            params["bot_city"],
            params["user_location"]
        )
        insert_bot_message(
            email=params["email"],
            bot_id=params["bot_id"],
            message=summary,
            activity_name="weekly_news_summary"
        )

def scheduled_major_event_alert():
    for params in get_all_news_agent_params():
        alert = check_and_alert_for_major_events(
            params["bot_prompt"],
            params["user_name"],
            params["language"],
            params["bot_city"],
            params["user_location"]
        )
        if alert:
            insert_bot_message(
                email=params["email"],
                bot_id=params["bot_id"],
                message=alert,
                activity_name="major_event_alert"
            )

def send_weather_user_alerts():
    for params in get_all_news_agent_params():
        alert = check_and_alert_for_weather_user(
            params["bot_prompt"],
            params["user_name"],
            params["language"],
            params["bot_city"],
            params["user_location"]
        )
        if alert:
            insert_bot_message(
                email=params["email"],
                bot_id=params["bot_id"],
                message=alert,
                activity_name="weather_alert_user"
            )

def send_weather_bot_alerts():
    for params in get_all_news_agent_params():
        alert = check_and_alert_for_weather_bot(
            params["bot_prompt"],
            params["user_name"],
            params["language"],
            params["bot_city"],
            params["user_location"]
        )
        if alert:
            insert_bot_message(
                email=params["email"],
                bot_id=params["bot_id"],
                message=alert,
                activity_name="weather_alert_bot"
            )

def send_news_user_alerts():
    for params in get_all_news_agent_params():
        alert = check_and_alert_for_major_events(
            params["bot_prompt"],
            params["user_name"],
            params["language"],
            params["bot_city"],
            params["user_location"]
        )
        if alert:
            insert_bot_message(
                email=params["email"],
                bot_id=params["bot_id"],
                message=alert,
                activity_name="news_alert_user"
            )

def start_scheduler():
    scheduler = BackgroundScheduler()
    scheduler.add_job(send_weather_user_alerts, 'cron', hour=8)
    scheduler.add_job(send_weather_bot_alerts, 'cron', hour=14)
    scheduler.add_job(send_news_user_alerts, 'cron', hour=19)
    scheduler.start()

# --- Startup event handlers ---
app.add_event_handler("startup", scheduled_weekly_news_summary)
# app.add_event_handler("startup", scheduled_major_event_alert)
app.add_event_handler("startup", start_scheduler)

# --- Manual trigger endpoints ---
@app.post("/run_weekly_summary")
async def run_weekly_summary():
    scheduled_weekly_news_summary()
    return {"status": "Weekly summary triggered"}

@app.post("/run_major_event_alert")
async def run_major_event_alert():
    scheduled_major_event_alert()
    return {"status": "Major event alert triggered"}

@app.post("/run_weather_alerts_user")
async def run_weather_alerts_user():
    count = 0
    for params in get_all_news_agent_params():
        alert = check_and_alert_for_weather_user(
            params["bot_prompt"],
            params["user_name"],
            params["language"],
            params["bot_city"],
            params["user_location"]
        )
        if alert:
            insert_bot_message(
                email=params["email"],
                bot_id=params["bot_id"],
                message=alert,
                activity_name="weather_alert_user_manual"
            )
            count += 1
    return {"status": f"Weather alerts for users triggered - {count} alerts sent"}

@app.post("/run_weather_alerts_bot")
async def run_weather_alerts_bot():
    count = 0
    for params in get_all_news_agent_params():
        alert = check_and_alert_for_weather_bot(
            params["bot_prompt"],
            params["user_name"],
            params["language"],
            params["bot_city"],
            params["user_location"]
        )
        if alert:
            insert_bot_message(
                email=params["email"],
                bot_id=params["bot_id"],
                message=alert,
                activity_name="weather_alert_bot_manual"
            )
            count += 1
    return {"status": f"Weather alerts for bots triggered - {count} alerts sent"}

@app.post("/run_news_alerts_bot")
async def run_news_alerts_bot():
    count = 0
    for params in get_all_news_agent_params():
        alert = check_and_alert_for_major_events_bot(
            params["bot_prompt"],
            params["user_name"],
            params["language"],
            params["bot_city"],
            params["user_location"]
        )
        if alert:
            insert_bot_message(
                email=params["email"],
                bot_id=params["bot_id"],
                message=alert,
                activity_name="news_alert_bot_manual"
            )
            count += 1
    return {"status": f"News alerts for bots triggered - {count} alerts sent"}

# --- Simple Storage Test Endpoint (bypasses AI) ---
@app.post("/test_storage")
async def test_storage(request: QuestionRequest):
    """Simple endpoint that just stores the message and returns a response - bypasses AI"""
    
    print(f"[API] 🚀 Received storage test request from {request.email or 'NO EMAIL'}")
    print(f"[API] 📨 Message: {request.message}")
    print(f"[API] 🤖 Bot ID: {request.bot_id}")
    print(f"[API] 👤 User: {request.user_name}")
    
    # Create a simple bot response
    bot_response = f"Hello {request.user_name}! I received your message: '{request.message}'. This is a storage test response."
    print(f"[API] 💬 Generated bot response: {bot_response}")
    
    # Save to Supabase
    print(f"[API] 💾 Attempting to save to Supabase...")
    success = log_activity_message_to_supabase(
        email=request.email,
        bot_id=request.bot_id,
        user_message=request.message or "",
        bot_response=bot_response,
        platform="weather_news",
        activity_name="storage_test"
    )
    
    if success:
        print(f"[API] ✅ Request completed successfully - message stored in Supabase")
        return {
            "status": "success",
            "message": "Message stored successfully in Supabase",
            "response": bot_response,
            "stored_data": {
                "email": request.email,
                "bot_id": request.bot_id,
                "user_message": request.message,
                "bot_response": bot_response
            }
        }
    else:
        print(f"[API] ❌ Request failed - message NOT stored in Supabase")
        return {
            "status": "error",
            "message": "Failed to store message in Supabase"
        }
