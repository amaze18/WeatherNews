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
print(f"[SUPABASE] 🔧 Initializing Supabase client...")
print(f"[SUPABASE] 🔧 SUPABASE_URL: {SUPABASE_URL[:20] + '...' if SUPABASE_URL else 'None'}")
print(f"[SUPABASE] 🔧 SUPABASE_KEY: {'Present' if SUPABASE_KEY else 'Missing'}")

if not SUPABASE_URL or not SUPABASE_KEY:
    print(f"[SUPABASE] ❌ ERROR: Missing Supabase credentials!")
    print(f"[SUPABASE] ❌ SUPABASE_URL: {SUPABASE_URL}")
    print(f"[SUPABASE] ❌ SUPABASE_KEY: {'Present' if SUPABASE_KEY else 'Missing'}")
    supabase = None
else:
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        print(f"[SUPABASE] ✅ Supabase client initialized successfully")
    except Exception as e:
        print(f"[SUPABASE] ❌ ERROR: Failed to initialize Supabase client: {e}")
        supabase = None

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
        if supabase is None:
            print(f"[SUPABASE] ❌ ERROR: Supabase client is not initialized in get_today_user_bot_pairs!")
            return []
            
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
        if supabase is None:
            print(f"[SUPABASE] ❌ ERROR: Supabase client is not initialized in get_all_news_agent_params!")
            continue
            
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

def get_all_active_users():
    """
    Get all users who have ever interacted with any bot, not just today's users.
    This is used for proactive messaging to reach all users, not just those who chatted today.
    """
    try:
        if supabase is None:
            print(f"[SUPABASE] ❌ ERROR: Supabase client is not initialized in get_all_active_users!")
            return []
            
        # Get all unique user-bot pairs from message history
        response = supabase.table("message_paritition") \
            .select("email, bot_id") \
            .execute()
        
        # Create unique pairs
        pairs = {(item["email"], item["bot_id"]) for item in response.data if item.get("email") and item.get("bot_id")}
        params_list = []
        
        for email, bot_id in pairs:
            try:
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
            except Exception as e:
                print(f"[SUPABASE] ⚠️  WARNING: Failed to get details for {email}/{bot_id}: {e}")
                continue
                
        print(f"[PROACTIVE] Found {len(params_list)} active user-bot pairs for proactive messaging")
        return params_list
    except Exception as e:
        logging.error(f"Exception in get_all_active_users: {e}")
        return []

def get_last_proactive_message_time(email, bot_id):
    """
    Get the timestamp of the last proactive message sent to a user-bot pair.
    Returns None if no proactive message has been sent before.
    """
    try:
        if supabase is None:
            print(f"[SUPABASE] ❌ ERROR: Supabase client is not initialized in get_last_proactive_message_time!")
            return None
            
        # Look for the most recent proactive message
        response = supabase.table("message_paritition") \
            .select("created_at") \
            .eq("email", email) \
            .eq("bot_id", bot_id) \
            .in_("activity_name", ["proactive_weather_update", "proactive_news_update", "proactive_general_update"]) \
            .order("created_at", desc=True) \
            .limit(1) \
            .execute()
        
        if response.data and len(response.data) > 0:
            last_message_time = response.data[0]["created_at"]
            print(f"[PROACTIVE] Last proactive message for {email}/{bot_id}: {last_message_time}")
            return last_message_time
        else:
            print(f"[PROACTIVE] No previous proactive messages found for {email}/{bot_id}")
            return None
            
    except Exception as e:
        logging.error(f"Exception in get_last_proactive_message_time for {email}/{bot_id}: {e}")
        return None

def should_send_proactive_message(email, bot_id, days_interval=3):
    """
    Check if enough time has passed since the last proactive message to send a new one.
    Returns True if it's time to send a proactive message.
    """
    try:
        last_message_time = get_last_proactive_message_time(email, bot_id)
        
        if last_message_time is None:
            # No previous proactive message, so it's time to send one
            print(f"[PROACTIVE] No previous proactive message for {email}/{bot_id}, will send first message")
            return True
        
        # Parse the timestamp and check if enough days have passed
        from datetime import datetime
        last_time = datetime.fromisoformat(last_message_time.replace('Z', '+00:00'))
        current_time = datetime.now(timezone.utc)
        time_diff = current_time - last_time
        
        days_passed = time_diff.days + (time_diff.seconds / 86400)  # Include fractional days
        
        if days_passed >= days_interval:
            print(f"[PROACTIVE] {days_passed:.1f} days passed since last proactive message for {email}/{bot_id}, will send new message")
            return True
        else:
            print(f"[PROACTIVE] Only {days_passed:.1f} days passed since last proactive message for {email}/{bot_id}, skipping")
            return False
            
    except Exception as e:
        logging.error(f"Exception in should_send_proactive_message for {email}/{bot_id}: {e}")
        return False

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
    print(f"[SUPABASE] 🚀 log_activity_message_to_supabase called with:")
    print(f"[SUPABASE] 🚀 email: {email}")
    print(f"[SUPABASE] 🚀 bot_id: {bot_id}")
    print(f"[SUPABASE] 🚀 platform: {platform}")
    print(f"[SUPABASE] 🚀 activity_name: {activity_name}")
    
    # Validate required parameters
    if not email or not email.strip():
        print(f"[SUPABASE] ❌ ERROR: Email is required for message logging")
        logging.error("Email is required for message logging")
        return False
    
    if not bot_id or not bot_id.strip():
        print(f"[SUPABASE] ❌ ERROR: Bot ID is required for message logging")
        logging.error("Bot ID is required for message logging")
        return False
    
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
        user_message = str(user_message or "").encode('utf-8', errors='ignore').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        bot_response = str(bot_response)
        user_message = str(user_message or "")
    
    # Truncate messages if they're too long (to prevent database issues)
    max_message_length = 10000  # Adjust based on your database column limits
    if len(user_message) > max_message_length:
        user_message = user_message[:max_message_length] + "... [truncated]"
    if len(bot_response) > max_message_length:
        bot_response = bot_response[:max_message_length] + "... [truncated]"
    
    data = {
        "email": email.strip(),
        "bot_id": bot_id.strip(),
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
        if supabase is None:
            print(f"[SUPABASE] ❌ ERROR: Supabase client is not initialized!")
            logging.error("Supabase client is not initialized")
            return False
            
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
        print(f"[SUPABASE] ❌ Error details: {str(e)}")
        logging.error(f"Failed to insert message for {email}: {e}")
        
        # Try to provide more specific error information
        if "duplicate key" in str(e).lower():
            print(f"[SUPABASE] ⚠️  WARNING: Possible duplicate message detected")
        elif "connection" in str(e).lower():
            print(f"[SUPABASE] ⚠️  WARNING: Connection issue with Supabase")
        elif "permission" in str(e).lower() or "unauthorized" in str(e).lower():
            print(f"[SUPABASE] ⚠️  WARNING: Permission issue with Supabase")
        
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

# --- Universal Message Logging Wrapper ---
async def log_and_process_chat(request: QuestionRequest, response_func, endpoint_name="unknown"):
    """
    Universal wrapper to ensure all chat interactions are logged to Supabase.
    This function wraps any chat processing function to guarantee message logging.
    """
    print(f"[CHAT_LOGGER] 🚀 Processing {endpoint_name} request from {request.email or 'NO EMAIL'}")
    print(f"[CHAT_LOGGER] 📨 Message: {request.message or 'NO MESSAGE'}")
    print(f"[CHAT_LOGGER] 🤖 Bot ID: {request.bot_id}")
    print(f"[CHAT_LOGGER] 👤 User: {request.user_name}")
    
    # Process the request using the provided function
    try:
        result = await response_func(request)
        bot_response = result.get("response", "") if isinstance(result, dict) else str(result)
    except Exception as e:
        print(f"[CHAT_LOGGER] ❌ Error processing request: {e}")
        bot_response = f"Sorry, I encountered an error: {str(e)}"
        result = {"response": bot_response, "error": str(e)}
    
    # Always attempt to log the conversation if email is provided
    if request.email and request.email.strip():
        print(f"[CHAT_LOGGER] ✅ Email provided: {request.email} - will save to Supabase")
        try:
            # Determine activity type based on user message
            activity_name = determine_activity_type(request.message)
            print(f"[CHAT_LOGGER] DEBUG: Determined activity_name: {activity_name}")
            
            storage_success = log_activity_message_to_supabase(
                email=request.email,
                bot_id=request.bot_id,
                user_message=request.message or "",
                bot_response=bot_response,
                platform="weather_news",
                activity_name=activity_name
            )
            if storage_success:
                print(f"[CHAT_LOGGER] ✅ Conversation successfully saved to Supabase for {request.email}")
                print(f"[CHAT_LOGGER] ✅ Activity type: {activity_name}")
                print(f"[CHAT_LOGGER] ✅ Endpoint: {endpoint_name}")
            else:
                print(f"[CHAT_LOGGER] ❌ Failed to save conversation to Supabase for {request.email}")
        except Exception as e:
            print(f"[CHAT_LOGGER] ❌ Exception while saving conversation: {e}")
            import traceback
            print(f"[CHAT_LOGGER] ❌ Full traceback: {traceback.format_exc()}")
            logging.error(f"Failed to save conversation for {request.email}: {e}")
    else:
        print(f"[CHAT_LOGGER] ⚠️  WARNING: No email provided, conversation NOT saved to database")
        print(f"[CHAT_LOGGER] ⚠️  To save conversations, include 'email' field in your request")
    
    # Add logging status to result
    if isinstance(result, dict):
        result["logged_to_supabase"] = bool(request.email and request.email.strip())
        result["endpoint"] = endpoint_name
    
    return result

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
    
    return {
        "response": response,
        "user_name": request.user_name,
        "language": request.language
    }

# --- Endpoints ---
@app.post("/news_weather_agent")
async def news_weather_agent(request: QuestionRequest):
    return await log_and_process_chat(request, handle_news_weather_agent, "news_weather_agent")

@app.post("/weather/news_weather_agent")
async def news_weather_agent_alias(request: QuestionRequest):
    return await log_and_process_chat(request, handle_news_weather_agent, "news_weather_agent_alias")

# --- Gaming Agents Compatible Endpoint ---
@app.post("/weather_news_agent")
async def weather_news_agent_compatible(request: QuestionRequest):
    """Gaming agents compatible endpoint that stores to Supabase"""
    return await log_and_process_chat(request, simple_chat, "weather_news_agent_compatible")

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

# --- Proactive Messaging Functions ---

def generate_proactive_weather_message(persona_prompt, user_name, language, bot_location, user_location="India"):
    """
    Generate a proactive weather update message for users.
    This sends weather information even when there's no breaking weather news.
    """
    try:
        from news_weather_agent import open_weather_map_tool, get_weather_response
        
        # Get current weather for user location
        weather_summary = open_weather_map_tool._run(city_name=user_location)
        
        # Generate a friendly weather update message
        response = get_weather_response(
            weather_summary, persona_prompt, user_name, language, bot_location, user_location, context="user"
        )
        
        # Add a friendly greeting to make it more proactive
        proactive_message = f"Hey {user_name}! Just wanted to share a quick weather update with you. {response}"
        
        return proactive_message
        
    except Exception as e:
        print(f"Error generating proactive weather message: {e}")
        return f"Hey {user_name}! Hope you're having a great day! Just wanted to check in and see how you're doing. How's the weather treating you?"

def generate_proactive_news_message(persona_prompt, user_name, language, bot_location, user_location="India"):
    """
    Generate a proactive news update message for users.
    This sends general news information to keep users informed.
    """
    try:
        from news_weather_agent import crew, get_news_response
        
        # Get latest news for user location
        api_key = os.environ.get("GEMINI_API_KEY")
        from llama_index.llms.google_genai import GoogleGenAI
        llm = GoogleGenAI(model="gemini-1.5-flash", api_key=api_key)
        location = llm.complete(f"Only give the answer for the question\nIf user_location is country, then answer the same name, if it is city, then answer in the country which that city belongs\nWhat is the location of {user_location}?\n")
        
        topic = f"Latest National news in {location} today"
        result = crew.kickoff(inputs={'topic': topic})
        news_summary = str(result)
        
        # Generate a friendly news update message
        response = get_news_response(
            news_summary, persona_prompt, user_name, language, bot_location, user_location, context="user"
        )
        
        # Add a friendly greeting to make it more proactive
        proactive_message = f"Hey {user_name}! Thought you might be interested in what's happening around. {response}"
        
        return proactive_message
        
    except Exception as e:
        print(f"Error generating proactive news message: {e}")
        return f"Hey {user_name}! Hope you're doing well! Just wanted to reach out and see how things are going with you. Anything interesting happening in your area?"

def send_proactive_weather_updates():
    """
    Send proactive weather updates to users who haven't received a message in 3 days.
    """
    print("[PROACTIVE] Starting proactive weather updates...")
    sent_count = 0
    
    for params in get_all_active_users():
        try:
            # Check if it's time to send a proactive message
            if should_send_proactive_message(params["email"], params["bot_id"], days_interval=3):
                # Generate proactive weather message
                message = generate_proactive_weather_message(
                    params["bot_prompt"],
                    params["user_name"],
                    params["language"],
                    params["bot_city"],
                    params["user_location"]
                )
                
                # Send the message
                success = insert_bot_message(
                    email=params["email"],
                    bot_id=params["bot_id"],
                    message=message,
                    activity_name="proactive_weather_update"
                )
                
                if success:
                    sent_count += 1
                    print(f"[PROACTIVE] Sent weather update to {params['email']}/{params['bot_id']}")
                else:
                    print(f"[PROACTIVE] Failed to send weather update to {params['email']}/{params['bot_id']}")
            else:
                print(f"[PROACTIVE] Skipping {params['email']}/{params['bot_id']} - too soon for next message")
                
        except Exception as e:
            print(f"[PROACTIVE] Error processing {params['email']}/{params['bot_id']}: {e}")
            continue
    
    print(f"[PROACTIVE] Proactive weather updates completed. Sent {sent_count} messages.")
    return sent_count

def send_proactive_news_updates():
    """
    Send proactive news updates to users who haven't received a message in 3 days.
    """
    print("[PROACTIVE] Starting proactive news updates...")
    sent_count = 0
    
    for params in get_all_active_users():
        try:
            # Check if it's time to send a proactive message
            if should_send_proactive_message(params["email"], params["bot_id"], days_interval=3):
                # Generate proactive news message
                message = generate_proactive_news_message(
                    params["bot_prompt"],
                    params["user_name"],
                    params["language"],
                    params["bot_city"],
                    params["user_location"]
                )
                
                # Send the message
                success = insert_bot_message(
                    email=params["email"],
                    bot_id=params["bot_id"],
                    message=message,
                    activity_name="proactive_news_update"
                )
                
                if success:
                    sent_count += 1
                    print(f"[PROACTIVE] Sent news update to {params['email']}/{params['bot_id']}")
                else:
                    print(f"[PROACTIVE] Failed to send news update to {params['email']}/{params['bot_id']}")
            else:
                print(f"[PROACTIVE] Skipping {params['email']}/{params['bot_id']} - too soon for next message")
                
        except Exception as e:
            print(f"[PROACTIVE] Error processing {params['email']}/{params['bot_id']}: {e}")
            continue
    
    print(f"[PROACTIVE] Proactive news updates completed. Sent {sent_count} messages.")
    return sent_count

def send_proactive_general_updates():
    """
    Send general proactive updates to users who haven't received a message in 3 days.
    This is a fallback for when weather/news updates aren't available.
    """
    print("[PROACTIVE] Starting proactive general updates...")
    sent_count = 0
    
    for params in get_all_active_users():
        try:
            # Check if it's time to send a proactive message
            if should_send_proactive_message(params["email"], params["bot_id"], days_interval=3):
                # Generate a general friendly message
                message = f"Hey {params['user_name']}! Hope you're having a wonderful day! Just wanted to reach out and see how you're doing. Feel free to chat with me anytime if you want to know about the weather or latest news!"
                
                # Send the message
                success = insert_bot_message(
                    email=params["email"],
                    bot_id=params["bot_id"],
                    message=message,
                    activity_name="proactive_general_update"
                )
                
                if success:
                    sent_count += 1
                    print(f"[PROACTIVE] Sent general update to {params['email']}/{params['bot_id']}")
                else:
                    print(f"[PROACTIVE] Failed to send general update to {params['email']}/{params['bot_id']}")
            else:
                print(f"[PROACTIVE] Skipping {params['email']}/{params['bot_id']} - too soon for next message")
                
        except Exception as e:
            print(f"[PROACTIVE] Error processing {params['email']}/{params['bot_id']}: {e}")
            continue
    
    print(f"[PROACTIVE] Proactive general updates completed. Sent {sent_count} messages.")
    return sent_count

def start_scheduler():
    scheduler = BackgroundScheduler()
    
    # New proactive messaging jobs (every 3 days)
    # Weather updates at 10 AM every 3 days
    scheduler.add_job(send_proactive_weather_updates, 'cron', hour=10, day='*/3')
    # News updates at 4 PM every 3 days  
    scheduler.add_job(send_proactive_news_updates, 'cron', hour=16, day='*/3')
    # General updates at 6 PM every 3 days (fallback)
    scheduler.add_job(send_proactive_general_updates, 'cron', hour=18, day='*/3')
    
    scheduler.start()
    print("[SCHEDULER] Proactive messaging scheduler started with 3-day intervals")

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

# --- Proactive Messaging Manual Trigger Endpoints ---

@app.post("/run_proactive_weather_updates")
async def run_proactive_weather_updates():
    """Manually trigger proactive weather updates for all users"""
    count = send_proactive_weather_updates()
    return {"status": f"Proactive weather updates triggered - {count} messages sent"}

@app.post("/run_proactive_news_updates")
async def run_proactive_news_updates():
    """Manually trigger proactive news updates for all users"""
    count = send_proactive_news_updates()
    return {"status": f"Proactive news updates triggered - {count} messages sent"}

@app.post("/run_proactive_general_updates")
async def run_proactive_general_updates():
    """Manually trigger proactive general updates for all users"""
    count = send_proactive_general_updates()
    return {"status": f"Proactive general updates triggered - {count} messages sent"}

@app.post("/run_all_proactive_updates")
async def run_all_proactive_updates():
    """Manually trigger all proactive updates for all users"""
    weather_count = send_proactive_weather_updates()
    news_count = send_proactive_news_updates()
    general_count = send_proactive_general_updates()
    total_count = weather_count + news_count + general_count
    return {
        "status": f"All proactive updates triggered",
        "weather_updates": weather_count,
        "news_updates": news_count,
        "general_updates": general_count,
        "total_messages": total_count
    }

@app.get("/proactive_status")
async def get_proactive_status():
    """Get status of proactive messaging system"""
    try:
        active_users = get_all_active_users()
        user_count = len(active_users)
        
        # Count users who would receive proactive messages
        eligible_count = 0
        for params in active_users:
            if should_send_proactive_message(params["email"], params["bot_id"], days_interval=3):
                eligible_count += 1
        
        return {
            "status": "success",
            "total_active_users": user_count,
            "eligible_for_proactive_messages": eligible_count,
            "proactive_interval_days": 3,
            "scheduled_times": {
                "weather_updates": "10:00 AM every 3 days",
                "news_updates": "4:00 PM every 3 days", 
                "general_updates": "6:00 PM every 3 days"
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to get proactive status: {str(e)}"
        }

# --- Supabase Connection Test Endpoint ---
@app.get("/test_supabase_connection")
async def test_supabase_connection():
    """Test Supabase connection and return status"""
    print(f"[TEST] Testing Supabase connection...")
    
    if supabase is None:
        return {
            "status": "error",
            "message": "Supabase client is not initialized",
            "supabase_url": SUPABASE_URL,
            "supabase_key": "Present" if SUPABASE_KEY else "Missing"
        }
    
    try:
        # Test connection by trying to read from the table
        response = supabase.table("message_paritition").select("id").limit(1).execute()
        return {
            "status": "success",
            "message": "Supabase connection successful",
            "data_count": len(response.data) if response.data else 0
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Supabase connection failed: {str(e)}"
        }

# --- Simplified Chat Endpoint (works without AI dependencies) ---
async def simple_chat(request: QuestionRequest):
    """Simplified chat endpoint that stores to Supabase without AI dependencies"""
    
    print(f"[SIMPLE_CHAT] Processing request from {request.email or 'NO EMAIL'}")
    print(f"[SIMPLE_CHAT] Message: {request.message}")
    print(f"[SIMPLE_CHAT] Bot ID: {request.bot_id}")
    
    # Create a simple response
    bot_response = f"Hello {request.user_name}! I received your message: '{request.message}'. This is a simplified response from the weather news agent."
    
    return {
        "response": bot_response,
        "user_name": request.user_name,
        "language": request.language
    }

@app.post("/simple_chat")
async def simple_chat_endpoint(request: QuestionRequest):
    """Public endpoint for simple chat with automatic logging"""
    return await log_and_process_chat(request, simple_chat, "simple_chat")

# --- Simple Storage Test Endpoint (bypasses AI) ---
async def test_storage_logic(request: QuestionRequest):
    """Logic for storage test endpoint"""
    print(f"[API] 🚀 Received storage test request from {request.email or 'NO EMAIL'}")
    print(f"[API] 📨 Message: {request.message}")
    print(f"[API] 🤖 Bot ID: {request.bot_id}")
    print(f"[API] 👤 User: {request.user_name}")
    
    # Create a simple bot response
    bot_response = f"Hello {request.user_name}! I received your message: '{request.message}'. This is a storage test response."
    print(f"[API] 💬 Generated bot response: {bot_response}")
    
    return {
        "status": "success",
        "message": "Message processed successfully",
        "response": bot_response,
        "test_data": {
            "email": request.email,
            "bot_id": request.bot_id,
            "user_message": request.message,
            "bot_response": bot_response
        }
    }

@app.post("/test_storage")
async def test_storage(request: QuestionRequest):
    """Simple endpoint that just stores the message and returns a response - bypasses AI"""
    return await log_and_process_chat(request, test_storage_logic, "test_storage")

# --- Message Logging Test Endpoint ---
@app.post("/test_message_logging")
async def test_message_logging(request: QuestionRequest):
    """Test endpoint specifically for verifying message logging functionality"""
    print(f"[MESSAGE_LOGGING_TEST] 🧪 Testing message logging for {request.email or 'NO EMAIL'}")
    
    # Create a test response
    test_response = f"Message logging test successful! Your message '{request.message}' was received and will be logged to Supabase."
    
    # Use the logging wrapper
    async def test_logic(req):
        return {
            "response": test_response,
            "user_name": req.user_name,
            "language": req.language,
            "test_type": "message_logging_verification"
        }
    
    result = await log_and_process_chat(request, test_logic, "test_message_logging")
    
    # Add additional test information
    result["logging_test"] = {
        "email_provided": bool(request.email and request.email.strip()),
        "message_length": len(request.message or ""),
        "bot_id": request.bot_id,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    return result
