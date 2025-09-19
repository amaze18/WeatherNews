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
import re

# --- Import new message logging system ---
from message_logging_system import (
    RedisManager, 
    RabbitMQManager, 
    SupabaseManager,
    log_and_publish_chat,
    get_redis_manager,
    get_rabbitmq_manager,
    get_supabase_manager,
    cleanup_connections,
    validate_environment
)

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

def filter_asterisks(text: str) -> str:
    """Removes asterisks from a string using regex."""
    if not isinstance(text, str):
        return text
    # The pattern r'\*' matches a literal asterisk
    return re.sub(r'\*', '', text)

def get_user_bot_friendship_time(email, bot_id):
    """
    Get the timestamp when a user first added the bot as a friend (first interaction).
    Returns None if no interaction has been recorded.
    """
    try:
        if supabase is None:
            print(f"[SUPABASE] ❌ ERROR: Supabase client is not initialized in get_user_bot_friendship_time!")
            return None
            
        # Look for the first interaction between user and bot
        response = supabase.table("message_paritition") \
            .select("created_at") \
            .eq("email", email) \
            .eq("bot_id", bot_id) \
            .order("created_at", desc=False) \
            .limit(1) \
            .execute()
        
        if response.data and len(response.data) > 0:
            friendship_time = response.data[0]["created_at"]
            print(f"[PROACTIVE] User {email} added bot {bot_id} as friend at: {friendship_time}")
            return friendship_time
        else:
            print(f"[PROACTIVE] No friendship record found for {email}/{bot_id}")
            return None
            
    except Exception as e:
        logging.error(f"Exception in get_user_bot_friendship_time for {email}/{bot_id}: {e}")
        return None

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
    Check if it's time to send a proactive message based on friendship time and last message time.
    Logic:
    1. If no previous proactive message: Check if 3+ days have passed since friendship
    2. If previous proactive message exists: Check if 3+ days have passed since last message
    """
    try:
        # Get when user first added the bot as friend
        friendship_time = get_user_bot_friendship_time(email, bot_id)
        if friendship_time is None:
            print(f"[PROACTIVE] No friendship record for {email}/{bot_id}, skipping")
            return False
        
        # Get last proactive message time
        last_message_time = get_last_proactive_message_time(email, bot_id)
        
        current_time = datetime.now(timezone.utc)
        
        if last_message_time is None:
            # No previous proactive message - check if 3+ days have passed since friendship
            friendship_datetime = datetime.fromisoformat(friendship_time.replace('Z', '+00:00'))
            time_since_friendship = current_time - friendship_datetime
            days_since_friendship = time_since_friendship.days + (time_since_friendship.seconds / 86400)
            
            if days_since_friendship >= days_interval:
                print(f"[PROACTIVE] {days_since_friendship:.1f} days since friendship for {email}/{bot_id}, will send first proactive message")
                return True
            else:
                print(f"[PROACTIVE] Only {days_since_friendship:.1f} days since friendship for {email}/{bot_id}, waiting for 3 days")
                return False
        else:
            # Previous proactive message exists - check if 3+ days have passed since last message
            last_time = datetime.fromisoformat(last_message_time.replace('Z', '+00:00'))
            time_since_last_message = current_time - last_time
            days_since_last_message = time_since_last_message.days + (time_since_last_message.seconds / 86400)
            
            if days_since_last_message >= days_interval:
                print(f"[PROACTIVE] {days_since_last_message:.1f} days since last proactive message for {email}/{bot_id}, will send new message")
                return True
            else:
                print(f"[PROACTIVE] Only {days_since_last_message:.1f} days since last proactive message for {email}/{bot_id}, skipping")
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

# --- Enhanced Message Logging with Redis, RabbitMQ, and Supabase ---
async def log_and_process_chat(request: QuestionRequest, response_func, endpoint_name="unknown"):
    """
    Enhanced universal wrapper that logs to Redis, RabbitMQ, and Supabase.
    This function wraps any chat processing function to guarantee comprehensive message logging.
    """
    print(f"[ENHANCED_LOGGER] 🚀 Processing {endpoint_name} request from {request.email or 'NO EMAIL'}")
    print(f"[ENHANCED_LOGGER] 📨 Message: {request.message or 'NO MESSAGE'}")
    print(f"[ENHANCED_LOGGER] 🤖 Bot ID: {request.bot_id}")
    print(f"[ENHANCED_LOGGER] 👤 User: {request.user_name}")
    
    # Process the request using the provided function
    try:
        result = await response_func(request)
        bot_response = result.get("response", "") if isinstance(result, dict) else str(result)
    except Exception as e:
        print(f"[ENHANCED_LOGGER] ❌ Error processing request: {e}")
        bot_response = f"Sorry, I encountered an error: {str(e)}"
        result = {"response": bot_response, "error": str(e)}
    
    # ✅ Apply the asterisk filter to the response
    bot_response = filter_asterisks(bot_response)

    # ✅ Update the result dictionary with the cleaned response
    if isinstance(result, dict):
        result["response"] = bot_response
    
    # Enhanced logging with Redis, RabbitMQ, and Supabase
    if request.email and request.email.strip():
        print(f"[ENHANCED_LOGGER] ✅ Email provided: {request.email} - will save to Redis, RabbitMQ, and Supabase")
        try:
            # Create user_id in the required format
            user_id = f"{request.email}:{request.bot_id}"
            
            # Determine activity type based on user message
            activity_name = determine_activity_type(request.message)
            print(f"[ENHANCED_LOGGER] DEBUG: Determined activity_name: {activity_name}")
            
            # Use the new comprehensive logging system
            redis_manager = get_redis_manager()
            logging_result = await log_and_publish_chat(
                redis_manager=redis_manager,
                user_id=user_id,
                user_input=request.message or "",
                bot_reply=bot_response,
                bot_id=request.bot_id,
                email=request.email,
                platform="weather_news",
                requested_time=request.request_time or None
            )
            
            # Log the results
            if logging_result["success"]:
                print(f"[ENHANCED_LOGGER] ✅ Comprehensive logging successful for {request.email}")
                print(f"[ENHANCED_LOGGER] ✅ Redis: {logging_result['redis_stored']}")
                print(f"[ENHANCED_LOGGER] ✅ RabbitMQ: {logging_result['rabbitmq_published']}")
                print(f"[ENHANCED_LOGGER] ✅ Supabase: {logging_result['supabase_stored']}")
            else:
                print(f"[ENHANCED_LOGGER] ❌ Comprehensive logging failed for {request.email}")
                if logging_result["errors"]:
                    for error in logging_result["errors"]:
                        print(f"[ENHANCED_LOGGER] ❌ Error: {error}")
            
            # Fallback to legacy Supabase logging if new system fails
            if not logging_result["supabase_stored"]:
                print(f"[ENHANCED_LOGGER] 🔄 Attempting fallback to legacy Supabase logging...")
                try:
                    storage_success = log_activity_message_to_supabase(
                        email=request.email,
                        bot_id=request.bot_id,
                        user_message=request.message or "",
                        bot_response=bot_response,
                        platform="weather_news",
                        activity_name=activity_name
                    )
                    if storage_success:
                        print(f"[ENHANCED_LOGGER] ✅ Fallback Supabase logging successful")
                    else:
                        print(f"[ENHANCED_LOGGER] ❌ Fallback Supabase logging failed")
                except Exception as e:
                    print(f"[ENHANCED_LOGGER] ❌ Fallback logging failed: {e}")
            
        except Exception as e:
            print(f"[ENHANCED_LOGGER] ❌ Exception in enhanced logging: {e}")
            import traceback
            print(f"[ENHANCED_LOGGER] ❌ Full traceback: {traceback.format_exc()}")
            
            # Fallback to legacy logging
            try:
                activity_name = determine_activity_type(request.message)
                storage_success = log_activity_message_to_supabase(
                    email=request.email,
                    bot_id=request.bot_id,
                    user_message=request.message or "",
                    bot_response=bot_response,
                    platform="weather_news",
                    activity_name=activity_name
                )
                if storage_success:
                    print(f"[ENHANCED_LOGGER] ✅ Fallback to legacy logging successful")
                else:
                    print(f"[ENHANCED_LOGGER] ❌ Fallback to legacy logging failed")
            except Exception as fallback_error:
                print(f"[ENHANCED_LOGGER] ❌ Fallback logging also failed: {fallback_error}")
                logging.error(f"All logging methods failed for {request.email}: {e}")
    else:
        print(f"[ENHANCED_LOGGER] ⚠️  WARNING: No email provided, conversation NOT saved to any database")
        print(f"[ENHANCED_LOGGER] ⚠️  To save conversations, include 'email' field in your request")
    
    # Add enhanced logging status to result
    if isinstance(result, dict):
        result["enhanced_logging"] = {
            "redis_available": bool(request.email and request.email.strip()),
            "rabbitmq_available": bool(request.email and request.email.strip()),
            "supabase_available": bool(request.email and request.email.strip()),
            "endpoint": endpoint_name
        }
    
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

def get_next_message_type(email, bot_id):
    """
    Determine the next message type (weather or news) based on the last proactive message sent.
    Alternates between weather and news updates.
    """
    try:
        if supabase is None:
            return "weather"  # Default to weather
            
        # Get the last proactive message type
        response = supabase.table("message_paritition") \
            .select("activity_name") \
            .eq("email", email) \
            .eq("bot_id", bot_id) \
            .in_("activity_name", ["proactive_weather_update", "proactive_news_update"]) \
            .order("created_at", desc=True) \
            .limit(1) \
            .execute()
        
        if response.data and len(response.data) > 0:
            last_activity = response.data[0]["activity_name"]
            if last_activity == "proactive_weather_update":
                return "news"  # Alternate to news
            else:
                return "weather"  # Alternate to weather
        else:
            # No previous proactive message, start with weather
            return "weather"
            
    except Exception as e:
        print(f"[PROACTIVE] Error determining next message type for {email}/{bot_id}: {e}")
        return "weather"  # Default to weather

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
                # Check if this user should get a weather update (alternating logic)
                next_type = get_next_message_type(params["email"], params["bot_id"])
                if next_type == "weather":
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
                    print(f"[PROACTIVE] Skipping {params['email']}/{params['bot_id']} - next message should be news")
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
                # Check if this user should get a news update (alternating logic)
                next_type = get_next_message_type(params["email"], params["bot_id"])
                if next_type == "news":
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
                    print(f"[PROACTIVE] Skipping {params['email']}/{params['bot_id']} - next message should be weather")
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
    
    # DISABLED: Daily alert jobs (these were causing multiple messages per day)
    # scheduler.add_job(send_weather_user_alerts, 'cron', hour=8)
    # scheduler.add_job(send_weather_bot_alerts, 'cron', hour=14)
    # scheduler.add_job(send_news_user_alerts, 'cron', hour=19)
    
    # Proactive messaging jobs (every 3 days only)
    # Combined proactive updates at 10 AM every 3 days (alternates between weather and news)
    scheduler.add_job(send_proactive_weather_updates, 'cron', hour=10, day='*/3')
    scheduler.add_job(send_proactive_news_updates, 'cron', hour=10, day='*/3')
    # General updates at 6 PM every 3 days (fallback)
    scheduler.add_job(send_proactive_general_updates, 'cron', hour=18, day='*/3')
    
    scheduler.start()
    print("[SCHEDULER] Proactive messaging scheduler started with 3-day intervals only (daily alerts disabled)")

# --- Startup event handlers ---
app.add_event_handler("startup", scheduled_weekly_news_summary)
# app.add_event_handler("startup", scheduled_major_event_alert)
app.add_event_handler("startup", start_scheduler)

# --- Enhanced Logging System Startup ---
@app.on_event("startup")
async def startup_logging_system():
    """Initialize the enhanced logging system on startup"""
    try:
        print("[STARTUP] 🚀 Initializing enhanced logging system...")
        
        # Test environment configuration
        try:
            validate_environment()
            print("[STARTUP] ✅ Environment variables validated")
        except ValueError as e:
            print(f"[STARTUP] ⚠️  Environment validation failed: {e}")
            print("[STARTUP] ⚠️  Some logging features may not work correctly")
        
        # Test individual connections
        connection_results = {"redis": False, "rabbitmq": False, "supabase": False, "errors": []}
        
        # Test Redis
        try:
            redis_manager = get_redis_manager()
            redis_manager._ensure_connection()
            connection_results["redis"] = True
            print("[STARTUP] ✅ Redis: Connected")
        except Exception as e:
            connection_results["errors"].append(f"Redis connection failed: {e}")
            print(f"[STARTUP] ❌ Redis: Failed - {e}")
        
        # Test RabbitMQ
        try:
            rabbitmq_manager = get_rabbitmq_manager()
            rabbitmq_manager._ensure_connection()
            connection_results["rabbitmq"] = True
            print("[STARTUP] ✅ RabbitMQ: Connected")
        except Exception as e:
            connection_results["errors"].append(f"RabbitMQ connection failed: {e}")
            print(f"[STARTUP] ❌ RabbitMQ: Failed - {e}")
        
        # Test Supabase
        try:
            supabase_manager = get_supabase_manager()
            connection_results["supabase"] = True
            print("[STARTUP] ✅ Supabase: Connected")
        except Exception as e:
            connection_results["errors"].append(f"Supabase connection failed: {e}")
            print(f"[STARTUP] ❌ Supabase: Failed - {e}")
        
        if all([connection_results["redis"], connection_results["rabbitmq"], connection_results["supabase"]]):
            print("[STARTUP] ✅ All logging systems (Redis, RabbitMQ, Supabase) are operational")
        else:
            print("[STARTUP] ⚠️  Some logging systems are not operational")
            if connection_results["errors"]:
                print("[STARTUP] Errors:")
                for error in connection_results["errors"]:
                    print(f"[STARTUP] ❌ {error}")
        
        print("[STARTUP] 🎯 Enhanced logging system initialization complete")
        
    except Exception as e:
        print(f"[STARTUP] ❌ Failed to initialize enhanced logging system: {e}")
        import traceback
        print(f"[STARTUP] ❌ Full traceback: {traceback.format_exc()}")

@app.on_event("shutdown")
async def shutdown_logging_system():
    """Clean up logging system connections on shutdown"""
    try:
        print("[SHUTDOWN] 🧹 Cleaning up logging system connections...")
        cleanup_connections()
        print("[SHUTDOWN] ✅ Logging system cleanup complete")
    except Exception as e:
        print(f"[SHUTDOWN] ❌ Error during cleanup: {e}")

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
            "message_logic": "First message sent 3 days after friendship, then alternating weather/news every 3 days",
            "scheduled_times": {
                "proactive_updates": "10:00 AM every 3 days (alternates weather/news)",
                "general_updates": "6:00 PM every 3 days (fallback)",
                "breaking_alerts": "8:00 AM (weather), 2:00 PM (weather), 7:00 PM (news) daily"
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

# --- Enhanced Logging System Test Endpoints ---
@app.post("/test_enhanced_logging")
async def test_enhanced_logging(request: QuestionRequest):
    """Test endpoint for the new Redis + RabbitMQ + Supabase logging system"""
    print(f"[ENHANCED_LOGGING_TEST] 🧪 Testing enhanced logging for {request.email or 'NO EMAIL'}")
    
    # Create a test response
    test_response = f"Enhanced logging test successful! Your message '{request.message}' will be logged to Redis, RabbitMQ, and Supabase."
    
    # Use the enhanced logging wrapper
    async def test_logic(req):
        return {
            "response": test_response,
            "user_name": req.user_name,
            "language": req.language,
            "test_type": "enhanced_logging_verification"
        }
    
    result = await log_and_process_chat(request, test_logic, "test_enhanced_logging")
    
    # Add enhanced test information
    result["enhanced_logging_test"] = {
        "email_provided": bool(request.email and request.email.strip()),
        "message_length": len(request.message or ""),
        "bot_id": request.bot_id,
        "timestamp": datetime.utcnow().isoformat(),
        "systems_tested": ["Redis", "RabbitMQ", "Supabase"]
    }
    
    return result

@app.get("/logging_system_status")
async def get_logging_system_status():
    """Get the status of all logging systems (Redis, RabbitMQ, Supabase)"""
    try:
        # Test environment configuration
        try:
            validate_environment()
            env_status = "valid"
        except ValueError as e:
            env_status = f"invalid: {str(e)}"
        
        # Test individual connections
        connection_results = {"redis": False, "rabbitmq": False, "supabase": False, "errors": []}
        
        # Test Redis
        try:
            redis_manager = get_redis_manager()
            redis_manager._ensure_connection()
            connection_results["redis"] = True
        except Exception as e:
            connection_results["errors"].append(f"Redis connection failed: {e}")
        
        # Test RabbitMQ
        try:
            rabbitmq_manager = get_rabbitmq_manager()
            rabbitmq_manager._ensure_connection()
            connection_results["rabbitmq"] = True
        except Exception as e:
            connection_results["errors"].append(f"RabbitMQ connection failed: {e}")
        
        # Test Supabase
        try:
            supabase_manager = get_supabase_manager()
            connection_results["supabase"] = True
        except Exception as e:
            connection_results["errors"].append(f"Supabase connection failed: {e}")
        
        return {
            "status": "success",
            "environment_config": {
                "redis_url_configured": bool(os.getenv("REDIS_URL")),
                "rabbitmq_url_configured": bool(os.getenv("RABBITMQ_URL")),
                "supabase_url_configured": bool(os.getenv("SUPABASE_URL")),
                "supabase_key_configured": bool(os.getenv("SUPABASE_KEY")),
                "validation_status": env_status
            },
            "connection_status": connection_results,
            "system_health": {
                "redis": connection_results["redis"],
                "rabbitmq": connection_results["rabbitmq"],
                "supabase": connection_results["supabase"]
            },
            "recommendations": [
                "Ensure all environment variables are set correctly in .env file",
                "Check that Redis server is running and accessible",
                "Check that RabbitMQ server is running and accessible", 
                "Verify Supabase credentials and network connectivity"
            ] if not all([connection_results["redis"], connection_results["rabbitmq"], connection_results["supabase"]]) else ["All systems operational"]
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to check logging system status: {str(e)}",
            "recommendations": [
                "Check environment variable configuration in .env file",
                "Verify all service connections",
                "Review error logs for specific issues"
            ]
        }

@app.get("/redis_status")
async def get_redis_status():
    """Get Redis connection and data status"""
    try:
        redis_manager = get_redis_manager()
        
        # Test basic operations
        test_key = "test:connection"
        test_data = {"test": "data", "timestamp": datetime.utcnow().isoformat()}
        
        # Store test data
        store_success = redis_manager.store_user_data(test_key, test_data, ttl=60)
        
        # Retrieve test data
        retrieved_data = redis_manager.get_user_data(test_key)
        
        # Clean up test data
        redis_manager.clear_user_data(test_key)
        
        return {
            "status": "success",
            "connection": "active",
            "test_operations": {
                "store": store_success,
                "retrieve": retrieved_data is not None,
                "data_match": retrieved_data == test_data if retrieved_data else False
            },
            "redis_url": redis_manager.redis_url[:20] + "..." if redis_manager.redis_url else "Not configured"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Redis connection failed: {str(e)}",
            "recommendations": [
                "Check Redis server is running",
                "Verify REDIS_URL environment variable",
                "Check network connectivity to Redis server"
            ]
        }

@app.get("/rabbitmq_status")
async def get_rabbitmq_status():
    """Get RabbitMQ connection status"""
    try:
        rabbitmq_manager = get_rabbitmq_manager()
        
        # Test connection
        rabbitmq_manager._ensure_connection()
        
        return {
            "status": "success",
            "connection": "active",
            "rabbitmq_url": rabbitmq_manager.rabbitmq_url[:20] + "..." if rabbitmq_manager.rabbitmq_url else "Not configured",
            "queues_configured": ["message_storage", "message_processing"]
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"RabbitMQ connection failed: {str(e)}",
            "recommendations": [
                "Check RabbitMQ server is running",
                "Verify RABBITMQ_URL environment variable",
                "Check network connectivity to RabbitMQ server"
            ]
        }

@app.get("/supabase_status")
async def get_supabase_status():
    """Get Supabase connection status"""
    try:
        supabase_manager = get_supabase_manager()
        
        # Test connection by trying to read from the table
        response = supabase_manager.supabase_client.table("message_paritition").select("id").limit(1).execute()
        
        return {
            "status": "success",
            "connection": "active",
            "supabase_url": supabase_manager.supabase_url[:20] + "..." if supabase_manager.supabase_url else "Not configured",
            "data_count": len(response.data) if response.data else 0,
            "table_accessible": True
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Supabase connection failed: {str(e)}",
            "recommendations": [
                "Check Supabase credentials",
                "Verify SUPABASE_URL and SUPABASE_KEY environment variables",
                "Check network connectivity to Supabase",
                "Verify table permissions"
            ]
        }

@app.post("/cleanup_connections")
async def cleanup_logging_connections():
    """Clean up all logging system connections"""
    try:
        cleanup_connections()
        return {
            "status": "success",
            "message": "All logging system connections cleaned up successfully"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to cleanup connections: {str(e)}"
        }
