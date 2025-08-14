from fastapi import FastAPI
from pydantic import BaseModel
from typing import Union
from news_weather_agent import (
    persona_response, 
    generate_weekly_news_summary, 
    check_and_alert_for_major_events,
    check_and_alert_for_major_events_user,
    check_and_alert_for_major_events_bot,
    check_and_alert_for_weather_user,
    check_and_alert_for_weather_bot
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
import asyncio
import random
import hashlib

# Helper function to create staggered delays
def get_staggered_delay(user_email, bot_id, base_delay_hours=0, max_additional_hours=24):
    """
    Creates a deterministic but pseudo-random delay based on user_email and bot_id.
    This ensures the same user-bot pair always gets the same delay, but different
    pairs get different delays to spread out the proactive messages.
    """
    # Create a hash based on email and bot_id for deterministic randomness
    hash_input = f"{user_email}_{bot_id}".encode()
    hash_value = hashlib.md5(hash_input).hexdigest()
    
    # Convert first 8 characters to int and get a delay between 0 and max_additional_hours
    delay_modifier = int(hash_value[:8], 16) % (max_additional_hours * 3600)  # Convert to seconds
    total_delay = base_delay_hours * 3600 + delay_modifier
    
    return total_delay

# Helper function to check if enough time has passed since last proactive message
def should_send_proactive_message(email, bot_id, message_type, min_interval_hours=24):
    """
    Check if enough time has passed since the last proactive message of this type
    was sent to this user-bot pair.
    """
    try:
        # Calculate the cutoff time
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=min_interval_hours)
        
        # Check if a message of this type was sent recently
        response = supabase.table("message_paritition") \
            .select("created_at") \
            .eq("email", email) \
            .eq("bot_id", bot_id) \
            .ilike("bot_response", f"%{message_type}%") \
            .gte("created_at", cutoff_time.isoformat()) \
            .limit(1) \
            .execute()
        
        # If no recent message found, it's okay to send
        return len(response.data) == 0
    except Exception as e:
        print(f"[DEBUG] Error checking last proactive message: {e}")
        # If there's an error, err on the side of not sending to avoid spam
        return False

@repeat_every(seconds=60*60*24*7)  # Every 7 days
def scheduled_weekly_news_summary():
    print("[DEBUG] Entered scheduled_weekly_news_summary")
    print("=== Running Weekly News Summary for all users/bots ===")
    print("⏰ Running Weekly News Summary task every 7 days")
    for params in get_all_news_agent_params():
        print(f"[DEBUG] Params for weekly summary: {params}")
        # Pass all required params to your agent function
        summary = generate_weekly_news_summary(
            params["bot_prompt"],  # persona_prompt
            params["user_name"],
            params["language"],
            params["bot_city"],    # bot_location
            params["user_location"] 
        )
        print(f"[DEBUG] Generated summary: {summary}")
        insert_bot_message(
            email=params["email"],
            bot_id=params["bot_id"],
            message=summary
        )

@repeat_every(seconds=60*60*6)  # Every 6 hours
def scheduled_major_event_alert():
    print("[DEBUG] Entered scheduled_major_event_alert")
    print("=== Checking for Major Events for all users/bots ===")
    print("⏰ Running Major Event Alert task every 6 hours")
    for params in get_all_news_agent_params():
        print(f"[DEBUG] Params for major event alert: {params}")
        alert = check_and_alert_for_major_events(
            params["bot_prompt"],
            params["user_name"],
            params["language"],
            params["bot_city"],
            params["user_location"]
        )
        print(f"[DEBUG] Alert generated: {alert}")
        if alert:
            insert_bot_message(
                email=params["email"],
                bot_id=params["bot_id"],
                message=alert
            )

# New staggered proactive tasks
async def send_staggered_proactive_messages():
    """
    Send proactive messages with staggered delays to avoid all messages being sent at once.
    """
    print("[DEBUG] Starting staggered proactive message sending")
    params_list = get_all_news_agent_params()
    
    # Create tasks for each type of proactive message
    tasks = []
    
    for params in params_list:
        email = params["email"]
        bot_id = params["bot_id"]
        
        # Weather alert for user location - every 12 hours
        if should_send_proactive_message(email, bot_id, "[WEATHER_USER]", min_interval_hours=12):
            delay = get_staggered_delay(email, bot_id, 0, 6)  # Spread over 6 hours
            tasks.append(send_weather_alert_user_delayed(delay, params))
        
        # Weather alert for bot location - every 12 hours  
        if should_send_proactive_message(email, bot_id, "[WEATHER_BOT]", min_interval_hours=12):
            delay = get_staggered_delay(email, bot_id, 6, 6)  # Spread over next 6 hours
            tasks.append(send_weather_alert_bot_delayed(delay, params))
        
        # News alert for user location - every 8 hours
        if should_send_proactive_message(email, bot_id, "[NEWS_USER]", min_interval_hours=8):
            delay = get_staggered_delay(email, bot_id, 12, 4)  # Spread over 4 hours
            tasks.append(send_news_alert_user_delayed(delay, params))
        
        # News alert for bot location - every 8 hours
        if should_send_proactive_message(email, bot_id, "[NEWS_BOT]", min_interval_hours=8):
            delay = get_staggered_delay(email, bot_id, 16, 4)  # Spread over next 4 hours
            tasks.append(send_news_alert_bot_delayed(delay, params))
    
    # Execute all tasks concurrently
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
        print(f"[DEBUG] Completed {len(tasks)} staggered proactive message tasks")

async def send_weather_alert_user_delayed(delay_seconds, params):
    """Send weather alert for user location after specified delay."""
    await asyncio.sleep(delay_seconds)
    try:
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
                message=f"[WEATHER_USER] {alert}"
            )
            print(f"[DEBUG] Sent weather alert for user {params['email']}")
    except Exception as e:
        print(f"[DEBUG] Error sending weather alert for user: {e}")

async def send_weather_alert_bot_delayed(delay_seconds, params):
    """Send weather alert for bot location after specified delay."""
    await asyncio.sleep(delay_seconds)
    try:
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
                message=f"[WEATHER_BOT] {alert}"
            )
            print(f"[DEBUG] Sent weather alert for bot {params['email']}")
    except Exception as e:
        print(f"[DEBUG] Error sending weather alert for bot: {e}")

async def send_news_alert_user_delayed(delay_seconds, params):
    """Send news alert for user location after specified delay."""
    await asyncio.sleep(delay_seconds)
    try:
        alert = check_and_alert_for_major_events_user(
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
                message=f"[NEWS_USER] {alert}"
            )
            print(f"[DEBUG] Sent news alert for user {params['email']}")
    except Exception as e:
        print(f"[DEBUG] Error sending news alert for user: {e}")

async def send_news_alert_bot_delayed(delay_seconds, params):
    """Send news alert for bot location after specified delay."""
    await asyncio.sleep(delay_seconds)
    try:
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
                message=f"[NEWS_BOT] {alert}"
            )
            print(f"[DEBUG] Sent news alert for bot {params['email']}")
    except Exception as e:
        print(f"[DEBUG] Error sending news alert for bot: {e}")

@repeat_every(seconds=60*60*4)  # Every 4 hours
async def scheduled_staggered_proactive_alerts():
    """
    Trigger staggered proactive alerts every 4 hours.
    """
    print("[DEBUG] Starting scheduled staggered proactive alerts")
    await send_staggered_proactive_messages()
app = FastAPI()

# Define a Pydantic model for the incoming request body
class QuestionRequest(BaseModel):
    message: Union[str, None] = None  # Question from the user
    bot_id: str = "delhi"  # Personality type
    #removed as per new changes
    #bot_prompt: str = ""  # Personality prompt
    custom_bot_name: str = ""  # Custom bot name
    user_name : str = ""  # User name
    user_gender : str="" # User Gender
    user_location : str = ""  # User location (city)
    language : str="" # Language
    traits : str="" # Traits
    previous_conversation: list = [] # previous conversation
    email: str = ""  # Email address
    request_time : str = "" # IP address
    platform: str = "" # Platform from which the request is made

# Define the endpoint for Agent functionality
@app.post("/news_weather_agent")
async def news_weather_agent(request: QuestionRequest):
    """
    Endpoint to handle the news and weather agent functionality.
    """
    print("[DEBUG] /news_weather_agent endpoint called")
    #fetching bot_prompt from the bot_prompt.py file
    raw_bot_prompt = get_bot_prompt(request.bot_id)
    # Define the bot prompt based on the bot_id
    bot_prompt = raw_bot_prompt.format(
            custom_bot_name=request.custom_bot_name,
            traitsString=request.traits,
            userName = request.user_name,
            userGender = request.user_gender,
            languageString=request.language
        )
    start = pytime.time()
    # Call the persona_response function with the request data
    response = await persona_response(
        user_message = request.message,
        persona_prompt = bot_prompt,
        language = request.language,
        user_name = request.user_name,
        user_location = request.user_location
    )
    end = pytime.time()
    print(f"Time taken for persona_response: {end - start} seconds")
    # Return the response from the persona_response function
    return {
        "response": response,
        #"bot_prompt": bot_prompt,
        "user_name": request.user_name,
        "language": request.language
    }

# Add event handlers for scheduled tasks
app.add_event_handler("startup", scheduled_weekly_news_summary)
app.add_event_handler("startup", scheduled_major_event_alert)
app.add_event_handler("startup", scheduled_staggered_proactive_alerts)

#if background tasks didn't work, use a cron scheduler to call this endpoint
@app.post("/run_weekly_summary")
async def run_weekly_summary():
    print("[DEBUG] /run_weekly_summary endpoint called")
    scheduled_weekly_news_summary()
    return {"status": "Weekly summary triggered"}

#if background tasks didn't work, use a cron scheduler to call this endpoint
@app.post("/run_major_event_alert")
async def run_major_event_alert():
    print("[DEBUG] /run_major_event_alert endpoint called")
    scheduled_major_event_alert()
    return {"status": "Major event alert triggered"}

#if background tasks didn't work, use a cron scheduler to call this endpoint
@app.post("/run_staggered_proactive_alerts")
async def run_staggered_proactive_alerts():
    print("[DEBUG] /run_staggered_proactive_alerts endpoint called")
    await send_staggered_proactive_messages()
    return {"status": "Staggered proactive alerts triggered"}

# Manual endpoints for testing individual alert types
@app.post("/run_weather_alerts_user")
async def run_weather_alerts_user():
    print("[DEBUG] /run_weather_alerts_user endpoint called")
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
                message=f"[WEATHER_USER] {alert}"
            )
            count += 1
    return {"status": f"Weather alerts for users triggered - {count} alerts sent"}

@app.post("/run_weather_alerts_bot")
async def run_weather_alerts_bot():
    print("[DEBUG] /run_weather_alerts_bot endpoint called")
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
                message=f"[WEATHER_BOT] {alert}"
            )
            count += 1
    return {"status": f"Weather alerts for bots triggered - {count} alerts sent"}

@app.post("/run_news_alerts_bot")
async def run_news_alerts_bot():
    print("[DEBUG] /run_news_alerts_bot endpoint called")
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
                message=f"[NEWS_BOT] {alert}"
            )
            count += 1
    return {"status": f"News alerts for bots triggered - {count} alerts sent"}

# Supabase connection details
SUPABASE_URL = os.getenv("SUPABASE_URL")  # Supabase project URL from environment variable
SUPABASE_KEY = os.getenv("SUPABASE_KEY")  # Supabase API key from environment variable

# Create a Supabase client using project URL and API key
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
# -------------------- Helper Functions --------------------
def get_today_user_bot_pairs():
    print("[DEBUG] Entered get_today_user_bot_pairs")
    try:
        utc_now = datetime.now(timezone.utc)
        print(f"[DEBUG] utc_now: {utc_now}")
        start = datetime.combine(utc_now.date(), time.min, tzinfo=timezone.utc).isoformat()
        end = datetime.combine(utc_now.date(), time.max, tzinfo=timezone.utc).isoformat()
        print(f"[DEBUG] start: {start}, end: {end}")
        response = supabase.table("message_paritition") \
            .select("email, bot_id") \
            .gte("created_at", start) \
            .lte("created_at", end) \
            .execute()
        print(f"[DEBUG] Supabase response: {response}")
        pairs = {(item["email"], item["bot_id"]) for item in response.data if item.get("email") and item.get("bot_id")}
        print(f"[DEBUG] User-bot pairs: {pairs}")
        return list(pairs)
    except Exception as e:
        logging.error(f"Exception in get_today_user_bot_pairs: {e}")
        print(f"[DEBUG] Exception in get_today_user_bot_pairs: {e}")
        return []

def get_all_news_agent_params():
    print("[DEBUG] Entered get_all_news_agent_params")
    pairs = get_today_user_bot_pairs()  # [(email, bot_id), ...]
    print(f"[DEBUG] Pairs: {pairs}")
    params_list = []
    for email, bot_id in pairs:
        print(f"[DEBUG] Processing email: {email}, bot_id: {bot_id}")
        # Get user details
        user_details = supabase.table("user_details").select("*").eq("email", email).single().execute().data or {}
        print(f"[DEBUG] user_details: {user_details}")
        user_name = user_details.get("name", email.split("@")[0])
        user_gender = user_details.get("gender", "Other")
        user_city = user_details.get("city") or "India" # Default to "Delhi" if not set
        # Get bot details
        bot_details = supabase.table("bot_personality_details").select("*").eq("bot_id", bot_id).single().execute().data or {}
        print(f"[DEBUG] bot_details: {bot_details}")
        custom_bot_name = bot_details.get("bot_name", bot_id)
        bot_city = bot_details.get("bot_city", "India") # Default to "India" or "Delhi" if not set
        # Language
        language = "English"
        # Traits (if you store it, else empty)
        traits = ""
        # Get bot prompt template and fill it
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
    print(f"[DEBUG] params_list: {params_list}")
    return params_list

# Function to insert a bot message into database
def insert_bot_message(email, bot_id, message):
    print(f"[DEBUG] Inserting bot message for email: {email}, bot_id: {bot_id}, message: {message}")
    """
    Inserts a bot message into the message partition table.
    """
    supabase.table("message_paritition").insert({
        "email": email,
        "bot_id": bot_id,
        "user_message": "",
        "bot_response": message,
    }).execute()
    
    