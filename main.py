
# --- Import required libraries and modules ---
from fastapi import FastAPI  # FastAPI for building APIs
from pydantic import BaseModel  # For request validation
from typing import Union  # For type hints
from news_weather_agent import (
    persona_response,  # Handles persona-based responses
    generate_weekly_news_summary,  # Generates weekly news summary
    check_and_alert_for_major_events,  # Checks for major news events
    check_and_alert_for_weather_user,  # Checks weather alerts for users
    check_and_alert_for_weather_bot,  # Checks weather alerts for bots
    check_and_alert_for_major_events_bot  # Checks major events for bots
)
from bot_prompt import get_bot_prompt  # Gets bot prompt template
import time as pytime  # For timing operations
import uvicorn  # ASGI server
import os  # For environment variables
from dotenv import load_dotenv  # Loads environment variables from .env file
load_dotenv()  # Load environment variables
from supabase import Client, create_client  # Supabase client for database operations
from datetime import datetime, timezone, time, timedelta  # For date/time operations
import logging  # For logging errors/debug info
from fastapi_utils.tasks import repeat_every  # For scheduled tasks
from apscheduler.schedulers.background import BackgroundScheduler  # For background scheduling
import asyncio  # For async operations
import random  # For random operations
import hashlib  # For hashing
from fastapi.middleware.cors import CORSMiddleware  # For CORS support


@repeat_every(seconds=60*60*24*7)  # Every 7 days
def scheduled_weekly_news_summary():
    """
    Scheduled task to generate and store weekly news summaries for all users/bots.
    Runs every 7 days using FastAPI's repeat_every decorator.
    """
    print("[DEBUG] Entered scheduled_weekly_news_summary")
    print("=== Running Weekly News Summary for all users/bots ===")
    print("⏰ Running Weekly News Summary task every 7 days")
    for params in get_all_news_agent_params():
        print(f"[DEBUG] Params for weekly summary: {params}")
        # Generate summary for each user/bot
        summary = generate_weekly_news_summary(
            params["bot_prompt"],  # persona_prompt
            params["user_name"],
            params["language"],
            params["bot_city"],    # bot_location
            params["user_location"] 
        )
        print(f"[DEBUG] Generated summary: {summary}")
        # Store summary in database
        insert_bot_message(
            email=params["email"],
            bot_id=params["bot_id"],
            message=summary
        )


# @repeat_every(seconds=60*60*6)  # Every 6 hours
def scheduled_major_event_alert():
    """
    Scheduled task to check and alert for major news events for all users/bots.
    Intended to run every 6 hours.
    """
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



# --- APScheduler based proactive scheduling ---
def send_weather_user_alerts():
    """
    Sends weather alerts to users at scheduled times.
    """
    print("[DEBUG] send_weather_user_alerts triggered")
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
                message=f"{alert}"
            )

def send_weather_bot_alerts():
    """
    Sends weather alerts to bots at scheduled times.
    """
    print("[DEBUG] send_weather_bot_alerts triggered")
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
                message=f"{alert}"
            )

def send_news_user_alerts():
    """
    Sends major news event alerts to users at scheduled times.
    """
    print("[DEBUG] send_news_user_alerts triggered")
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
                message=f"{alert}"
            )

# Start APScheduler on FastAPI startup
def start_scheduler():
    """
    Initializes and starts the APScheduler with proactive jobs for weather/news alerts.
    """
    scheduler = BackgroundScheduler()
    scheduler.add_job(send_weather_user_alerts, 'cron', hour=8)   # 8 AM
    scheduler.add_job(send_weather_bot_alerts, 'cron', hour=14)   # 2 PM
    scheduler.add_job(send_news_user_alerts, 'cron', hour=19)     # 7 PM
    scheduler.start()
    print("[DEBUG] APScheduler started with proactive jobs.")

# --- Initialize FastAPI app ---
app = FastAPI()


# Add CORS middleware to allow requests from all origins (for frontend-backend communication)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from any origin
    allow_credentials=True,  # Allow credentials (e.g., cookies, headers)
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all HTTP headers
)


# Define a Pydantic model for the incoming request body
class QuestionRequest(BaseModel):
    """
    Model for incoming requests to the news_weather_agent endpoint.
    Contains all relevant user and bot information.
    """
    message: Union[str, None] = None  # Question from the user
    bot_id: str = "delhi"  # Personality type
    # bot_prompt removed as per new changes
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
    Accepts a QuestionRequest and returns a persona-based response.
    """
    print("[DEBUG] /news_weather_agent endpoint called")
    # Fetch bot prompt template from bot_prompt.py
    raw_bot_prompt = get_bot_prompt(request.bot_id)
    # Format the bot prompt with user/bot details
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
app.add_event_handler("startup", scheduled_weekly_news_summary)  # Weekly summary on startup
#app.add_event_handler("startup", scheduled_major_event_alert)   # Major event alert (disabled)
app.add_event_handler("startup", start_scheduler)  # Start proactive scheduler


# --- Manual endpoints for triggering scheduled tasks (for testing/cron) ---
@app.post("/run_weekly_summary")
async def run_weekly_summary():
    """
    Endpoint to manually trigger weekly news summary task.
    """
    print("[DEBUG] /run_weekly_summary endpoint called")
    scheduled_weekly_news_summary()
    return {"status": "Weekly summary triggered"}

@app.post("/run_major_event_alert")
async def run_major_event_alert():
    """
    Endpoint to manually trigger major event alert task.
    """
    print("[DEBUG] /run_major_event_alert endpoint called")
    scheduled_major_event_alert()
    return {"status": "Major event alert triggered"}

# Manual endpoints for testing individual alert types
@app.post("/run_weather_alerts_user")
async def run_weather_alerts_user():
    """
    Endpoint to manually trigger weather alerts for users.
    """
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
                message=f"{alert}"
            )
            count += 1
    return {"status": f"Weather alerts for users triggered - {count} alerts sent"}

@app.post("/run_weather_alerts_bot")
async def run_weather_alerts_bot():
    """
    Endpoint to manually trigger weather alerts for bots.
    """
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
                message=f"{alert}"
            )
            count += 1
    return {"status": f"Weather alerts for bots triggered - {count} alerts sent"}

@app.post("/run_news_alerts_bot")
async def run_news_alerts_bot():
    """
    Endpoint to manually trigger news alerts for bots.
    """
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
                message=f"{alert}"
            )
            count += 1
    return {"status": f"News alerts for bots triggered - {count} alerts sent"}


# --- Supabase connection details ---
SUPABASE_URL = os.getenv("SUPABASE_URL")  # Supabase project URL from environment variable
SUPABASE_KEY = os.getenv("SUPABASE_KEY")  # Supabase API key from environment variable

# Create a Supabase client using project URL and API key
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# -------------------- Helper Functions --------------------
def get_today_user_bot_pairs():
    """
    Fetches all (email, bot_id) pairs for today from the message partition table.
    Returns a list of tuples.
    """
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
    """
    For each user-bot pair, fetches user and bot details, formats the bot prompt,
    and returns a list of parameter dictionaries for agent functions.
    """
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
        user_city = user_details.get("city") or "India" # Default to "India" if not set
        # Get bot details
        bot_details = supabase.table("bot_personality_details").select("*").eq("bot_id", bot_id).single().execute().data or {}
        print(f"[DEBUG] bot_details: {bot_details}")
        custom_bot_name = bot_details.get("bot_name", bot_id)
        bot_city = bot_details.get("bot_city", "India") # Default to "India" if not set
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

def insert_bot_message(email, bot_id, message):
    """
    Inserts a bot message into the message partition table in Supabase.
    """
    print(f"[DEBUG] Inserting bot message for email: {email}, bot_id: {bot_id}, message: {message}")
    supabase.table("message_paritition").insert({
        "email": email,
        "bot_id": bot_id,
        "user_message": "",
        "bot_response": message,
    }).execute()
    
