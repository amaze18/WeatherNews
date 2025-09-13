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

def insert_bot_message(email, bot_id, message):
    supabase.table("message_paritition").insert({
        "email": email,
        "bot_id": bot_id,
        "user_message": "",
        "bot_response": message,
    }).execute()

def insert_user_message(email, bot_id, user_message, bot_response):
    """Insert both user message and bot response into the database"""
    print(f"[SUPABASE] Attempting to store message for {email} with bot {bot_id}")
    print(f"[SUPABASE] User message: {user_message[:100]}{'...' if len(user_message) > 100 else ''}")
    print(f"[SUPABASE] Bot response: {bot_response[:100]}{'...' if len(bot_response) > 100 else ''}")
    
    try:
        response = supabase.table("message_paritition").insert({
            "email": email,
            "bot_id": bot_id,
            "user_message": user_message,
            "bot_response": bot_response,
        }).execute()
        
        if response.data and len(response.data) > 0:
            record_id = response.data[0].get('id', 'N/A')
            print(f"[SUPABASE] ✅ SUCCESS: Message stored with ID {record_id}")
            print(f"[SUPABASE] ✅ Email: {email}")
            print(f"[SUPABASE] ✅ Bot ID: {bot_id}")
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
        logging.error(f"Failed to insert message for {email}: {e}")
        return False

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
            storage_success = insert_user_message(
                email=request.email,
                bot_id=request.bot_id,
                user_message=request.message or "",
                bot_response=response
            )
            if storage_success:
                print(f"[CHAT] ✅ Conversation successfully saved to Supabase for {request.email}")
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
            message=summary
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
                message=alert
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
                message=alert
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
                message=alert
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
                message=alert
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
                message=alert
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
                message=alert
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
                message=alert
            )
            count += 1
    return {"status": f"News alerts for bots triggered - {count} alerts sent"}
