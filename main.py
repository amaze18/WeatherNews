from fastapi import FastAPI
from pydantic import BaseModel
from typing import Union
from news_weather_agent import persona_response, generate_weekly_news_summary, check_and_alert_for_major_events
from bot_prompt import get_bot_prompt
import time as pytime
import uvicorn
import os
from dotenv import load_dotenv
load_dotenv()
from supabase import Client, create_client
from datetime import datetime, timezone, time
import logging
#from fastapi_utils.tasks import repeat_every

#@repeat_every(seconds=60*60*24*7)  # Every 7 days
def scheduled_weekly_news_summary():
    print("=== Running Weekly News Summary for all users/bots ===")
    print("⏰ Running Weekly News Summary task every 7 days")
    for params in get_all_news_agent_params():
        # Pass all required params to your agent function
        summary = generate_weekly_news_summary(
            params["bot_prompt"],  # persona_prompt
            params["user_name"],
            params["language"],
            params["bot_city"],    # bot_location
            params["user_location"] 
        )
        insert_bot_message(
            email=params["email"],
            bot_id=params["bot_id"],
            message=summary
        )

#@repeat_every(seconds=60*60*6)  # Every 6 hours
def scheduled_major_event_alert():
    print("=== Checking for Major Events for all users/bots ===")
    print("⏰ Running Major Event Alert task every 6 hours")
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
#app.add_event_handler("startup", scheduled_weekly_news_summary)
#app.add_event_handler("startup", scheduled_major_event_alert)

@app.post("/run_weekly_summary")
async def run_weekly_summary():
    scheduled_weekly_news_summary()
    return {"status": "Weekly summary triggered"}

@app.post("/run_major_event_alert")
async def run_major_event_alert():
    scheduled_major_event_alert()
    return {"status": "Major event alert triggered"}

# Supabase connection details
SUPABASE_URL = os.getenv("SUPABASE_URL")  # Supabase project URL from environment variable
SUPABASE_KEY = os.getenv("SUPABASE_KEY")  # Supabase API key from environment variable

# Create a Supabase client using project URL and API key
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
# -------------------- Helper Functions --------------------
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
    pairs = get_today_user_bot_pairs()  # [(email, bot_id), ...]
    params_list = []
    for email, bot_id in pairs:
        # Get user details
        user_details = supabase.table("user_details").select("*").eq("email", email).single().execute().data or {}
        user_name = user_details.get("name", email.split("@")[0])
        user_gender = user_details.get("gender", "Other")
        user_city = user_details.get("city") or "India" # Default to "Delhi" if not set
        # Get bot details
        bot_details = supabase.table("bot_personality_details").select("*").eq("bot_id", bot_id).single().execute().data or {}
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
    return params_list

# Function to insert a bot message into database
def insert_bot_message(email, bot_id, message):
    """
    Inserts a bot message into the message partition table.
    """
    supabase.table("message_paritition").insert({
        "email": email,
        "bot_id": bot_id,
        "user_message": "",
        "bot_response": message,
    }).execute()
    
    