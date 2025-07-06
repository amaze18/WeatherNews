from fastapi import FastAPI
from pydantic import BaseModel
from typing import Union
from news_weather_agent import persona_response
from bot_prompt import get_bot_prompt
import time
import uvicorn
import os

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
    start = time.time()
    # Call the persona_response function with the request data
    response = await persona_response(
        user_message = request.message,
        persona_prompt = bot_prompt,
        language = request.language,
        user_name = request.user_name,
        user_location = None
    )
    end = time.time()
    print(f"Time taken for persona_response: {end - start} seconds")
    # Return the response from the persona_response function
    return {
        "response": response,
        #"bot_prompt": bot_prompt,
        "user_name": request.user_name,
        "language": request.language
    }

