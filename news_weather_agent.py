import pycountry
# Import required modules
from crewai import Task, Crew, Agent, Process
from googleapiclient.discovery import build
from crewai.tools import BaseTool
import requests
from bs4 import BeautifulSoup
from crewai.tools import BaseTool
import re, os, time
from pathlib import Path
from dotenv import load_dotenv
from llama_index.llms.google_genai import GoogleGenAI
import asyncio

# --- Robust Environment Variable Loading ---
project_dir = Path.cwd()
dotenv_path = project_dir / ".env"
load_dotenv(dotenv_path=dotenv_path)

# --- Tool Definitions ---

class OpenWeatherMapTool(BaseTool):
    name: str = "OpenWeatherMap Weather Fetch"
    description: str = "Fetches the current weather conditions for a given city."

    def _normalize_location(self, city_name: str) -> str:
        mapping = {"jumeirah": "Dubai"}
        return mapping.get(city_name.strip().lower(), city_name)

    def _run(self, city_name: str) -> str:
        normalized_city = self._normalize_location(city_name)
        api_key = os.environ.get("OPENWEATHER_API_KEY")
        if not api_key: return "Error: OPENWEATHER_API_KEY not set."
        
        base_url = "http://api.openweathermap.org/data/2.5/weather?"
        complete_url = f"{base_url}q={normalized_city}&appid={api_key}&units=metric"

        try:
            response = requests.get(complete_url, timeout=10)
            response.raise_for_status()
            data = response.json()
            main_data = data["main"]
            weather_data = data["weather"][0]
            return (
                f"city: {normalized_city}, temperature: {main_data['temp']}°C, "
                f"pressure: {main_data['pressure']} hPa, humidity: {main_data['humidity']}%, "
                f"weather: {weather_data['description']}"
            )
        except Exception as e:
            return f"Error fetching weather data for {normalized_city}: {e}"

class NewsAPITool(BaseTool):
    name: str = "Keyword News Search"
    description: str = "Searches for recent news articles based on a keyword or topic (e.g., 'India', 'Paris')."

    def _run(self, query: str) -> str:
        api_key = os.environ.get("NEWSAPI_API_KEY")
        if not api_key:
            return "Error: NEWSAPI_API_KEY not found."

        try:
            country_name = pycountry.countries.lookup(query).name
        except LookupError:
            country_name = query

        intelligent_query = f'"{query}" AND ("{country_name}" OR politics OR government OR economy OR national)'
        
        url = f"https://newsapi.org/v2/everything?q={intelligent_query}&language=en&sortBy=relevancy&pageSize=5&apiKey={api_key}"

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            articles = data.get("articles", [])

            if not articles:
                return f"No relevant national or political news found for '{query}'."

            formatted_headlines = [f"- {art['title']} (Source: {art['source']['name']})" for art in articles]
            return "Here are the most relevant headlines:\n" + "\n".join(formatted_headlines)

        except Exception as e:
            return f"An unexpected error occurred while fetching news: {e}"

# --- Initialize Tools ---
open_weather_map_tool = OpenWeatherMapTool()
news_api_tool = NewsAPITool()

# --- Agent and Task Definitions ---
llm = 'gemini/gemini-1.5-flash'

news_researcher = Agent(
    role='Expert News Reporter',
    goal='Fetch and concisely summarize recent news articles for a given search {topic}.',
    verbose=True,
    memory=False,
    backstory="You are a top-tier news reporter skilled at finding the most relevant stories for a topic.",
    tools=[news_api_tool],
    llm=llm,
)

research_task = Task(
    description="Search for the latest news related to the {topic}. The topic will be a location or subject (e.g., 'India', 'Paris').",
    expected_output="A clean, bulleted list of the top news headlines found, including the title and source.",
    agent=news_researcher,
)

crew = Crew(
    agents=[news_researcher],
    tasks=[research_task],
    process=Process.sequential,
)

# --- Helper Functions ---
async def call_gemma_classify(user_message: str) -> str:
    prompt = f"""
Classify the following user message as 'news', 'weather', or 'other'. Respond with only one word.

Here are some examples:
- "What's the weather like in Delhi?" -> weather
- "Is it going to rain tomorrow?" -> weather
- "How hot is it in Dubai?" -> weather
- "Tell me the latest news in India." -> news
- "What's happening in Germany?" -> news
- "Any updates on the election?" -> news
- "How was your day?" -> other
- "What's the news at my brother's wedding?" -> other
- "Did you see that movie?" -> other

User message: "{user_message}"
"""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key: return "other"
    
    model = "gemini-1.5-flash"
    llm = GoogleGenAI(model=model, api_key=api_key)
    
    for attempt in range(2):
        try:
            response = llm.complete(prompt)
            result = response.text.strip().lower()
            if result in ["news", "weather", "other"]:
                print(f"[DEBUG] Classification result for '{user_message}': {result}")
                return result
        except Exception as e:
            print(f"[ERROR] LLM classification call failed on attempt {attempt + 1}: {e}")
            if attempt < 1:
                time.sleep(1)
    
    print("[DEBUG] Classification failed after multiple attempts.")
    return "other"

def extract_bot_location(persona_prompt):
    match = re.search(r'(?:from|in)\s+([A-Za-z\s]+)[\.,]', persona_prompt, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return "Delhi"

def extract_user_location(user_message, user_location):
    api_key = os.environ.get("GEMINI_API_KEY")
    model = "gemini-1.5-flash"
    llm = GoogleGenAI(model=model, api_key=api_key)
    llm_prompt = f"From the user message '{user_message}', extract only the main city or country name mentioned. If no location is found, respond with '{user_location or 'Hyderabad'}'."
    try:
        response = llm.complete(llm_prompt)
        return response.text.strip()
    except Exception:
        return user_location or "Hyderabad"

def detect_location_context(user_message, persona_prompt):
    if any(keyword in user_message.lower() for keyword in ['your city', 'your place', 'where you are']):
        return 'bot'
    return 'user'

# --- Persona and Response Generation ---

def get_persona_feeling(persona_prompt, summary, user_name, language, bot_location, context="bot", topic="weather"):
    if topic == "news":
        llm_prompt = f"Based on this personality: '{persona_prompt}', and this news: '{summary[:400]}', write a 1-2 sentence reaction in {language} as if talking to your friend {user_name}."
    else: # weather
        llm_prompt = f"Based on this personality: '{persona_prompt}', and this weather report: '{summary[:200]}', write a 1-2 sentence reaction in {language} as if talking to your friend {user_name}. Do not repeat the weather conditions."

    api_key = os.environ.get("GEMINI_API_KEY")
    model = "gemini-1.5-flash"
    llm = GoogleGenAI(model=model, api_key=api_key)
    try:
        response = llm.complete(llm_prompt)
        return response.text.strip()
    except Exception as e:
        print(f"[ERROR] Persona feeling generation failed: {e}")
        return "That's quite interesting, isn't it?"

def create_persona_summary(news_summary: str, persona_prompt: str, user_name: str, location: str, language: str) -> str:
    """
    Takes a news summary and generates a conversational, persona-driven response.
    """
    llm_prompt = f"""
Your Persona: {persona_prompt}

You are talking to your friend, {user_name}, in {language}.
You just read the following news headlines about {location}:
---
{news_summary}
---
Based on your persona, synthesize these headlines into a brief, 1-2 sentence conversational summary. 
Start by mentioning the location. Pick the most interesting headline to briefly comment on, according to your personality. 
End with a natural, engaging question to your friend. Do not just list the headlines.

Example Response: "I saw a few things happening in Paris, dear. It seems there's some big political debate going on. It does make one think, doesn't it? What's on your mind?"

Your turn:
"""
    api_key = os.environ.get("GEMINI_API_KEY")
    model = "gemini-1.5-flash"
    llm = GoogleGenAI(model=model, api_key=api_key)
    try:
        response = llm.complete(llm_prompt)
        return response.text.strip()
    except Exception as e:
        print(f"[ERROR] Persona summary generation failed: {e}")
        return "I saw some news, but I'm not sure what to make of it. What do you think?"

def get_weather_response(weather_summary, persona_prompt, user_name, language, bot_location, user_location = None , context = "bot"):
    match = re.search(r"weather:\s*(.*?)(?:$|,)", weather_summary, re.IGNORECASE)
    weather_desc = match.group(1).lower().strip() if match else "pleasant"
    
    feeling = get_persona_feeling(persona_prompt, weather_summary, user_name, language, bot_location, context=context, topic="weather")
    
    if context == "user" and user_location:
        return f"I saw {user_location} had {weather_desc} weather, {feeling}."
    else:
        return f"It’s {weather_desc} in {bot_location}, {feeling}."

def get_temperature_response(weather_summary, persona_prompt, user_name, language, location):
    match = re.search(r"temperature:\s*([\d\.-]+)°C", weather_summary, re.IGNORECASE)
    if not match: return "I'm sorry, I couldn't find the temperature."
    temperature = match.group(1)

    llm_prompt = f"""
Based on the personality given, Respond in {language}, 1 or 2 sentence, describing the temperature in {location}, using your personality traits below.
You must include the temperature of {temperature}°C in your response.
Only the temperature should be expressed in numbers, not words.

Personality: {persona_prompt}

Format: Only say how you feel about the temperature, in {language}, as if talking to a friend named {user_name}. Include the temperature in the response and make it sound natural to your persona.Only the temperature should be expressed in numbers, not words.
"""
    api_key = os.environ.get("GEMINI_API_KEY")
    model = "gemini-1.5-flash"
    llm = GoogleGenAI(model=model, api_key=api_key)
    try:
        response = llm.complete(llm_prompt)
        return response.text.strip()
    except Exception as e:
        print(f"[ERROR] Temperature response generation failed: {e}")
        return f"Ugh, my phone died. It's {temperature}°C in {location} though, I guess."

# --- Main Handler ---
async def persona_response(user_message, persona_prompt, language, user_name, user_location=None):
    print(f"[DEBUG] persona_response called with user_message: {user_message}")
    category = await call_gemma_classify(user_message)

    bot_location = extract_bot_location(persona_prompt)
    current_user_location = extract_user_location(user_message, user_location)
    context = detect_location_context(user_message, persona_prompt)

    response = "I'm sorry, something went wrong. Could you ask that again?"

    if category == "news":
        news_location = current_user_location if context == "user" else bot_location
        if news_location:
            result = crew.kickoff(inputs={'topic': news_location})
            response = create_persona_summary(str(result), persona_prompt, user_name, news_location, language)
        else:
            response = "I'm not sure which location you're asking about. Could you be more specific?"

    elif category == "weather":
        weather_location = current_user_location if context == "user" else bot_location
        if weather_location:
            result = open_weather_map_tool._run(city_name=weather_location)
            if re.search(r'\btemperature\b|\bhow hot\b|\bhow cold\b', user_message, re.IGNORECASE):
                response = get_temperature_response(result, persona_prompt, user_name, language, weather_location)
            else:
                response = get_weather_response(result, persona_prompt, user_name, language, bot_location, weather_location, context)
        else:
             response = "I'm not sure which location you're asking about for the weather."
    else:
        response = "I can chat about recent news or the current weather. What's on your mind?"
        
    print(f"[DEBUG] Final response: {response}")
    return response

# --- Alerting and Summary Functions ---
def is_interesting_weather(weather_text):
    keywords = [
        "storm", "thunderstorm", "heavy rain", "downpour", "flooding", "hailstorm",
        "heat wave", "cold wave", "freezing", "snow", "blizzard", "cyclone", "hurricane",
        "tornado", "extreme", "severe", "warning", "alert", "advisory", "dangerous",
        "record high", "record low", "unusual", "unprecedented", "fog", "smog",
        "very hot", "very cold", "scorching", "chilly", "humid"
    ]
    pattern = r"|".join([re.escape(word) for word in keywords])
    return re.search(pattern, weather_text, re.IGNORECASE) is not None

def is_major_event(news_text):
    keywords = [
        "election", "government collapse", "cabinet reshuffle",
        "inflation", "stock market crash", "economic crisis", "policy change", "sanctions",
        "war", "protest", "strike", "currency devaluation", "interest rate hike", "recession",
        "earthquake", "flood", "protest", "accident", "crime", "strike",
        "curfew", "violence", "celebration", "shutdown", "alert", "breaking", "death", "murder", "investigation"
    ]
    pattern = r"|".join([re.escape(word) for word in keywords])
    return re.search(pattern, news_text, re.IGNORECASE) is not None

def generate_weekly_news_summary(persona_prompt, user_name, language,bot_location, user_location = "India"):
    topic = f"Latest National news in {user_location} this week"
    result = crew.kickoff(inputs={'topic': topic})
    news_summary = str(result)
    # The create_persona_summary function is better for this
    response = create_persona_summary(news_summary, persona_prompt, user_name, user_location, language)
    return response

def check_and_alert_for_weather_user(persona_prompt, user_name, language, bot_location, user_location="India"):
    try:
        weather_summary = open_weather_map_tool._run(city_name=user_location)
        if is_interesting_weather(weather_summary):
            context = "user" if user_location != bot_location else "bot"
            response = get_weather_response(weather_summary, persona_prompt, user_name, language, bot_location, user_location, context)
            return response
        else:
            print(f"No interesting weather conditions for {user_location}.")
            return None
    except Exception as e:
        print(f"Error in weather alert for user location: {e}")
        return None

def check_and_alert_for_weather_bot(persona_prompt, user_name, language, bot_location, user_location="India"):
    try:
        weather_summary = open_weather_map_tool._run(city_name=bot_location)
        if is_interesting_weather(weather_summary):
            response = get_weather_response(weather_summary, persona_prompt, user_name, language, bot_location, user_location, "bot")
            return response
        else:
            print(f"No interesting weather conditions for {bot_location}.")
            return None
    except Exception as e:
        print(f"Error in weather alert for bot location: {e}")
        return None

def check_and_alert_for_major_events_user(persona_prompt, user_name, language, bot_location, user_location="India"):
    try:
        topic = f"Latest National news in {user_location}"
        result = crew.kickoff(inputs={'topic': topic})
        news_summary = str(result)
        if is_major_event(news_summary):
            response = create_persona_summary(news_summary, persona_prompt, user_name, user_location, language)
            return response
        else:
            print(f"No major political/economic/tragic event detected for {user_location}.")
            return None
    except Exception as e:
        print(f"Error in news alert for user location: {e}")
        return None

def check_and_alert_for_major_events_bot(persona_prompt, user_name, language, bot_location, user_location="India"):
    try:
        topic = f"Latest National news in {bot_location}"
        result = crew.kickoff(inputs={'topic': topic})
        news_summary = str(result)
        if is_major_event(news_summary):
            response = create_persona_summary(news_summary, persona_prompt, user_name, bot_location, language)
            return response
        else:
            print(f"No major political/economic/tragic event detected for {bot_location}.")
            return None
    except Exception as e:
        print(f"Error in news alert for bot location: {e}")
        return None

def check_and_alert_for_major_events(persona_prompt, user_name, language, bot_location, user_location="India"):
    return check_and_alert_for_major_events_user(persona_prompt, user_name, language, bot_location, user_location)