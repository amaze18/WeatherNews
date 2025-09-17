import pycountry
# Import required modules
from crewai import Task, Crew, Agent, Process
from crewai.tools import BaseTool
import requests
import re, os, time, json
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

    def _run(self, city_name: str) -> str:
        api_key = os.environ.get("OPENWEATHER_API_KEY")
        if not api_key: return "Error: OPENWEATHER_API_KEY not set."
        base_url = "http://api.openweathermap.org/data/2.5/weather?"
        complete_url = f"{base_url}q={city_name}&appid={api_key}&units=metric"
        try:
            response = requests.get(complete_url, timeout=10)
            response.raise_for_status()
            data = response.json()
            main_data = data["main"]
            weather_data = data["weather"][0]
            return (
                f"city: {city_name}, temperature: {main_data['temp']}°C, "
                f"weather: {weather_data['description']}"
            )
        except Exception as e:
            return f"Error fetching weather data for {city_name}: {e}"

class NewsAPITool(BaseTool):
    name: str = "Context-Aware News Search"
    description: str = "Searches for recent, relevant news articles for a given location."

    def _get_country_context(self, query: str) -> str:
        try:
            return pycountry.countries.lookup(query).name
        except LookupError:
            return query

    def _run(self, query: str) -> str:
        api_key = os.environ.get("NEWSAPI_API_KEY")
        if not api_key: return "Error: NEWSAPI_API_KEY not found."
        context_term = self._get_country_context(query)
        intelligent_query = f'"{query}"'
        if query.lower() != context_term.lower():
            intelligent_query += f' AND "{context_term}"'
        
        print(f"[NewsAPITool] INFO: Constructed search query: {intelligent_query}")
        url = f"https://newsapi.org/v2/everything?qInTitle={intelligent_query}&language=en&sortBy=relevancy&pageSize=5&apiKey={api_key}"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            articles = data.get("articles", [])
            if not articles:
                return f"No relevant news found with '{query}' in the headline."
            formatted_headlines = [f"- {art['title']} (Source: {art['source']['name']})" for art in articles]
            return "Here are the most relevant headlines:\n" + "\n".join(formatted_headlines)
        except Exception as e:
            return f"An unexpected error occurred while fetching news: {e}"

# --- Initialize Tools & Crew ---
open_weather_map_tool = OpenWeatherMapTool()
news_api_tool = NewsAPITool()
llm = 'gemini/gemini-1.5-flash'

news_researcher = Agent(
    role='Expert News Reporter',
    goal='Fetch recent news articles for a given search {topic}.',
    verbose=True, memory=False, backstory="You are a reporter skilled at finding relevant stories.",
    tools=[news_api_tool], llm=llm,
)
research_task = Task(
    description="Search for the latest news related to {topic}.",
    expected_output="A bulleted list of top headlines and their sources.",
    agent=news_researcher,
)
crew = Crew(agents=[news_researcher], tasks=[research_task], process=Process.sequential)

# --- Core Helper Functions ---

# ✅ NEW: This single function replaces call_gemma_classify and extract_user_location. It's much faster.
async def analyze_user_request(user_message: str, user_location: str) -> dict:
    prompt = f"""
Analyze the user's message and return a JSON object with three keys: "category", "location", and "is_temperature_query".

1.  **category**: Classify the message as "news", "weather", or "other".
2.  **location**: Extract the main city or country. If no location is mentioned, use the provided default location.
3.  **is_temperature_query**: Set to `true` if the message specifically asks about temperature, otherwise `false`.

User Message: "{user_message}"
Default Location: "{user_location or 'Hyderabad'}"

JSON Response:
"""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key: return {"category": "other", "location": user_location or "Hyderabad", "is_temperature_query": False}
    
    model = "gemini-1.5-flash"
    llm = GoogleGenAI(model=model, api_key=api_key)
    
    try:
        response = llm.complete(prompt)
        # Clean up the response to ensure it's valid JSON
        clean_response = response.text.strip().replace("```json", "").replace("```", "")
        analysis = json.loads(clean_response)
        print(f"[DEBUG] Request Analysis: {analysis}")
        return analysis
    except Exception as e:
        print(f"[ERROR] Failed to analyze user request: {e}")
        return {"category": "other", "location": user_location or "Hyderabad", "is_temperature_query": False}

def extract_bot_location(persona_prompt):
    match = re.search(r'(?:from|in)\s+([A-Za-z\s]+)[\.,]', persona_prompt, re.IGNORECASE)
    return match.group(1).strip() if match else "Delhi"

# --- Persona and Response Generation ---

def create_persona_summary(news_summary: str, persona_prompt: str, user_name: str, location: str, language: str) -> str:
    # This is the detailed prompt for high-quality news responses
    llm_prompt = f"""
Your Persona: {persona_prompt}
You are talking to your friend, {user_name}, in {language}.
You just read these news headlines about {location}:
---
{news_summary}
---
Synthesize these headlines into a detailed, 3-4 sentence conversational summary. Mention at least two interesting headlines and offer a brief reflection on them from your persona's point of view. End with a natural, engaging question. Do not just list the headlines.
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

# ✅ NEW, UPGRADED FUNCTION
def create_persona_weather_response(weather_data: str, persona_prompt: str, user_name: str, location: str, language: str, is_temp_query: bool):
    """
    Generates a high-quality, persona-driven response for weather queries.
    It handles general weather and specific temperature requests differently.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    model = "gemini-1.5-flash"
    llm = GoogleGenAI(model=model, api_key=api_key)

    if is_temp_query:
        # Task is specifically about temperature, so we must find and include the number.
        match = re.search(r"temperature:\s*([\d\.-]+)°C", weather_data)
        if not match:
            return f"I'm sorry, I couldn't find the exact temperature for {location} right now."
        
        temperature = match.group(1) # e.g., "21.58"

        llm_prompt = f"""
Your Persona: {persona_prompt}
You are talking to your friend, {user_name}, in {language}.
Your task is to tell them the temperature in {location}.

The exact temperature is {temperature}°C.

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

Based on your persona, write a brief, 1-2 sentence conversational message.
You MUST include the exact temperature value '{temperature}°C' in your response. The temperature must be expressed in numbers with its decimal points, not written out as words.
End with a natural, engaging question.

Example: "I just checked, and it's exactly {temperature}°C in {location} right now. A bit chilly if you ask me! Are you staying warm?"


Format: Only say how you feel about the temperature, in {language}, as if talking to a friend named {user_name}. Include the temperature in the response and make it sound natural to your persona.Only the temperature should be expressed in numbers, not words.
"""
    else:
        # Task is about general weather.
        llm_prompt = f"""
Your Persona: {persona_prompt}
You are talking to your friend, {user_name}, in {language}.
You just saw the weather report for {location}: "{weather_data}"
Based on your persona, write a brief, 1-2 sentence conversational message about the general weather. Do not just repeat the data. React to it naturally and end with an engaging question.
"""

    try:
        response = llm.complete(llm_prompt)
        return response.text.strip()
    except Exception as e:
        print(f"[ERROR] Persona weather response generation failed: {e}")
        # Provide a simple but accurate fallback if the LLM fails
        if is_temp_query and 'temperature' in locals():
             return f"It's {temperature}°C in {location}. Hope you're having a good day!"
        else:
            return f"I was just looking at the weather in {location}. Hope you're having a good day!"
# --- Main Handler ---
async def persona_response(user_message, persona_prompt, language, user_name, user_location=None):
    start_time = time.time()
    
    # Step 1: Analyze the request in a single, fast call
    analysis = await analyze_user_request(user_message, user_location)
    category = analysis.get("category", "other")
    location = analysis.get("location", user_location or "Hyderabad")
    is_temp_query = analysis.get("is_temperature_query", False)

    bot_location = extract_bot_location(persona_prompt)
    context = 'bot' if any(keyword in user_message.lower() for keyword in ['your city', 'your place']) else 'user'
    
    response = "I can chat about recent news or the current weather. What's on your mind?"

    if category == "news":
        news_location = location if context == "user" else bot_location
        if news_location:
            result = crew.kickoff(inputs={'topic': news_location})
            response = create_persona_summary(str(result), persona_prompt, user_name, news_location, language)

    elif category == "weather":
        weather_location = location if context == "user" else bot_location
        if weather_location:
            result = open_weather_map_tool._run(city_name=weather_location)
            response = create_persona_weather_response(result, persona_prompt, user_name, weather_location, language, is_temp_query)
            
    end_time = time.time()
    print(f"[DEBUG] Final response generated in {end_time - start_time:.2f} seconds: {response}")
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