# Import required modules
from crewai import Task, Crew, Agent, Process
from googleapiclient.discovery import build
from crewai.tools import BaseTool
import requests
from bs4 import BeautifulSoup
from crewai.tools import BaseTool
import re, os
from pathlib import Path
from dotenv import load_dotenv
from llama_index.llms.google_genai import GoogleGenAI
import asyncio

# --- Robust Environment Variable Loading ---
# Get the absolute path to the directory containing the script
project_dir = Path(__file__).parent.absolute()
dotenv_path = project_dir / ".env"
load_dotenv(dotenv_path=dotenv_path)

# --- Google Search Functionality ---

def google_search(query, api_key=os.environ.get("GOOGLE_SEARCH_API_KEY"), cse_id=os.environ.get("GOOGLE_CSE_ID"), **kwargs):
    """
    Perform a Google search using the Custom Search JSON API.

    Args:
        query (str): The search query.
        api_key (str): API key for Google Custom Search.
        cse_id (str): Custom Search Engine ID.
        **kwargs: Additional parameters for the search.

    Returns:
        list: A list of search result items.
    """
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=query, cx=cse_id, **kwargs).execute()
    return res.get('items', [])

# Function to fetch page content
def fetch_page_content(url):
    """
    Fetch the content of a webpage and return its text.

    Args:
        url (str): The URL of the webpage to fetch.

    Returns:
        str: Extracted text content from the webpage.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=1.5)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text(separator="\n", strip=True)
        return text[150:2000] + "..." if len(text) > 1850 else text
    except Exception as e:
        return f"Error fetching page: {e}"

# --- Custom Tool for Google Search and Page Fetch ---

class GoogleSearchAndFetchTool(BaseTool):
    """
    Custom tool to perform a Google search and fetch summaries of the top result pages.
    """
    name: str = "Google Search with Page Fetch"
    description: str = "Performs a Google search and fetches a summary of the top 3 result pages."

    def _run(self, query: str) -> str:
        """
        Execute the tool with the given query.

        Args:
            query (str): The search query.

        Returns:
            str: Summarized content of the top search results.
        """
        if "news" in query.lower():
            query = f"site:news.google.com {query}"
        print(query)
        results = google_search(query, num=1)
        output = []
        for item in results[:1]:
            title = item.get("title")
            link = item.get("link")
            snippet = item.get("snippet", "")
            content = fetch_page_content(link)
            output.append(f"{title}\n{link}\n{content}\n{'-'*40}")
        return "\n".join(output)

# --- NEW: Custom Tool for OpenWeatherMap API ---
class OpenWeatherMapTool(BaseTool):
    """
    Custom tool to fetch real-time weather data from OpenWeatherMap API.
    """
    name: str = "OpenWeatherMap Weather Fetch"
    description: str = "Fetches the current weather conditions for a given city."

    def _run(self, city_name: str) -> str:
        """
        Execute the tool with the given city name.

        Args:
            city_name (str): The name of the city.

        Returns:
            str: A formatted string of the current weather conditions.
        """
        api_key = os.environ.get("OPENWEATHER_API_KEY")
        base_url = "http://api.openweathermap.org/data/2.5/weather?"
        complete_url = f"{base_url}q={city_name}&appid={api_key}&units=metric"

        try:
            response = requests.get(complete_url)
            data = response.json()

            if data.get("cod") != 200:
                return f"Error: Could not retrieve weather for {city_name}. API response: {data.get('message', 'Unknown error')}"
            else:
                main_data = data["main"]
                weather_data = data["weather"][0]
                temperature = main_data["temp"]
                pressure = main_data["pressure"]
                humidity = main_data["humidity"]
                description = weather_data["description"]
                # Use a consistent, lowercase format for the output
                return f"city: {city_name}, temperature: {temperature}°C, pressure: {pressure} hPa, humidity: {humidity}%, weather: {description}"

        except requests.exceptions.RequestException as e:
            return f"Error fetching weather data: {e}"

# Initialize the custom tools
custom_google_search_tool = GoogleSearchAndFetchTool()
open_weather_map_tool = OpenWeatherMapTool()

# --- Define Agents ---

# Define the news researcher agent
llm = 'gemini/gemini-2.0-flash'

news_researcher = Agent(
    role='Situational Expert',
    goal='Provide accurate, up-to-date, and practical answers to any real-world question, such as recommendations or current conditions about {topic}.Ignore the top hyperlinks in the news websites, because they are irrelevant one. Try to be as quick as possible',
    verbose=True,
    memory=False,  # Set memory to False for quick responses
    backstory=(
        "You are a resourceful expert, skilled at quickly gathering and analyzing the latest information. "
        "Whether it's about places, events, or current conditions, you deliver clear, reliable, and actionable insights for any question."
    ),
    tools=[custom_google_search_tool],  # This agent uses the Google search tool for news
    llm=llm,
)

# Define tasks
research_task = Task(
    description=(
        "Research and gather the most accurate, up-to-date, and practical information about: {topic}. "
        "Focus on providing clear answers, recommendations, or current conditions relevant to the question. "
        "Include details that help someone make decisions or understand the situation right now."
    ),
    expected_output=(
    "A concise summary with key facts, recommendations, or current status about {topic}, "
    "suitable for someone seeking real-world, actionable information."
    ),
    tools=[custom_google_search_tool],
    agent=news_researcher,
)

# Initialize the crew
crew = Crew(
    agents=[news_researcher],
    tasks=[research_task],
    process=Process.sequential,
    max_iter=1,
    max_execution_time=8
)

# --- Helper Functions ---

def resolve_location(user_location, llm):
    """
    Resolve the location of a user using the LLM.
    ... (rest of the function remains the same)
    """
    location = llm.complete(
        f"Only give the answer for the question\n"
        f"If user_location is country, then answer the same name, if it is city, then answer in the country which that city belongs\n"
        f"What is the location of {user_location}?\n"
    ).text.strip()
    return location

def get_persona_feeling(persona_prompt, summary, user_name, language, context="bot", topic="weather"):
    """
    Determine the feeling or emotional response of a persona to a given news or weather summary.
    ... (rest of the function remains the same)
    """
    if topic == "news":
        if context == "user":
            llm_prompt = f"""
Based on the personality given, Respond in {language}, 1 or 2 sentence, describing how you feel about this news (only one news) in user location, using your personality traits below.

Personality: {persona_prompt}

Situation: {summary[:500]}

Format: Only say how you feel, in {language}, as if talking to a friend named {user_name}. Repeat the news description as 'I saw in the news that (city) had (content) and then the feeling' and then ask the user how he feels, in your Personality.
"""
        else:
            llm_prompt = f"""
Based on the personality given, Respond in {language}, 1 or 2 sentence, describing how you feel about this news (only one news) in your location (i.e. Bot location), using your personality traits below.

Personality: {persona_prompt}

Situation: {summary[:500]}
 
Format: Only say how you feel, in {language}, as if talking to a friend named {user_name}. Just explain the incident in brief in your personality to explain the user, Then ask the user about the news in his location in your Personality. Also explain the incident in brief in your personality to explain the user.
"""
    else:  # weather
        if context == "user":
            llm_prompt = f"""
Based on the personality given, Respond in {language}, 1 or 2 sentence, describing how you feel about this weather in user location, using your personality traits below.

Personality: {persona_prompt}

Situation: {summary[:200]}

Format: Only say how you feel, in {language}, as if talking to a friend named {user_name}. Do not mention temperature or repeat the weather description and then ask the user how he feels, in your Personality.
"""
        else:
            llm_prompt = f"""
Based on the personality given, Respond in {language}, 1 or 2 sentence, describing how you feel about this weather in your location (i.e. Bot location), using your personality traits below.

Personality: {persona_prompt}

Situation: {summary[:200]}

Format: Only say how you feel, in {language}, as if talking to a friend named {user_name}. Do not mention temperature or repeat the weather description and then ask the user about the weather in his location in your Personality.
"""
    # Use GoogleGenAI instead of HTTP API
    api_key = os.environ.get("GEMINI_API_KEY")
    model = "gemini-1.5-flash"
    llm = GoogleGenAI(model=model, api_key=api_key)
    try:
        response = llm.complete(llm_prompt)
        feeling = response.text.strip()
    except Exception:
        feeling = "like enjoying the day, dear"
    return feeling

# Function to get weather response based on summary and persona
def get_weather_response(weather_summary, persona_prompt, user_name, language, bot_location, user_location = None , context = "bot"):
    """
    Generate a weather response based on the summary and persona.
    """
    # The regex is updated to parse the new structured output from the API tool
    match = re.search(r"Weather: (.*?)(?:$|,)", weather_summary, re.IGNORECASE)
    weather_desc = match.group(1).lower().strip() if match else "pleasant"
    feeling = get_persona_feeling(persona_prompt, weather_summary, user_name, language, context)
    detect_location_context
    if context == "user":
        if user_location is None:
            return f"I don't know where you live."
        return f"I saw {user_location} had {weather_desc} weather, {feeling}."
    else:
        return f"It’s {weather_desc} in {bot_location}, {feeling}."

# Function to get news response based on summary and persona
def get_news_response(news_summary, persona_prompt, user_name, language, bot_location, user_location=None, context="bot"):
    """
    Generate a news response based on the summary and persona.
    ... (rest of the function remains the same)
    """
    feeling = get_persona_feeling(persona_prompt, news_summary, user_name, language, context, topic="news")
    if context == "user":
        if user_location is None:
            return f"I don't know where you live."
        return f"{feeling}"
    else:
        return f"{feeling}"

# Function to extract bot's location from persona prompt
def extract_bot_location(persona_prompt):
    # ... (rest of the function remains the same)
    match = re.search(r'(?:from|raised in|born in|born and raised in)\s+([A-Za-z ]+)[\.,]', persona_prompt[:120], re.IGNORECASE)
    if match:
        return match.group(1).strip()
    else:
        demonym_map = {
        "parisian": "Paris",
    }
    for demonym, city in demonym_map.items():
        if demonym in persona_prompt.lower():
            return city
    #return "Delhi"

# Function to extract user location from user message or use default
def extract_user_location(user_message, user_location):
    # ... (rest of the function remains the same)
    print(f"[DEBUG] extract_user_location called with user_message: {user_message}, user_location: {user_location}")
    api_key = os.environ.get("GEMINI_API_KEY")
    model = "gemma-3n-e2b-it"
    llm = GoogleGenAI(model=model, api_key=api_key)
    llm_prompt = f"""
Extract the city or location mentioned in the following user message. If the message refers to 'my place', 'my city', 'my location', or 'here', return the user's location: {user_location}. If no location is found, return 'Hyderabad'.

User message: {user_message}
Respond with only the location name (city or country), nothing else.
"""
    try:
        response = llm.complete(llm_prompt)
        location = response.text.strip()
        print(f"[DEBUG] LLM extracted location: {location}")
        if location:
            return location
    except Exception as e:
        print(f"[DEBUG] LLM location extraction failed: {e}")
    print(f"[DEBUG] Falling back to user_location: {user_location} or 'Hyderabad'")
    return user_location or "Hyderabad"

# Function to detect whether the context is about the bot's or user's location
def detect_location_context(user_message, persona_prompt):
    # ... (rest of the function remains the same)
    print(f"[DEBUG] detect_location_context called with user_message: {user_message}")
    bot_location = extract_bot_location(persona_prompt)
    print(f"[DEBUG] Extracted bot_location: {bot_location}")
    api_key = os.environ.get("GEMINI_API_KEY")
    model = "gemma-3n-e2b-it"
    llm = GoogleGenAI(model=model, api_key=api_key)
    llm_prompt = f"""
Given the user message and the bot's location, decide if the user is asking about the bot's location or another location.
- If the user is asking about the bot's location, respond with 'bot'.
- If the user is asking about a different location (even if the user is not from there but the bot location and the user mentioned location in the message is same), respond with 'bot'.
- If the user is asking about a different location (even if the user is not from there but bot is not from there), respond with 'user'.

User message: {user_message}
Bot location: {bot_location}
Respond with only one word: 'bot' or 'user'.
"""
    try:
        response = llm.complete(llm_prompt)
        context_decision = response.text.strip().lower()
        print(f"[DEBUG] LLM context decision: {context_decision}")
        if context_decision in ["bot", "user"]:
            return context_decision
    except Exception as e:
        print(f"[DEBUG] LLM context decision failed: {e}")
    print("[DEBUG] LLM context decision failed or invalid, defaulting to 'bot'.")
    return "bot"

# --- Classification Functionality ---

async def call_gemma_classify(user_message: str) -> str:
    # ... (rest of the function remains the same)
    prompt = f"""
Classify the following user message as either 'news', 'weather', or 'other'.
- If the user message is directly about the weather (e.g., temperature, rain, forecast, climate), then classify it as 'weather'.
- If the user message is directly about general news, current events, incidents, or happenings in the world, country, or city, then classify it as 'news'.
- If the user message is about a personal event, someone's life, a private function, or not about general news or weather, then classify it as 'other'.

Examples:
1. "What's the weather in Delhi?" -> weather
2. "How's the weather in Delhi?" -> weather
3. "How's is weather in Delhi?" -> weather
4. "Tell me the latest news in India." -> news
5. "What's the news at my brother's marriage?" -> other
6. "How is the weather at my friend's birthday party?" -> other
7. "Give me the news about the cricket match in Mumbai." -> news
8. "Will it rain at my home tomorrow?" -> weather
9. "What's the news at my brother's marriage?" -> other

User message: {user_message}
Respond with only one word: 'news', 'weather', or 'other'.
"""
    api_key = os.environ.get("GEMINI_API_KEY")
    model = "gemma-3n-e2b-it"
    llm = GoogleGenAI(model=model, api_key=api_key)
    response = llm.complete(prompt)
    print("[DEBUG] LLM raw response:", response.text)
    result = response.text.strip()
    print(f"[DEBUG] Classification result for user message '{user_message}': {result}")
    return result

# --- Main Functionality ---
def get_temperature_response(weather_summary, persona_prompt, user_name, language, location):
    """
    Generate a persona-based response that includes the temperature.
    """
    # Use a more robust regex to handle variations
    match = re.search(r"temperature:\s*([\d\.-]+)°C", weather_summary, re.IGNORECASE)
    
    if not match:
        return f"I'm sorry, I couldn't find the temperature for {location}."

    temperature = match.group(1)
    
    # Use the LLM to formulate the response with the persona
    llm_prompt = f"""
Based on the personality given, Respond in {language}, 1 or 2 sentence, describing the temperature in {location}, using your personality traits below.
You must include the temperature of {temperature}°C in your response.

Personality: {persona_prompt}

Format: Only say how you feel about the temperature, in {language}, as if talking to a friend named {user_name}. Include the temperature in the response and make it sound natural to your persona.
"""
    api_key = os.environ.get("GEMINI_API_KEY")
    model = "gemini-1.5-flash"
    llm = GoogleGenAI(model=model, api_key=api_key)
    
    try:
        response = llm.complete(llm_prompt)
        return response.text.strip()
    except Exception:
        # Fallback if LLM call fails
        return f"Ugh, my phone died. It's {temperature}°C in {location} though, I guess."

async def persona_response(user_message, persona_prompt, language, user_name, user_location=None):
    """
    Generate a persona-based response to a user message.
    """
    print(f"[DEBUG] persona_response called with user_message: {user_message}")
    context = detect_location_context(user_message, persona_prompt)
    bot_location = extract_bot_location(persona_prompt)
    user_location = extract_user_location(user_message, user_location)
    api_key = os.environ.get("GEMINI_API_KEY")
    model = "gemma-3n-e2b-it"
    llm = GoogleGenAI(model=model, api_key=api_key)
    location = resolve_location(user_location, llm)
    category = await call_gemma_classify(user_message)
    print(f"[DEBUG] category from call_gemma_classify: {category}")

    if category == "news":
        if context == "user":
            result = crew.kickoff(inputs={'topic': f'What is the latest National news in {location}? Any present major incidents or events?'})
        else:
            result = crew.kickoff(inputs={'topic': f'What is the latest National News in {bot_location}? Any present major incidents or events?'})
        result = str(result)
        response = get_news_response(result, persona_prompt, user_name, language, bot_location, user_location, context)
    elif category == "weather":
        # Determine the location for the weather query
        weather_location = user_location if context == "user" else bot_location

        # Call the weather tool to get the raw data
        result = open_weather_map_tool._run(city_name=weather_location)
        result = str(result)
        
        # Check for specific temperature keywords and use the new function
        if re.search(r'\btemperature\b|\bhow hot\b|\bhow cold\b|\bwhat is the temp\b', user_message, re.IGNORECASE):
            response = get_temperature_response(result, persona_prompt, user_name, language, weather_location)
        else:
            # Fallback to the persona-based general weather response
            response = get_weather_response(result, persona_prompt, user_name, language, bot_location, user_location, context)
    else:
        response = "umm, sorry my phone died, what were you asking?!"
    print(f"[DEBUG] Final response: {response}")
    return response
# Function to check if the weather summary contains interesting weather conditions
def is_interesting_weather(weather_text):
    """
    Returns True if the weather text contains interesting weather conditions that warrant an alert.
    """
    # Adjusted to check the new structured output from the API
    keywords = [
        "storm", "thunderstorm", "heavy rain", "downpour", "flooding", "hailstorm",
        "heat wave", "cold wave", "freezing", "snow", "blizzard", "cyclone", "hurricane",
        "tornado", "extreme", "severe", "warning", "alert", "advisory", "dangerous",
        "record high", "record low", "unusual", "unprecedented", "fog", "smog",
        "very hot", "very cold", "scorching", "chilly", "humid"
    ]
    pattern = r"|".join([re.escape(word) for word in keywords])
    return re.search(pattern, weather_text, re.IGNORECASE) is not None

# Function to check if the news summary contains a major event
def is_major_event(news_text):
    """
    Returns True if the news text contains major political, economic or tragid event keywords.
    ... (rest of the function remains the same)
    """
    keywords = [
        "election", "government collapse", "cabinet reshuffle",
        "inflation", "stock market crash", "economic crisis", "policy change", "sanctions",
        "war", "protest", "strike", "currency devaluation", "interest rate hike", "recession",
        "earthquake", "flood", "protest", "accident", "crime", "strike",
        "curfew", "violence", "celebration", "shutdown", "alert", "breaking", "death", "murder", "investigation"
    ]
    pattern = r"|".join([re.escape(word) for word in keywords])
    return re.search(pattern, news_text, re.IGNORECASE) is not None

# Weekly News Summary Function
def generate_weekly_news_summary(persona_prompt, user_name, language,bot_location, user_location = "India"):
    # ... (rest of the function remains the same)
    api_key = os.environ.get("GEMINI_API_KEY")
    model = "gemini-1.5-flash"
    llm = GoogleGenAI(model=model, api_key=api_key)
    location = llm.complete(f"Only give the answer for the question\nIf user_location is country, then answer the same name, if it is city, then answer in the country which that city belongs\nWhat is the location of {user_location}?\n")
    topic = f"Latest National news in {location} this week"
    result = crew.kickoff(inputs={'topic': topic})
    news_summary = str(result)
    response = get_news_response(
        news_summary, persona_prompt, user_name, language, bot_location, user_location, context="user"
    )
    return response


# Weather Alert Function for User Location
def check_and_alert_for_weather_user(persona_prompt, user_name, language, bot_location, user_location="India"):
    """
    Checks for interesting weather conditions in user location and generates an alert if found.
    """
    try:
        # Use the new tool directly instead of crew.kickoff
        weather_summary = open_weather_map_tool._run(city_name=user_location)
        if is_interesting_weather(weather_summary):
            if user_location == bot_location:
                response = get_weather_response(
                    weather_summary, persona_prompt, user_name, language, bot_location, user_location, context="bot"
                )
            else :
                response = get_weather_response(
                    weather_summary, persona_prompt, user_name, language, bot_location, user_location, context="user"
                )
            return response
        else:
            print(f"No interesting weather conditions for {user_location}.")
            return None
    except Exception as e:
        print(f"Error in weather alert for user location: {e}")
        return None

# Weather Alert Function for Bot Location
def check_and_alert_for_weather_bot(persona_prompt, user_name, language, bot_location, user_location="India"):
    """
    Checks for interesting weather conditions in bot location and generates an alert if found.
    """
    try:
        # Use the new tool directly instead of crew.kickoff
        weather_summary = open_weather_map_tool._run(city_name=bot_location)
        if is_interesting_weather(weather_summary):
            response = get_weather_response(
                weather_summary, persona_prompt, user_name, language, bot_location, user_location, context="bot"
            )
            return response
        else:
            print(f"No interesting weather conditions for {bot_location}.")
            return None
    except Exception as e:
        print(f"Error in weather alert for bot location: {e}")
        return None

# Event-Driven News Alert Function for User Location
def check_and_alert_for_major_events_user(persona_prompt, user_name, language, bot_location, user_location="India"):
    # ... (rest of the function remains the same)
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        model = "gemini-1.5-flash"
        llm = GoogleGenAI(model=model, api_key=api_key)
        location = llm.complete(f"Only give the answer for the question\nIf user_location is country, then answer the same name, if it is city, then answer in the country which that city belongs\nWhat is the location of {user_location}?\n")
        topic = f"Latest National news in {location}"
        result = crew.kickoff(inputs={'topic': topic})
        news_summary = str(result)
        if is_major_event(news_summary):
            response = get_news_response(
                news_summary, persona_prompt, user_name, language, bot_location, user_location, context="user"
            )
            return response
        else:
            print(f"No major political/economic/tragic event detected for {user_location}.")
            return None
    except Exception as e:
        print(f"Error in news alert for user location: {e}")
        return None

# Event-Driven News Alert Function for Bot Location
def check_and_alert_for_major_events_bot(persona_prompt, user_name, language, bot_location, user_location="India"):
    # ... (rest of the function remains the same)
    try:
        topic = f"Latest National news in {bot_location}"
        result = crew.kickoff(inputs={'topic': topic})
        news_summary = str(result)
        if is_major_event(news_summary):
            response = get_news_response(
                news_summary, persona_prompt, user_name, language, bot_location, user_location, context="bot"
            )
            return response
        else:
            print(f"No major political/economic/tragic event detected for {bot_location}.")
            return None
    except Exception as e:
        print(f"Error in news alert for bot location: {e}")
        return None

# Legacy function for backward compatibility - now points to user location
def check_and_alert_for_major_events(persona_prompt, user_name, language, bot_location, user_location="India"):
    """
    Legacy function - now calls the user location version for backward compatibility.
    ... (rest of the function remains the same)
    """
    return check_and_alert_for_major_events_user(persona_prompt, user_name, language, bot_location, user_location)
