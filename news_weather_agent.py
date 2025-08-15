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

# Load environment variables from .env file
dotenv_path = Path('.env')
load_dotenv(dotenv_path=dotenv_path)
import asyncio

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

# Initialize the custom tool
custom_google_search_tool = GoogleSearchAndFetchTool()

# --- Define Agents ---

# Define the news researcher agent
llm = 'gemini/gemini-1.5-flash'

news_researcher = Agent(
    role='Situational Expert',
    goal='Provide accurate, up-to-date, and practical answers to any real-world question, such as recommendations or current conditions about {topic}.Ignore the top hyperlinks in the news websites, because they are irrelevant one. Try to be as quick as possible',
    verbose=True,
    memory=False,  # Set memory to False for quick responses
    backstory=(
        "You are a resourceful expert, skilled at quickly gathering and analyzing the latest information. "
        "Whether it's about places, events, or current conditions, you deliver clear, reliable, and actionable insights for any question."
    ),
    tools=[custom_google_search_tool],  # Use the initialized custom tool
    llm=llm,
#    allow_delegation=True
)

#news_writer = Agent(
#    role='Conversational Communicator',
#    goal="Present information in a friendly, clear, and engaging way, making it easy for anyone to understand and act on answers about {topic}.",
#    verbose=True,
#    memory=False,  # Set memory to False for quick responses
#    backstory=(
#        "You specialize in turning facts and recommendations into helpful, easy-to-read responses. "
#        "Your writing is approachable and practical, perfect for answering real-world questions people care about."
#    ),
#    #tools=[tool],
#    llm=llm,
#    allow_delegation=False
#)

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
    tools=[custom_google_search_tool],  # Use the initialized custom tool
    agent=news_researcher,
)

#write_task = Task(
#    description=(
#    "Based on the research, write a clear and friendly response about {topic}. "
#    "Make the answer easy to understand, practical, and directly useful for the question asked. "
#    "If relevant, include tips, examples, or next steps."
#    ),
#    expected_output=(
#    "A short, well-structured answer (2-3 paragraphs) in markdown format, "
#    "providing practical and up-to-date information about {topic}."
#    ),
    #tools=[tool],
#    agent=news_writer,
#    async_execution=False,
#    #output_file='new-blog-post.md'
#)

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

    Args:
        user_location (str): The user's location (city or country).
        llm: The language model instance.

    Returns:
        str: Resolved location.
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

    Args:
        persona_prompt (str): The persona's prompt.
        summary (str): The news or weather summary.
        user_name (str): The user's name.
        language (str): The language for the response.
        context (str, optional): The context of the response ('bot' or 'user').
        topic (str, optional): The topic of the summary ('news' or 'weather').

    Returns:
        str: The persona's feeling or emotional response.
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

    Args:
        weather_summary (str): The weather summary.
        persona_prompt (str): The persona's prompt.
        user_name (str): The user's name.
        language (str): The language for the response.
        bot_location (str): The bot's location.
        user_location (str, optional): The user's location.
        context (str, optional): The context of the response ('bot' or 'user').

    Returns:
        str: The generated weather response.
    """
    match = re.search(r"(sunny|cloudy|rainy|humid|very warm|pleasant|chilly|hot|cold|breezy|stormy|clear|overcast)", weather_summary, re.IGNORECASE)
    weather_desc = match.group(1).lower() if match else "pleasant"
    feeling = get_persona_feeling(persona_prompt, weather_summary, user_name, language, context)
    detect_location_context
    if context == "user":
        if user_location is None:
            return f"I don't know where you live."
        return f"I saw {user_location} had {weather_desc} weather in the news, {feeling}."
    else:
        return f"It’s {weather_desc} in {bot_location}, {feeling}."
    
# Function to get news response based on summary and persona
def get_news_response(news_summary, persona_prompt, user_name, language, bot_location, user_location=None, context="bot"):
    """
    Generate a news response based on the summary and persona.

    Args:
        news_summary (str): The news summary.
        persona_prompt (str): The persona's prompt.
        user_name (str): The user's name.
        language (str): The language for the response.
        bot_location (str): The bot's location.
        user_location (str, optional): The user's location.
        context (str, optional): The context of the response ('bot' or 'user').

    Returns:
        str: The generated news response.
    """
    # Extract a main event/incident keyword for variety
    #match = re.search(r"(earthquake|flood|protest|accident|crime|festival|strike|curfew|violence|celebration|shutdown|alert|breaking|trending|election|government collapse|cabinet reshuffle|inflation|stock market crash|economic crisis|policy change|war|currency devaluation|interest rate hike|recession)", news_summary, re.IGNORECASE)
    #news_desc = match.group(1).lower() if match else "something interesting"
    feeling = get_persona_feeling(persona_prompt, news_summary, user_name, language, context, topic="news")
    if context == "user":
        if user_location is None:
            return f"I don't know where you live."
        return f"{feeling}"
    else:
        #return f"There is a {news_desc} in {bot_location}, {feeling}"
        return f"{feeling}"

# Function to extract bot's location from persona prompt
def extract_bot_location(persona_prompt):
    """
    Extract the bot's location from the persona prompt.

    Args:
        persona_prompt (str): The persona's prompt.

    Returns:
        str: The extracted bot location.
    """
    match = re.search(r'(?:from|raised in|born in|born and raised in)\s+([A-Za-z ]+)[\.,]', persona_prompt[:120], re.IGNORECASE)
    if match:
        return match.group(1).strip()
    else:
        demonym_map = {
        "parisian": "Paris",
        # Add more as needed
    }
    for demonym, city in demonym_map.items():
        if demonym in persona_prompt.lower():
            return city
    #return "Delhi"
# Function to extract user location from user message or use default
def extract_user_location(user_message, user_location):
    """
    Extract the user's location from the user message or use the default location.

    Args:
        user_message (str): The user message.
        user_location (str): The default user location.

    Returns:
        str: The extracted or default user location.
    """
    print(f"[DEBUG] extract_user_location called with user_message: {user_message}, user_location: {user_location}")
    api_key = os.environ.get("GEMINI_API_KEY")
    model = "gemma-3n-e2b-it"
    llm = GoogleGenAI(model=model, api_key=api_key)
    llm_prompt = f"""
Extract the city or location mentioned in the following user message. If the message refers to 'my place', 'my city', 'my location', or 'here', return the user's location: {user_location}. If no location is found, return 'Hyderabad'.

User message: {user_message}
Respond with only the location name (city or country), nothing else.
"""
    #print(f"[DEBUG] LLM prompt for location extraction:\n{llm_prompt}")
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
    """
    Detect the context of the location in the user message.

    Args:
        user_message (str): The user message.
        persona_prompt (str): The persona's prompt.

    Returns:
        str: The detected context ('bot' or 'user').
    """
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
    """
    Calls the gemma-3n-e2b-it model to classify the user message as 'news', 'weather', or 'other'.

    Args:
        user_message (str): The user message.

    Returns:
        str: The classification result ('news', 'weather', or 'other').
    """
    # Improved prompt with explicit instructions and examples
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
    #print("[DEBUG] Classification prompt sent to gemma-3n-e2b-it:\n", prompt)
    api_key = os.environ.get("GEMINI_API_KEY")
    model = "gemma-3n-e2b-it"
    llm = GoogleGenAI(model=model, api_key=api_key)
    response = llm.complete(prompt)
    print("[DEBUG] LLM raw response:", response.text)
    result = response.text.strip()
    print(f"[DEBUG] Classification result for user message '{user_message}': {result}")
    return result

# --- Main Functionality ---

async def persona_response(user_message, persona_prompt, language, user_name, user_location=None):
    """
    Generate a persona-based response to a user message.

    Args:
        user_message (str): The user's message.
        persona_prompt (str): The persona's prompt.
        language (str): The language for the response.
        user_name (str): The user's name.
        user_location (str, optional): The user's location.

    Returns:
        str: The generated response.
    """
    print(f"[DEBUG] persona_response called with user_message: {user_message}")
    context = detect_location_context(user_message, persona_prompt)
    bot_location = extract_bot_location(persona_prompt)
    user_location = extract_user_location(user_message, user_location)
    # Initialize LLM once
    api_key = os.environ.get("GEMINI_API_KEY")
    model = "gemma-3n-e2b-it"
    llm = GoogleGenAI(model=model, api_key=api_key)
    location = resolve_location(user_location, llm)
    # Use LLM to classify the user message
    category = await call_gemma_classify(user_message)
    print(f"[DEBUG] category from call_gemma_classify: {category}")
    if category == "news":
        # News flow
        if context == "user":
            result = crew.kickoff(inputs={'topic': f'What is the latest National news in {location}? Any present major incidents or events?'})
        else:
            result = crew.kickoff(inputs={'topic': f'What is the latest National News in {bot_location}? Any present major incidents or events?'})
        result = str(result)
        response = get_news_response(result, persona_prompt, user_name, language, bot_location, user_location, context)
    elif category == "weather":
        # Weather flow
        if context == "user":
            result = crew.kickoff(inputs={'topic': f'What is the weather in {user_location}?'})
        else:
            result = crew.kickoff(inputs={'topic': f'What is the weather in {bot_location}?'})
        result = str(result)
        response = get_weather_response(result, persona_prompt, user_name, language, bot_location, user_location, context)
    else:
        response = "umm, sorry my phone died, what were you asking?!"
    print(f"[DEBUG] Final response: {response}")
    return response

# --- Monitoring and Alerting Functions ---

# Function to check if the weather summary contains interesting weather conditions
def is_interesting_weather(weather_text):
    """
    Returns True if the weather text contains interesting weather conditions that warrant an alert.

    Args:
        weather_text (str): The weather text.

    Returns:
        bool: True if interesting weather conditions are present, False otherwise.
    """
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

    Args:
        news_text (str): The news text.

    Returns:
        bool: True if a major event is detected, False otherwise.
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
    """
    Generates a weekly news summary for the given user location.

    Args:
        persona_prompt (str): The persona's prompt.
        user_name (str): The user's name.
        language (str): The language for the response.
        bot_location (str): The bot's location.
        user_location (str, optional): The user's location.

    Returns:
        str: The generated weekly news summary.
    """
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
    #print(f"Weekly News Summary for {user_location}:\n{response}")
    return response


# Weather Alert Function for User Location
def check_and_alert_for_weather_user(persona_prompt, user_name, language, bot_location, user_location="India"):
    """
    Checks for interesting weather conditions in user location and generates an alert if found.

    Args:
        persona_prompt (str): The persona's prompt.
        user_name (str): The user's name.
        language (str): The language for the response.
        bot_location (str): The bot's location.
        user_location (str, optional): The user's location.

    Returns:
        str or None: The weather alert response or None if no interesting conditions are found.
    """
    try:
        topic = f"Current weather conditions and forecast for {user_location}"
        result = crew.kickoff(inputs={'topic': topic})
        weather_summary = str(result)
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

    Args:
        persona_prompt (str): The persona's prompt.
        user_name (str): The user's name.
        language (str): The language for the response.
        bot_location (str): The bot's location.
        user_location (str, optional): The user's location.

    Returns:
        str or None: The weather alert response or None if no interesting conditions are found.
    """
    try:
        topic = f"Current weather conditions and forecast for {bot_location}"
        result = crew.kickoff(inputs={'topic': topic})
        weather_summary = str(result)
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

#  Event-Driven News Alert Function for User Location
def check_and_alert_for_major_events_user(persona_prompt, user_name, language, bot_location, user_location="India"):
    """
    Checks for major political/economic/tragic events in user location and generates an alert if found.

    Args:
        persona_prompt (str): The persona's prompt.
        user_name (str): The user's name.
        language (str): The language for the response.
        bot_location (str): The bot's location.
        user_location (str, optional): The user's location.

    Returns:
        str or None: The news alert response or None if no major events are found.
    """
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

#  Event-Driven News Alert Function for Bot Location
def check_and_alert_for_major_events_bot(persona_prompt, user_name, language, bot_location, user_location="India"):
    """
    Checks for major political/economic/tragic events in bot location and generates an alert if found.

    Args:
        persona_prompt (str): The persona's prompt.
        user_name (str): The user's name.
        language (str): The language for the response.
        bot_location (str): The bot's location.
        user_location (str, optional): The user's location.

    Returns:
        str or None: The news alert response or None if no major events are found.
    """
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

    Args:
        persona_prompt (str): The persona's prompt.
        user_name (str): The user's name.
        language (str): The language for the response.
        bot_location (str): The bot's location.
        user_location (str, optional): The user's location.

    Returns:
        str or None: The news alert response or None if no major events are found.
    """
    return check_and_alert_for_major_events_user(persona_prompt, user_name, language, bot_location, user_location)
