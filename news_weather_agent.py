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
dotenv_path = Path('.env')
load_dotenv(dotenv_path=dotenv_path)

# Defining the Google Search function
def google_search(query, api_key=os.environ.get("GOOGLE_SEARCH_API_KEY"), cse_id=os.environ.get("GOOGLE_CSE_ID"), **kwargs):
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=query, cx=cse_id, **kwargs).execute()
    return res.get('items', [])

# Function to fetch page content
def fetch_page_content(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text(separator="\n", strip=True)
        return text[150:2000] + "..." if len(text) > 1850 else text
    except Exception as e:
        return f"Error fetching page: {e}"

# Custom tool for Google Search and Page Fetch
class GoogleSearchAndFetchTool(BaseTool):
    name: str = "Google Search with Page Fetch"
    description: str = "Performs a Google search and fetches a summary of the top 3 result pages."

    def _run(self, query: str) -> str:
        results = google_search(query)
        output = []
        for item in results[:2]:
            title = item.get("title")
            link = item.get("link")
            content = fetch_page_content(link)
            output.append(f"{title}\n{link}\n{content}\n{'-'*40}")
        return "\n".join(output)

# Initialize the custom tool
custom_google_search_tool = GoogleSearchAndFetchTool()

# Define agents
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
    allow_delegation=True
)

news_writer = Agent(
    role='Conversational Communicator',
    goal="Present information in a friendly, clear, and engaging way, making it easy for anyone to understand and act on answers about {topic}.",
    verbose=True,
    memory=False,  # Set memory to False for quick responses
    backstory=(
        "You specialize in turning facts and recommendations into helpful, easy-to-read responses. "
        "Your writing is approachable and practical, perfect for answering real-world questions people care about."
    ),
    #tools=[tool],
    llm=llm,
    allow_delegation=False
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
    tools=[custom_google_search_tool], # Use the initialized custom tool
    agent=news_researcher,
)

write_task = Task(
    description=(
    "Based on the research, write a clear and friendly response about {topic}. "
    "Make the answer easy to understand, practical, and directly useful for the question asked. "
    "If relevant, include tips, examples, or next steps."
    ),
    expected_output=(
    "A short, well-structured answer (2-3 paragraphs) in markdown format, "
    "providing practical and up-to-date information about {topic}."
    ),
    #tools=[tool],
    agent=news_writer,
    async_execution=False,
    #output_file='new-blog-post.md'
)

# Initialize the crew
crew = Crew(
    agents=[news_researcher, news_writer],
    tasks=[research_task, write_task],
    process=Process.sequential,
)

def get_persona_feeling(persona_prompt, summary, user_name, language, context="bot", topic="weather"):
    if topic == "news":
        if context == "user":
            llm_prompt = f"""
Based on the personality given, Respond in {language}, 1 or 2 sentence, describing how you feel about this news (only one news) in user location, using your personality traits below.

Personality: {persona_prompt}

Situation: {summary}

Format: Only say how you feel, in {language}, as if talking to a friend named {user_name}. Do not repeat the news description and then ask the user how he feels, in your Personality.
"""
        else:
            llm_prompt = f"""
Based on the personality given, Respond in {language}, 1 or 2 sentence, describing how you feel about this news (only one news) in your location (i.e. Bot location), using your personality traits below.

Personality: {persona_prompt}

Situation: {summary}

Format: Only say how you feel, in {language}, as if talking to a friend named {user_name}. Just explain the incident in brief in your personality to explain the user, Then ask the user about the news in his location in your Personality.
"""
    else:  # weather
        if context == "user":
            llm_prompt = f"""
Based on the personality given, Respond in {language}, 1 or 2 sentence, describing how you feel about this weather in user location, using your personality traits below.

Personality: {persona_prompt}

Situation: {summary}

Format: Only say how you feel, in {language}, as if talking to a friend named {user_name}. Do not mention temperature or repeat the weather description and then ask the user how he feels, in your Personality.
"""
        else:
            llm_prompt = f"""
Based on the personality given, Respond in {language}, 1 or 2 sentence, describing how you feel about this weather in your location (i.e. Bot location), using your personality traits below.

Personality: {persona_prompt}

Situation: {summary}

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
    match = re.search(r"(sunny|cloudy|rainy|humid|very warm|pleasant|chilly|hot|cold|breezy|stormy|clear|overcast)", weather_summary, re.IGNORECASE)
    weather_desc = match.group(1).lower() if match else "pleasant"
    feeling = get_persona_feeling(persona_prompt, weather_summary, user_name, language, context)
    if context == "user":
        if user_location is None:
            return f"I don't know where you live."
        return f"I saw {user_location} had {weather_desc} weather in the news, {feeling}."
    else:
        return f"It’s {weather_desc} in {bot_location}, {feeling}."
    
# Function to get news response based on summary and persona
def get_news_response(news_summary, persona_prompt, user_name, language, bot_location, user_location=None, context="bot"):
    # Extract a main event/incident keyword for variety
    match = re.search(r"(earthquake|flood|protest|accident|crime|festival|strike|curfew|violence|celebration|shutdown|alert|breaking|trending|election|government collapse|prime minister|president|parliament|cabinet reshuffle|budget|inflation|stock market crash|economic crisis|policy change|sanctions|war|conflict|protest|strike|currency devaluation|interest rate hike|GDP|recession)", news_summary, re.IGNORECASE)
    news_desc = match.group(1).lower() if match else "something interesting"
    feeling = get_persona_feeling(persona_prompt, news_summary, user_name, language, context, topic="news")
    if context == "user":
        if user_location is None:
            return f"I don't know where you live."
        return f"I saw in the news that {user_location} had {news_desc}, {feeling}"
    else:
        #return f"There is a {news_desc} in {bot_location}, {feeling}"
        return f"{feeling}"

# Function to extract bot's location from persona prompt
def extract_bot_location(persona_prompt):
    match = re.search(r'(?:from|raised in)\s+([A-Za-z ]+)[\.,]', persona_prompt)
    return match.group(1).strip() if match else "Delhi"

# Function to extract user location from user message or use default
def extract_user_location(user_message, user_location):
    # Try to extract a city from the message
    match = re.search(r"(?:in|at|of)\s+([A-Za-z ]+)", user_message, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # If user says "my place", "here", etc.
    if re.search(r"my place|my city|my location|here", user_message, re.IGNORECASE):
        return user_location
    return "Hyderabad"

# Function to detect whether the context is about the bot's or user's location
def detect_location_context(user_message, persona_prompt):
    # If user asks about "your place", "where you are", etc. → bot's location
    if re.search(r"(your place|where you are|your city|your location|at your end|there)", user_message, re.IGNORECASE):
        return "bot"
    # If user asks about "my place", "here", etc. → user's location
    if re.search(r"(my place|my city|my location|here)", user_message, re.IGNORECASE):
        return "user"
    # If user mentions a city directly
    location = re.search(r"weather (in|at|of) ([A-Za-z ]+)", user_message, re.IGNORECASE)
    if location:
        location = location.group(2).strip().lower()
        bot_location = extract_bot_location(persona_prompt).lower()# if persona_prompt else "delhi"
        if location == bot_location:
            return "bot"
        else:
            return "user"
    return "bot"  # Default to bot's location

# Function to check if the user message is a news query
def is_news_query(user_message):
    news_keywords = [
        "news", "heard", "see", "read", "latest", "update", "incident", "happening",
        "earthquake", "flood", "protest", "accident", "crime", "festival", "strike",
        "curfew", "violence", "celebration", "shutdown", "alert", "breaking", "trending"
    ]
    pattern = r"|".join([re.escape(word) for word in news_keywords])
    return re.search(pattern, user_message, re.IGNORECASE) is not None

# Main function to handle user message and persona response
async def persona_response(user_message, persona_prompt, language, user_name, user_location=None):
    context = detect_location_context(user_message, persona_prompt)
    bot_location = extract_bot_location(persona_prompt)
    if user_location is None:
        user_location = extract_user_location(user_message, user_location)
    if is_news_query(user_message):
        # News flow
        if context == "user":
            result = crew.kickoff(inputs={'topic': f'What is the latest National news in {user_location}? Any present major incidents or events?'})
        else:
            result = crew.kickoff(inputs={'topic': f'What is the latest National News in {bot_location}? Any present major incidents or events?'})
        result = str(result)
        response = get_news_response(result, persona_prompt, user_name, language, bot_location, user_location, context)
    else:
        # Weather flow
        if context == "user":
            result = crew.kickoff(inputs={'topic': f'What is the weather in {user_location}?'})
        else:
            result = crew.kickoff(inputs={'topic': f'What is the weather in {bot_location}?'})
        result = str(result)
        response = get_weather_response(result, persona_prompt, user_name, language, bot_location, user_location, context)
    #print(response)
    return response

# Function to check if the news summary contains a major event
def is_major_event(news_text):
    """
    Returns True if the news text contains major political, economic or tragid event keywords.
    """
    keywords = [
        "election", "government collapse", "prime minister", "president", "parliament", "cabinet reshuffle",
        "budget", "inflation", "stock market crash", "economic crisis", "policy change", "sanctions",
        "war", "conflict", "protest", "strike", "currency devaluation", "interest rate hike", "GDP", "recession",
        "earthquake", "flood", "protest", "accident", "crime", "festival", "strike",
        "curfew", "violence", "celebration", "shutdown", "alert", "breaking", "death"
    ]
    pattern = r"|".join([re.escape(word) for word in keywords])
    return re.search(pattern, news_text, re.IGNORECASE) is not None

# Weekly News Summary Function
def generate_weekly_news_summary(persona_prompt, user_name, language,bot_location, user_location = "India"):
    """
    Generates a weekly news summary for the given user location.
    """
    topic = f"Major political, economic and tragid news in {user_location} country this week"
    result = crew.kickoff(inputs={'topic': topic})
    news_summary = str(result)
    # Optionally, filter for major events
    if is_major_event(news_summary):
        print("🚨 Major Event Detected! 🚨")
    response = get_news_response(
        news_summary, persona_prompt, user_name, language, bot_location, user_location, context="user"
    )
    #print(f"Weekly News Summary for {user_location}:\n{response}")
    return response


#  Event-Driven News Alert Function
def check_and_alert_for_major_events(persona_prompt, user_name, language,bot_location, user_location = "India"):
    """
    Checks for major political/economic/tragid events and generates an alert if found.
    """
    topic = f"Latest breaking political, economic or tragid news in {user_location} country"
    result = crew.kickoff(inputs={'topic': topic})
    news_summary = str(result)
    if is_major_event(news_summary):
        response = get_news_response(
            news_summary, persona_prompt, user_name, language, bot_location, user_location, context="user"
        )
        #print(f"🚨 Major Event Alert for {user_location}:\n{response}")
        # Here you can send the response via email, push notification, etc.
        return response
    else:
        print("No major political/economic/tragid event detected.")
        return None