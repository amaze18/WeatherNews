# WeatherNews

WeatherNews is a small FastAPI-based service that uses CrewAI agents and Google GenAI to fetch and summarize weather and news information, then deliver persona-driven responses. The project includes scheduled proactive alerts (weekly summary + daily proactive checks) and Supabase integration for storing messages and user/bot metadata.

## What this repo contains

- `main.py` — FastAPI application, scheduling entrypoints, Supabase helpers, and API endpoints to trigger tasks.
- `news_weather_agent.py` — CrewAI-based agent logic for searching, classifying, and generating persona responses (news vs weather flows).
- `bot_prompt.py` — Templates for bot persona prompts (used to generate persona-specific replies).
- `PROACTIVE_ALERTS_SUMMARY.md` — Implementation summary for proactive alerts (updated to reflect current code).
- `requirements.txt` — Python dependencies (used to install needed packages).

## Key features

- Persona-driven responses: The agent formats replies using persona prompts from `bot_prompt.py`.
- Scheduled weekly news summaries: runs every 7 days using FastAPI `repeat_every`.
- APScheduler-driven proactive jobs: runs weather/news checks at fixed daily hours (08:00, 14:00, 19:00).
- Supabase integration: stores messages and reads user/bot metadata.
- Manual endpoints: trigger scheduled tasks on demand for testing.

## Requirements

Install dependencies (preferably inside a virtualenv):

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # PowerShell on Windows
pip install -r requirements.txt
```

Note: `requirements.txt` should include packages like `fastapi`, `uvicorn`, `supabase`, `python-dotenv`, `crewai` (if available), `llama-index`, `google-api-python-client`, `beautifulsoup4`, etc.

## Environment variables

Create a `.env` file in the repo root (or set these in your environment):

- `SUPABASE_URL` — your Supabase project URL
- `SUPABASE_KEY` — your Supabase service role or anon key
- `GEMINI_API_KEY` — Google/LLM API key for GoogleGenAI
- `GOOGLE_SEARCH_API_KEY`  — for Custom Search
- `GOOGLE_CSE_ID`  — custom search engine id

## How to run

Start the FastAPI app locally with uvicorn:

```bash
uvicorn main:app --reload
```

On startup the app registers scheduled tasks and starts the APScheduler jobs.

## API Endpoints

- `POST /news_weather_agent` — main agent endpoint. Accepts a JSON body matching `QuestionRequest` (see `main.py`). Returns persona-generated response.
- `POST /run_weekly_summary` — manually trigger the weekly summary job.
- `POST /run_major_event_alert` — manually trigger the major event alert flow.
- `POST /run_weather_alerts_user` — trigger weather alerts for user locations (testing).
- `POST /run_weather_alerts_bot` — trigger weather alerts for bot locations (testing).
- `POST /run_news_alerts_bot` — trigger news alerts for bot locations (testing).

Example `QuestionRequest` JSON:

```json
{
	"message": "What's the news in New Delhi?",
	"bot_id": "delhi",
	"custom_bot_name": "SunnyBot",
	"user_name": "Alice",
	"user_location": "Delhi",
	"language": "English",
	"email": "alice@example.com"
}
```

## Database expectations

The code expects the following tables/columns in Supabase:

- `message_paritition` — columns: `email`, `bot_id`, `user_message`, `bot_response`, `created_at`
- `user_details` — columns at least: `email`, `name`, `gender`, `city`
- `bot_personality_details` — columns at least: `bot_id`, `bot_name`, `bot_city`

If your schema differs, adapt helper functions in `main.py` accordingly.

## Tests

Run tests via pytest:

```bash
pytest -q
```

Test files present in the repo cover basic flows for the agent and proactive helpers.

## Implementation notes & next steps

- The code now contains many helpful comments and docstrings to make it easier to understand and extend.
- If you want staggered deterministic delays (to avoid mass simultaneous messages) or per-message-type rate-limiting, I can add a deterministic hashing-based delay calculator and message-history checks against Supabase.

If you'd like, I can implement the staggered-delay distribution and the stronger spam-prevention checks next.
