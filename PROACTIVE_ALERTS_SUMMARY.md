# Proactive Alerts System — Current Implementation Summary

This document describes the proactive alerting features actually implemented in this repository and highlights recent code-level improvements (comments, docstrings, and clearer scheduling). It intentionally focuses on what is present in code rather than speculative or planned features.

## High-level overview
- Weekly news summary is scheduled using FastAPI's `repeat_every` decorator.
- APScheduler is used to run proactive jobs (weather/news) at fixed times of day.
- Manual API endpoints are available to trigger scheduled tasks for testing.
- Supabase is used as the backing store for messages, user details, and bot personality details.
- Several helper functions provide user-bot pair detection and message insertion.

## Implemented features (what's in the code)

1) Scheduled weekly summary
- `scheduled_weekly_news_summary()` — decorated with `@repeat_every(seconds=60*60*24*7)` to run once every 7 days. It collects agent parameters and stores generated weekly summaries via `insert_bot_message()`.

2) APScheduler-based proactive jobs
- `start_scheduler()` initializes an APScheduler `BackgroundScheduler` and registers three jobs:
	- `send_weather_user_alerts` — scheduled at 08:00
	- `send_weather_bot_alerts` — scheduled at 14:00
	- `send_news_user_alerts` — scheduled at 19:00

3) Manual endpoints for testing and on-demand runs
- `POST /run_weekly_summary` — manually trigger the weekly summary task
- `POST /run_major_event_alert` — manually trigger the major event alert flow
- `POST /run_weather_alerts_user`, `POST /run_weather_alerts_bot`, `POST /run_news_alerts_bot` — endpoints to run specific alert types for testing

4) Supabase integration and helper utilities
- `SUPABASE_URL` and `SUPABASE_KEY` are read from environment variables and used to create a Supabase client.
- `get_today_user_bot_pairs()` — fetches unique (email, bot_id) pairs from today's messages (table `message_paritition`).
- `get_all_news_agent_params()` — for each pair it loads `user_details` and `bot_personality_details`, formats the bot prompt using `get_bot_prompt()`, and returns a params list used by alert tasks.
- `insert_bot_message(email, bot_id, message)` — inserts a bot response into `message_paritition`.

5) News & Weather agent functions (in `news_weather_agent.py`)
- `persona_response(...)` — main async function that classifies messages (news/weather/other), calls the crew agent, and returns persona-based replies.
- `generate_weekly_news_summary(...)`, `check_and_alert_for_weather_user(...)`, `check_and_alert_for_weather_bot(...)`, `check_and_alert_for_major_events_user(...)`, `check_and_alert_for_major_events_bot(...)` — implemented in `news_weather_agent.py` and used by the scheduled routines.

## Important configuration / environment variables

- `SUPABASE_URL` — Supabase project URL
- `SUPABASE_KEY` — Supabase service role or anon key
- `GEMINI_API_KEY` — LLM/API key used by `news_weather_agent` (GoogleGenAI)
- `GOOGLE_SEARCH_API_KEY` / `GOOGLE_CSE_ID` — optional (used by Google search helper in `news_weather_agent.py`)

Ensure these are placed in the repository `.env` file or set in the environment prior to running the app.

## How scheduling behaves at runtime

- On FastAPI startup the `scheduled_weekly_news_summary` task is registered via `app.add_event_handler('startup', scheduled_weekly_news_summary)` and will run once per week while the process runs.
- `start_scheduler()` is also registered on startup and starts the APScheduler jobs for proactive alerts at fixed hours (08:00, 14:00, 19:00). These jobs iterate through today's user-bot pairs and call the corresponding check/alert functions.

## Database / table expectations

- `message_paritition` (note the existing table name in code)
	- Expected columns: `email`, `bot_id`, `user_message`, `bot_response`, `created_at` (used for date filtering)
- `user_details` — at least `email`, `name`, `gender`, `city` fields expected by the helper utilities
- `bot_personality_details` — expected fields: `bot_id`, `bot_name`, `bot_city`
