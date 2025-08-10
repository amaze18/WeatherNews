#!/usr/bin/env python3
"""
Test script for proactive alerts system
"""

import asyncio
import os
from dotenv import load_dotenv
from main import (
    get_today_user_bot_pairs,
    get_all_news_agent_params,
    should_send_proactive_message,
    get_staggered_delay,
    send_staggered_proactive_messages
)

# Load environment variables
load_dotenv()

async def test_proactive_alerts():
    """Test the proactive alerts system"""
    print("=== Testing Proactive Alerts System ===\n")
    
    # Test 1: Check if we can get today's user-bot pairs
    print("1. Testing get_today_user_bot_pairs()...")
    pairs = get_today_user_bot_pairs()
    print(f"   Found {len(pairs)} today's user-bot pairs")
    for email, bot_id in pairs[:3]:  # Show first 3
        print(f"   - {email} -> {bot_id}")
    print()
    
    # Test 2: Check if we can get all parameters
    print("2. Testing get_all_news_agent_params()...")
    params_list = get_all_news_agent_params()
    print(f"   Found {len(params_list)} parameter sets")
    if params_list:
        sample_params = params_list[0]
        print(f"   Sample params: {sample_params['email']} -> {sample_params['bot_id']}")
        print(f"   User location: {sample_params['user_location']}")
        print(f"   Bot city: {sample_params['bot_city']}")
    print()
    
    # Test 3: Test staggered delay calculation
    print("3. Testing get_staggered_delay()...")
    if params_list:
        email = params_list[0]['email']
        bot_id = params_list[0]['bot_id']
        delay = get_staggered_delay(email, bot_id, 0, 6)
        print(f"   Delay for {email} -> {bot_id}: {delay} seconds ({delay/3600:.2f} hours)")
    print()
    
    # Test 4: Test spam prevention
    print("4. Testing should_send_proactive_message()...")
    if params_list:
        email = params_list[0]['email']
        bot_id = params_list[0]['bot_id']
        
        # Test different message types
        message_types = ["[WEATHER_USER]", "[WEATHER_BOT]", "[NEWS_USER]", "[NEWS_BOT]"]
        for msg_type in message_types:
            should_send = should_send_proactive_message(email, bot_id, msg_type, min_interval_hours=12)
            print(f"   {msg_type}: {'Should send' if should_send else 'Should NOT send'}")
    print()
    
    # Test 5: Test the full staggered message system
    print("5. Testing send_staggered_proactive_messages()...")
    if params_list:
        print("   Starting staggered proactive messages...")
        try:
            await send_staggered_proactive_messages()
            print("   ✅ Staggered proactive messages completed successfully")
        except Exception as e:
            print(f"   ❌ Error in staggered proactive messages: {e}")
    else:
        print("   ⚠️  No user-bot pairs found, skipping test")
    print()
    
    print("=== Test Complete ===")

if __name__ == "__main__":
    # Check if required environment variables are set
    required_vars = ["SUPABASE_URL", "SUPABASE_KEY", "GEMINI_API_KEY", "GOOGLE_SEARCH_API_KEY", "GOOGLE_CSE_ID"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {missing_vars}")
        print("Please set these in your .env file")
        exit(1)
    
    # Run the test
    asyncio.run(test_proactive_alerts())
