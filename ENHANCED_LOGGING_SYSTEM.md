# Enhanced Message Logging System

## Overview

This document describes the robust message logging system implemented for the Weather News Agent, which provides comprehensive message storage and processing using Redis, RabbitMQ, and Supabase.

## Architecture

The system consists of three main components:

1. **Redis** - Immediate context/caching with TTL
2. **RabbitMQ** - Message queuing and processing
3. **Supabase** - Permanent storage and analytics

## Components

### 1. RedisManager Class

Handles Redis operations for immediate data access and caching.

**Features:**
- Connection management with retry mechanism
- User data storage with TTL
- Chat history management
- Memory management
- Automatic reconnection on connection loss

**Key Methods:**
- `store_user_data(user_id, data, ttl)` - Store user data with TTL
- `get_user_data(user_id)` - Retrieve user data
- `clear_user_data(user_id)` - Clear user data
- `store_chat_history(user_id, message, ttl)` - Store chat history
- `get_chat_history(user_id, limit)` - Retrieve chat history
- `store_memory(user_id, memory, ttl)` - Store memory data
- `get_memory(user_id)` - Retrieve memory data
- `clear_memory(user_id)` - Clear memory data

### 2. RabbitMQManager Class

Handles message queuing and processing with RabbitMQ.

**Features:**
- Connection management with retry mechanism
- Queue declaration and management
- Message publishing with persistence
- Retry mechanism for failed publishes
- Automatic reconnection on connection loss

**Key Methods:**
- `publish_to_both_queues(message, retry_count)` - Publish to both storage and processing queues
- `_ensure_connection()` - Ensure connection is active
- `close()` - Close connections

**Queues:**
- `message_storage` - For permanent storage processing
- `message_processing` - For analytics and processing

### 3. SupabaseManager Class

Handles permanent storage operations with Supabase.

**Features:**
- Connection management
- Message storage in `message_paritition` table
- User message retrieval
- Error handling and logging

**Key Methods:**
- `store_message(message_data)` - Store message in Supabase
- `get_user_messages(email, bot_id, limit)` - Retrieve user messages

### 4. Core Logging Function

**`log_and_publish_chat()`** - The main function that orchestrates all logging operations.

**Parameters:**
- `redis_manager` - RedisManager instance
- `user_id` - User identifier (format: "email:bot_id")
- `user_input` - User's message
- `bot_reply` - Bot's response
- `bot_id` - Bot identifier
- `email` - User's email
- `platform` - Platform identifier
- `requested_time` - Timestamp (optional)
- `memory` - Related memory data (optional)
- `mem_id` - Memory identifier (optional)

**Returns:**
- Dictionary with operation results and status

## Environment Configuration

### Required Environment Variables

Create a `.env` file in your project root with the following variables:

```bash
# Redis Configuration
REDIS_URL=redis://localhost:6379/0

# RabbitMQ Configuration  
RABBITMQ_URL=amqp://localhost:5672/

# Supabase Configuration
SUPABASE_URL=your_supabase_url_here
SUPABASE_KEY=your_supabase_key_here
```

### Optional Environment Variables

```bash
# TTL Settings (in seconds)
REDIS_USER_DATA_TTL=3600        # 1 hour
REDIS_CHAT_HISTORY_TTL=86400    # 24 hours
REDIS_MEMORY_TTL=604800         # 7 days

# Retry Settings
RABBITMQ_MAX_RETRIES=3
REDIS_MAX_RETRIES=3
```

## Installation and Setup

### 1. Install Dependencies

```bash
pip install redis pika asyncio-redis
```

### 2. Set Environment Variables

Create a `.env` file with the required environment variables:

```bash
# Copy from env_example.txt or create manually
REDIS_URL=redis://localhost:6379/0
RABBITMQ_URL=amqp://localhost:5672/
SUPABASE_URL=your_supabase_url_here
SUPABASE_KEY=your_supabase_key_here
```

### 3. Start Required Services

**Redis:**
```bash
# Using Docker
docker run -d -p 6379:6379 redis:latest

# Or install locally
redis-server
```

**RabbitMQ:**
```bash
# Using Docker
docker run -d -p 5672:5672 -p 15672:15672 rabbitmq:3-management

# Or install locally
rabbitmq-server
```

## Usage

### Basic Usage

```python
from message_logging_system import log_and_publish_chat, get_redis_manager

# Get Redis manager
redis_manager = get_redis_manager()

# Log a message
result = await log_and_publish_chat(
    redis_manager=redis_manager,
    user_id="user@example.com:bot_id",
    user_input="Hello, how are you?",
    bot_reply="I'm doing well, thank you!",
    bot_id="weather_bot",
    email="user@example.com",
    platform="web"
)

print(f"Logging result: {result}")
```

### Advanced Usage with Memory

```python
# Log with memory data
memory_data = {
    "conversation_context": "weather_discussion",
    "user_preferences": {"location": "New York"},
    "session_id": "session_123"
}

result = await log_and_publish_chat(
    redis_manager=redis_manager,
    user_id="user@example.com:bot_id",
    user_input="What's the weather like?",
    bot_reply="It's sunny and 75°F in New York.",
    bot_id="weather_bot",
    email="user@example.com",
    platform="mobile",
    memory=memory_data,
    mem_id="mem_123"
)
```

## API Endpoints

### Test Endpoints

- `POST /test_enhanced_logging` - Test the enhanced logging system
- `GET /logging_system_status` - Get status of all logging systems
- `GET /redis_status` - Get Redis connection status
- `GET /rabbitmq_status` - Get RabbitMQ connection status
- `GET /supabase_status` - Get Supabase connection status
- `POST /cleanup_connections` - Clean up all connections

### Example API Usage

```bash
# Test enhanced logging
curl -X POST "http://localhost:8000/test_enhanced_logging" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello, test message",
    "email": "test@example.com",
    "bot_id": "test_bot",
    "user_name": "Test User"
  }'

# Check system status
curl -X GET "http://localhost:8000/logging_system_status"
```

## Error Handling

The system includes comprehensive error handling:

1. **Connection Retry** - Automatic reconnection on connection loss
2. **Fallback Mechanisms** - Falls back to legacy Supabase logging if new system fails
3. **Error Logging** - Detailed error logging and reporting
4. **Graceful Degradation** - System continues to work even if some components fail

## Monitoring and Debugging

### Log Levels

The system uses structured logging with different levels:
- `INFO` - Normal operations
- `WARNING` - Non-critical issues
- `ERROR` - Critical failures

### Status Endpoints

Use the status endpoints to monitor system health:

```python
# Check overall system status
GET /logging_system_status

# Check individual components
GET /redis_status
GET /rabbitmq_status  
GET /supabase_status
```

## Performance Considerations

### Redis TTL Settings

- **User Data**: 1 hour (3600s) - For quick access to recent user context
- **Chat History**: 24 hours (86400s) - For conversation continuity
- **Memory**: 7 days (604800s) - For long-term user preferences

### RabbitMQ Queues

- **Durable Queues**: Messages persist across server restarts
- **Persistent Messages**: Messages survive server restarts
- **Retry Mechanism**: Failed publishes are retried with exponential backoff

## Troubleshooting

### Common Issues

1. **Redis Connection Failed**
   - Check Redis server is running
   - Verify REDIS_URL is correct
   - Check network connectivity

2. **RabbitMQ Connection Failed**
   - Check RabbitMQ server is running
   - Verify RABBITMQ_URL is correct
   - Check network connectivity

3. **Supabase Connection Failed**
   - Verify SUPABASE_URL and SUPABASE_KEY
   - Check network connectivity
   - Verify table permissions

### Debug Steps

1. Check environment variables are set correctly
2. Test individual component connections
3. Review error logs for specific issues
4. Use status endpoints to identify problems

## Integration with Existing System

The enhanced logging system is integrated into the existing FastAPI application:

1. **Automatic Integration** - All existing endpoints now use enhanced logging
2. **Backward Compatibility** - Falls back to legacy logging if new system fails
3. **Zero Downtime** - System continues to work during integration
4. **Gradual Migration** - Can be enabled/disabled per endpoint

## Future Enhancements

1. **Message Analytics** - Process RabbitMQ messages for analytics
2. **Real-time Monitoring** - Dashboard for system health
3. **Auto-scaling** - Dynamic scaling based on load
4. **Message Encryption** - Encrypt sensitive data in transit
5. **Audit Logging** - Track all system operations

## Support

For issues or questions:

1. Check the troubleshooting section
2. Review error logs
3. Use status endpoints for diagnostics
4. Check environment configuration

## License

This enhanced logging system is part of the Weather News Agent project and follows the same license terms.
