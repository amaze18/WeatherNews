"""
Robust Message Logging System with Redis, RabbitMQ, and Supabase
===============================================================

This module provides a comprehensive message logging system that:
- Uses Redis for immediate context/caching with TTL
- Uses RabbitMQ for message queuing and processing
- Uses Supabase for permanent storage
- Implements retry mechanisms and error handling
- Provides connection management and memory management
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
import redis
import pika
from pika.exceptions import AMQPConnectionError, AMQPChannelError
from supabase import Client, create_client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variable validation
def validate_environment():
    """Validate that all required environment variables are set."""
    required_vars = ["REDIS_URL", "RABBITMQ_URL", "SUPABASE_URL", "SUPABASE_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing_vars)}\n"
            f"Please set the following in your .env file:\n"
            f"- REDIS_URL=redis://localhost:6379/0\n"
            f"- RABBITMQ_URL=amqp://localhost:5672/\n"
            f"- SUPABASE_URL=your_supabase_url\n"
            f"- SUPABASE_KEY=your_supabase_key"
        )

# Validate environment on import
try:
    validate_environment()
except ValueError as e:
    logger.warning(f"Environment validation failed: {e}")
    logger.warning("Some features may not work correctly without proper environment configuration")


class RedisManager:
    """
    Redis Manager for handling immediate context/caching operations.
    Provides connection management, user data loading/clearing, chat history storage, and memory management.
    """
    
    def __init__(self, redis_url: str = None):
        """
        Initialize Redis connection.
        
        Args:
            redis_url: Redis connection URL. If None, uses REDIS_URL from environment.
        """
        self.redis_url = redis_url or os.getenv("REDIS_URL")
        if not self.redis_url:
            raise ValueError("Redis URL is required. Set REDIS_URL environment variable or pass redis_url parameter.")
        
        self.redis_client = None
        self.connection_pool = None
        self._connect()
    
    def _connect(self):
        """Establish Redis connection with retry mechanism."""
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
                # Test connection
                self.redis_client.ping()
                logger.info(f"Redis connection established successfully")
                return
            except Exception as e:
                logger.warning(f"Redis connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    logger.error(f"Failed to connect to Redis after {max_retries} attempts")
                    raise
    
    def _ensure_connection(self):
        """Ensure Redis connection is active, reconnect if necessary."""
        try:
            self.redis_client.ping()
        except Exception:
            logger.warning("Redis connection lost, attempting to reconnect...")
            self._connect()
    
    def store_user_data(self, user_id: str, data: Dict[str, Any], ttl: int = 3600) -> bool:
        """
        Store user data in Redis with TTL using chats_idx structure.
        
        Args:
            user_id: User identifier (format: "email:bot_id")
            data: Data to store
            ttl: Time to live in seconds (default: 1 hour)
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self._ensure_connection()
            # Use chats_idx structure for consistency
            key = f"chats_idx:user_data:{user_id}"
            self.redis_client.setex(key, ttl, json.dumps(data))
            logger.info(f"User data stored in chats_idx for {user_id} with TTL {ttl}s")
            return True
        except Exception as e:
            logger.error(f"Failed to store user data for {user_id}: {e}")
            return False
    
    def get_user_data(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve user data from Redis using chats_idx structure.
        
        Args:
            user_id: User identifier (format: "email:bot_id")
        
        Returns:
            Dict containing user data or None if not found
        """
        try:
            self._ensure_connection()
            key = f"chats_idx:user_data:{user_id}"
            data = self.redis_client.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Failed to get user data for {user_id}: {e}")
            return None
    
    def clear_user_data(self, user_id: str) -> bool:
        """
        Clear user data from Redis using chats_idx structure.
        
        Args:
            user_id: User identifier (format: "email:bot_id")
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self._ensure_connection()
            key = f"chats_idx:user_data:{user_id}"
            result = self.redis_client.delete(key)
            logger.info(f"User data cleared from chats_idx for {user_id}")
            return bool(result)
        except Exception as e:
            logger.error(f"Failed to clear user data for {user_id}: {e}")
            return False
    
    def store_chat_history(self, user_id: str, message: Dict[str, Any], ttl: int = 86400) -> bool:
        """
        Store chat history in Redis using the chats_idx index and update Redis Search index.
        
        Args:
            user_id: User identifier (format: "email:bot_id")
            message: Message data to store
            ttl: Time to live in seconds (default: 24 hours)
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self._ensure_connection()
            # Use the same index structure as your existing chats
            key = f"chats_idx:{user_id}"
            # Store as a list in Redis
            self.redis_client.lpush(key, json.dumps(message))
            self.redis_client.expire(key, ttl)
            
            # Also index the message in Redis Search
            try:
                # Create a document ID for Redis Search
                doc_id = f"msg:{user_id}:{int(time.time())}"
                
                # Prepare document for indexing
                search_doc = {
                    "email": message.get("email", ""),
                    "bot_id": message.get("bot_id", ""),
                    "user_input": message.get("user_input", ""),
                    "bot_reply": message.get("bot_reply", ""),
                    "platform": message.get("platform", ""),
                    "timestamp": message.get("timestamp", ""),
                    "user_id": user_id
                }
                
                # Index the document in Redis Search
                self.redis_client.execute_command(
                    "FT.ADD", "chats_idx", doc_id, "1.0", "FIELDS",
                    *[f"{k}:{v}" for k, v in search_doc.items() if v]
                )
                
                logger.info(f"Message indexed in Redis Search for {user_id}")
                
            except Exception as search_error:
                logger.warning(f"Redis Search indexing failed for {user_id}: {search_error}")
                # Don't fail the entire operation if search indexing fails
            
            logger.info(f"Chat history stored in chats_idx for {user_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to store chat history for {user_id}: {e}")
            return False
    
    def get_chat_history(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Retrieve chat history from Redis using chats_idx.
        
        Args:
            user_id: User identifier (format: "email:bot_id")
            limit: Maximum number of messages to retrieve
        
        Returns:
            List of message dictionaries
        """
        try:
            self._ensure_connection()
            key = f"chats_idx:{user_id}"
            messages = self.redis_client.lrange(key, 0, limit - 1)
            return [json.loads(msg) for msg in messages]
        except Exception as e:
            logger.error(f"Failed to get chat history for {user_id}: {e}")
            return []
    
    def store_memory(self, user_id: str, memory: Dict[str, Any], ttl: int = 604800) -> bool:
        """
        Store memory data in Redis with TTL using chats_idx structure.
        
        Args:
            user_id: User identifier (format: "email:bot_id")
            memory: Memory data to store
            ttl: Time to live in seconds (default: 7 days)
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self._ensure_connection()
            key = f"chats_idx:memory:{user_id}"
            self.redis_client.setex(key, ttl, json.dumps(memory))
            logger.info(f"Memory stored in chats_idx for {user_id} with TTL {ttl}s")
            return True
        except Exception as e:
            logger.error(f"Failed to store memory for {user_id}: {e}")
            return False
    
    def get_memory(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve memory data from Redis using chats_idx structure.
        
        Args:
            user_id: User identifier (format: "email:bot_id")
        
        Returns:
            Dict containing memory data or None if not found
        """
        try:
            self._ensure_connection()
            key = f"chats_idx:memory:{user_id}"
            data = self.redis_client.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Failed to get memory for {user_id}: {e}")
            return None
    
    def clear_memory(self, user_id: str) -> bool:
        """
        Clear memory data from Redis using chats_idx structure.
        
        Args:
            user_id: User identifier (format: "email:bot_id")
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self._ensure_connection()
            key = f"chats_idx:memory:{user_id}"
            result = self.redis_client.delete(key)
            logger.info(f"Memory cleared from chats_idx for {user_id}")
            return bool(result)
        except Exception as e:
            logger.error(f"Failed to clear memory for {user_id}: {e}")
            return False
    
    def close(self):
        """Close Redis connection."""
        if self.redis_client:
            self.redis_client.close()
            logger.info("Redis connection closed")


class RabbitMQManager:
    """
    RabbitMQ Manager for message queuing and processing.
    Handles connection management, channel management, and retry mechanisms.
    """
    
    def __init__(self, rabbitmq_url: str = None):
        """
        Initialize RabbitMQ connection.
        
        Args:
            rabbitmq_url: RabbitMQ connection URL. If None, uses RABBITMQ_URL from environment.
        """
        self.rabbitmq_url = rabbitmq_url or os.getenv("RABBITMQ_URL")
        if not self.rabbitmq_url:
            raise ValueError("RabbitMQ URL is required. Set RABBITMQ_URL environment variable or pass rabbitmq_url parameter.")
        
        self.connection = None
        self.channel = None
        self._connect()
    
    def _connect(self):
        """Establish RabbitMQ connection with retry mechanism."""
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                self.connection = pika.BlockingConnection(pika.URLParameters(self.rabbitmq_url))
                self.channel = self.connection.channel()
                
                # Declare queues
                self.channel.queue_declare(queue='message_storage', durable=True)
                self.channel.queue_declare(queue='message_processing', durable=True)
                
                logger.info("RabbitMQ connection established successfully")
                return
            except Exception as e:
                logger.warning(f"RabbitMQ connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    logger.error(f"Failed to connect to RabbitMQ after {max_retries} attempts")
                    raise
    
    def _ensure_connection(self):
        """Ensure RabbitMQ connection is active, reconnect if necessary."""
        try:
            if not self.connection or self.connection.is_closed:
                self._connect()
        except Exception as e:
            logger.warning(f"RabbitMQ connection issue: {e}, attempting to reconnect...")
            self._connect()
    
    def publish_to_both_queues(self, message: Dict[str, Any], retry_count: int = 3) -> bool:
        """
        Publish message to both storage and processing queues.
        
        Args:
            message: Message data to publish
            retry_count: Number of retry attempts
        
        Returns:
            bool: True if successful, False otherwise
        """
        for attempt in range(retry_count):
            try:
                self._ensure_connection()
                
                # Publish to storage queue
                self.channel.basic_publish(
                    exchange='',
                    routing_key='message_storage',
                    body=json.dumps(message),
                    properties=pika.BasicProperties(
                        delivery_mode=2,  # Make message persistent
                        timestamp=int(time.time())
                    )
                )
                
                # Publish to processing queue
                self.channel.basic_publish(
                    exchange='',
                    routing_key='message_processing',
                    body=json.dumps(message),
                    properties=pika.BasicProperties(
                        delivery_mode=2,  # Make message persistent
                        timestamp=int(time.time())
                    )
                )
                
                logger.info(f"Message published to both queues successfully (attempt {attempt + 1})")
                return True
                
            except (AMQPConnectionError, AMQPChannelError) as e:
                logger.warning(f"RabbitMQ publish attempt {attempt + 1} failed: {e}")
                if attempt < retry_count - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    self._connect()  # Reconnect
                else:
                    logger.error(f"Failed to publish message after {retry_count} attempts")
                    return False
            except Exception as e:
                logger.error(f"Unexpected error publishing message: {e}")
                return False
        
        return False
    
    def close(self):
        """Close RabbitMQ connection."""
        if self.channel and not self.channel.is_closed:
            self.channel.close()
        if self.connection and not self.connection.is_closed:
            self.connection.close()
        logger.info("RabbitMQ connection closed")


class SupabaseManager:
    """
    Supabase Manager for permanent storage operations.
    Handles connection management and data persistence.
    """
    
    def __init__(self, supabase_url: str = None, supabase_key: str = None):
        """
        Initialize Supabase connection.
        
        Args:
            supabase_url: Supabase URL. If None, uses SUPABASE_URL from environment.
            supabase_key: Supabase key. If None, uses SUPABASE_KEY from environment.
        """
        self.supabase_url = supabase_url or os.getenv("SUPABASE_URL")
        self.supabase_key = supabase_key or os.getenv("SUPABASE_KEY")
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("Supabase URL and key are required. Set SUPABASE_URL and SUPABASE_KEY environment variables.")
        
        self.supabase_client = None
        self._connect()
    
    def _connect(self):
        """Establish Supabase connection."""
        try:
            self.supabase_client = create_client(self.supabase_url, self.supabase_key)
            logger.info("Supabase connection established successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Supabase: {e}")
            raise
    
    def store_message(self, message_data: Dict[str, Any]) -> bool:
        """
        Store message in Supabase message_paritition table.
        
        Args:
            message_data: Message data to store
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            response = self.supabase_client.table("message_paritition").insert(message_data).execute()
            
            if response.data and len(response.data) > 0:
                record_id = response.data[0].get('id', 'N/A')
                logger.info(f"Message stored in Supabase with ID {record_id}")
                return True
            else:
                logger.error("No data returned from Supabase insert operation")
                return False
                
        except Exception as e:
            logger.error(f"Failed to store message in Supabase: {e}")
            return False
    
    def get_user_messages(self, email: str, bot_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve user messages from Supabase.
        
        Args:
            email: User email
            bot_id: Bot identifier
            limit: Maximum number of messages to retrieve
        
        Returns:
            List of message dictionaries
        """
        try:
            response = self.supabase_client.table("message_paritition") \
                .select("*") \
                .eq("email", email) \
                .eq("bot_id", bot_id) \
                .order("created_at", desc=True) \
                .limit(limit) \
                .execute()
            
            return response.data if response.data else []
        except Exception as e:
            logger.error(f"Failed to get user messages from Supabase: {e}")
            return []


async def log_and_publish_chat(
    redis_manager: RedisManager,
    user_id: str,
    user_input: str,
    bot_reply: str,
    bot_id: str,
    email: str,
    platform: str,
    requested_time: str = None,
    memory: dict = None,
    mem_id: str = None
) -> Dict[str, Any]:
    """
    Comprehensive message logging function that handles Redis, RabbitMQ, and Supabase.
    
    Args:
        redis_manager: RedisManager instance for caching
        user_id: User identifier (format: "email:bot_id")
        user_input: User's message
        bot_reply: Bot's response
        bot_id: Bot identifier
        email: User's email
        platform: Platform identifier
        requested_time: Timestamp (defaults to current time)
        memory: Related memory data
        mem_id: Memory identifier
    
    Returns:
        Dict containing operation results and status
    """
    result = {
        "success": False,
        "redis_stored": False,
        "rabbitmq_published": False,
        "supabase_stored": False,
        "errors": []
    }
    
    # Set default timestamp
    if not requested_time:
        requested_time = datetime.now(timezone.utc).isoformat()
    
    # Prepare message data
    message_data = {
        "email": email,
        "bot_id": bot_id,
        "user_message": user_input,
        "bot_response": bot_reply,
        "requested_time": requested_time,
        "platform": platform,
        "created_at": requested_time
    }
    
    # Add memory data if provided
    if memory:
        message_data["memory"] = json.dumps(memory)
    if mem_id:
        message_data["mem_id"] = mem_id
    
    # 1. Store in Redis for immediate access
    try:
        redis_manager.store_user_data(user_id, {
            "last_message": user_input,
            "last_reply": bot_reply,
            "timestamp": requested_time
        })
        
        # Store chat history
        redis_manager.store_chat_history(user_id, message_data)
        
        # Store memory if provided
        if memory:
            redis_manager.store_memory(user_id, memory)
        
        result["redis_stored"] = True
        logger.info(f"Message stored in Redis for {user_id}")
        
    except Exception as e:
        error_msg = f"Redis storage failed: {e}"
        logger.error(error_msg)
        result["errors"].append(error_msg)
    
    # 2. Publish to RabbitMQ for processing
    try:
        rabbitmq_manager = RabbitMQManager()
        success = rabbitmq_manager.publish_to_both_queues(message_data)
        rabbitmq_manager.close()
        
        if success:
            result["rabbitmq_published"] = True
            logger.info(f"Message published to RabbitMQ for {user_id}")
        else:
            result["errors"].append("RabbitMQ publish failed")
            
    except Exception as e:
        error_msg = f"RabbitMQ publish failed: {e}"
        logger.error(error_msg)
        result["errors"].append(error_msg)
    
    # 3. Store in Supabase for permanent storage
    try:
        supabase_manager = SupabaseManager()
        success = supabase_manager.store_message(message_data)
        
        if success:
            result["supabase_stored"] = True
            logger.info(f"Message stored in Supabase for {user_id}")
        else:
            result["errors"].append("Supabase storage failed")
            
    except Exception as e:
        error_msg = f"Supabase storage failed: {e}"
        logger.error(error_msg)
        result["errors"].append(error_msg)
    
    # Determine overall success
    result["success"] = result["redis_stored"] and result["supabase_stored"]
    
    logger.info(f"Message logging completed for {user_id}: Redis={result['redis_stored']}, "
                f"RabbitMQ={result['rabbitmq_published']}, Supabase={result['supabase_stored']}")
    
    return result


# Global managers (singleton pattern)
_redis_manager = None
_rabbitmq_manager = None
_supabase_manager = None


def get_redis_manager() -> RedisManager:
    """Get or create Redis manager instance."""
    global _redis_manager
    if _redis_manager is None:
        _redis_manager = RedisManager()
    return _redis_manager


def get_rabbitmq_manager() -> RabbitMQManager:
    """Get or create RabbitMQ manager instance."""
    global _rabbitmq_manager
    if _rabbitmq_manager is None:
        _rabbitmq_manager = RabbitMQManager()
    return _rabbitmq_manager


def get_supabase_manager() -> SupabaseManager:
    """Get or create Supabase manager instance."""
    global _supabase_manager
    if _supabase_manager is None:
        _supabase_manager = SupabaseManager()
    return _supabase_manager


# Cleanup function
def cleanup_connections():
    """Clean up all connections."""
    global _redis_manager, _rabbitmq_manager, _supabase_manager
    
    if _redis_manager:
        _redis_manager.close()
        _redis_manager = None
    
    if _rabbitmq_manager:
        _rabbitmq_manager.close()
        _rabbitmq_manager = None
    
    logger.info("All connections cleaned up")


# Example usage and testing functions
async def test_message_logging_system():
    """Test the complete message logging system."""
    try:
        # Test Redis
        redis_manager = get_redis_manager()
        test_user_id = "test@example.com:test_bot"
        redis_manager.store_user_data(test_user_id, {"test": "data"})
        data = redis_manager.get_user_data(test_user_id)
        print(f"Redis test: {data}")
        
        # Test message logging
        result = await log_and_publish_chat(
            redis_manager=redis_manager,
            user_id=test_user_id,
            user_input="Hello, how are you?",
            bot_reply="I'm doing well, thank you!",
            bot_id="test_bot",
            email="test@example.com",
            platform="test_platform"
        )
        
        print(f"Message logging result: {result}")
        return result
        
    except Exception as e:
        print(f"Test failed: {e}")
        return None
    finally:
        cleanup_connections()


if __name__ == "__main__":
    # Run test if executed directly
    asyncio.run(test_message_logging_system())
