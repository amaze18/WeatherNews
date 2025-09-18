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

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_environment():
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

try:
    validate_environment()
except ValueError as e:
    logger.warning(f"Environment validation failed: {e}")
    logger.warning("Some features may not work correctly without proper environment configuration")

class RedisManager:
    """Redis connection and data management with chats_idx structure."""
    
    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.redis_client = None
        self._connect()
    
    def _connect(self):
        """Establish Redis connection."""
        try:
            self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
            self.redis_client.ping()
            logger.info("Redis connection established successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def _ensure_connection(self):
        """Ensure Redis connection is active."""
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

class RabbitMQManager:
    """RabbitMQ connection and message publishing with retry mechanism."""
    
    def __init__(self, rabbitmq_url: str = None):
        self.rabbitmq_url = rabbitmq_url or os.getenv("RABBITMQ_URL", "amqp://localhost:5672/")
        self.connection = None
        self.channel = None
    
    def _connect(self):
        """Establish RabbitMQ connection."""
        try:
            self.connection = pika.BlockingConnection(pika.URLParameters(self.rabbitmq_url))
            self.channel = self.connection.channel()
            logger.info("RabbitMQ connection established successfully")
        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            raise
    
    def _ensure_connection(self):
        """Ensure RabbitMQ connection is active."""
        if not self.connection or self.connection.is_closed:
            logger.warning("RabbitMQ connection lost, attempting to reconnect...")
            self._connect()
    
    def publish_to_both_queues(self, message_data: Dict[str, Any], max_retries: int = 3) -> bool:
        """
        Publish message to both storage and processing queues with retry mechanism.
        
        Args:
            message_data: Message data to publish
            max_retries: Maximum number of retry attempts
        
        Returns:
            bool: True if successful, False otherwise
        """
        for attempt in range(max_retries):
            try:
                self._ensure_connection()
                
                # Declare queues
                self.channel.queue_declare(queue='message_storage', durable=True)
                self.channel.queue_declare(queue='message_processing', durable=True)
                
                # Publish to storage queue
                self.channel.basic_publish(
                    exchange='',
                    routing_key='message_storage',
                    body=json.dumps(message_data),
                    properties=pika.BasicProperties(
                        delivery_mode=2,  # Make message persistent
                        timestamp=int(time.time())
                    )
                )
                
                # Publish to processing queue
                self.channel.basic_publish(
                    exchange='',
                    routing_key='message_processing',
                    body=json.dumps(message_data),
                    properties=pika.BasicProperties(
                        delivery_mode=2,  # Make message persistent
                        timestamp=int(time.time())
                    )
                )
                
                logger.info(f"Message published to both queues successfully (attempt {attempt + 1})")
                return True
                
            except (AMQPConnectionError, AMQPChannelError) as e:
                logger.warning(f"RabbitMQ publish attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    self.connection = None  # Force reconnection
                else:
                    logger.error(f"All RabbitMQ publish attempts failed after {max_retries} tries")
                    return False
            except Exception as e:
                logger.error(f"Unexpected error in RabbitMQ publish: {e}")
                return False
        
        return False
    
    def close(self):
        """Close RabbitMQ connection."""
        if self.channel:
            self.channel.close()
        if self.connection and not self.connection.is_closed:
            self.connection.close()
            logger.info("RabbitMQ connection closed")

class SupabaseManager:
    """Supabase connection and message storage."""
    
    def __init__(self, supabase_url: str = None, supabase_key: str = None):
        self.supabase_url = supabase_url or os.getenv("SUPABASE_URL")
        self.supabase_key = supabase_key or os.getenv("SUPABASE_KEY")
        self.client = None
        self._connect()
    
    def _connect(self):
        """Establish Supabase connection."""
        try:
            if not self.supabase_url or not self.supabase_key:
                raise ValueError("Supabase URL and KEY are required")
            
            self.client = create_client(self.supabase_url, self.supabase_key)
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
            result = self.client.table('message_paritition').insert(message_data).execute()
            if result.data:
                logger.info(f"Message stored in Supabase with ID {result.data[0].get('id', 'unknown')}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to store message in Supabase: {e}")
            return False
    
    def get_connection_status(self) -> bool:
        """Check if Supabase connection is active."""
        try:
            # Simple query to test connection
            result = self.client.table('message_paritition').select('id').limit(1).execute()
            return True
        except Exception:
            return False

async def log_and_publish_chat(
    redis_manager: RedisManager,
    user_id: str,
    user_input: str,
    bot_reply: str,
    bot_id: str,
    email: str,
    platform: str,
    requested_time: str = None,
    memory: Dict[str, Any] = None,
    mem_id: str = None
) -> Dict[str, Any]:
    """
    Comprehensive message logging function that stores to Redis, RabbitMQ, and Supabase.
    
    Args:
        redis_manager: RedisManager instance
        user_id: User identifier (format: "email:bot_id")
        user_input: User's message
        bot_reply: Bot's response
        bot_id: Bot identifier
        email: User's email
        platform: Platform identifier
        requested_time: Timestamp
        memory: Related memory data
        mem_id: Memory identifier
    
    Returns:
        Dict with logging results and status
    """
    result = {
        "success": False,
        "redis_stored": False,
        "rabbitmq_published": False,
        "supabase_stored": False,
        "errors": []
    }
    
    try:
        # Prepare message data
        current_time = datetime.now(timezone.utc).isoformat()
        message_data = {
            "user_id": user_id,
            "email": email,
            "bot_id": bot_id,
            "user_input": user_input,
            "bot_reply": bot_reply,
            "platform": platform,
            "timestamp": requested_time or current_time,
            "created_at": current_time,
            "memory": memory,
            "mem_id": mem_id
        }
        
        # 1. Store in Redis with chats_idx structure
        try:
            redis_success = redis_manager.store_chat_history(user_id, message_data)
            result["redis_stored"] = redis_success
            if redis_success:
                logger.info(f"Message stored in Redis for {user_id}")
            else:
                result["errors"].append("Failed to store in Redis")
        except Exception as e:
            result["errors"].append(f"Redis storage error: {e}")
            logger.error(f"Redis storage failed for {user_id}: {e}")
        
        # 2. Store user data in Redis
        try:
            user_data = {
                "email": email,
                "bot_id": bot_id,
                "last_activity": current_time,
                "platform": platform
            }
            redis_manager.store_user_data(user_id, user_data)
        except Exception as e:
            logger.warning(f"Failed to store user data for {user_id}: {e}")
        
        # 3. Store memory if provided
        if memory:
            try:
                redis_manager.store_memory(user_id, memory)
            except Exception as e:
                logger.warning(f"Failed to store memory for {user_id}: {e}")
        
        # 4. Publish to RabbitMQ
        try:
            rabbitmq_manager = get_rabbitmq_manager()
            rabbitmq_success = rabbitmq_manager.publish_to_both_queues(message_data)
            result["rabbitmq_published"] = rabbitmq_success
            if rabbitmq_success:
                logger.info(f"Message published to RabbitMQ for {user_id}")
            else:
                result["errors"].append("Failed to publish to RabbitMQ")
        except Exception as e:
            result["errors"].append(f"RabbitMQ publish error: {e}")
            logger.error(f"RabbitMQ publish failed for {user_id}: {e}")
        
        # 5. Store in Supabase
        try:
            supabase_manager = get_supabase_manager()
            supabase_success = supabase_manager.store_message(message_data)
            result["supabase_stored"] = supabase_success
            if supabase_success:
                logger.info(f"Message stored in Supabase for {user_id}")
            else:
                result["errors"].append("Failed to store in Supabase")
        except Exception as e:
            result["errors"].append(f"Supabase storage error: {e}")
            logger.error(f"Supabase storage failed for {user_id}: {e}")
        
        # Determine overall success
        result["success"] = result["redis_stored"] or result["rabbitmq_published"] or result["supabase_stored"]
        
        if result["success"]:
            logger.info(f"Message logging completed for {user_id}: Redis={result['redis_stored']}, RabbitMQ={result['rabbitmq_published']}, Supabase={result['supabase_stored']}")
        else:
            logger.error(f"All logging methods failed for {user_id}")
        
        return result
        
    except Exception as e:
        result["errors"].append(f"Unexpected error: {e}")
        logger.error(f"Unexpected error in log_and_publish_chat for {user_id}: {e}")
        return result

# Global instances and cleanup
_redis_manager: Optional[RedisManager] = None
_rabbitmq_manager: Optional[RabbitMQManager] = None
_supabase_manager: Optional[SupabaseManager] = None

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

def cleanup_connections():
    """Clean up all connections."""
    global _redis_manager, _rabbitmq_manager, _supabase_manager
    
    if _redis_manager:
        _redis_manager.close()
        _redis_manager = None
    
    if _rabbitmq_manager:
        _rabbitmq_manager.close()
        _rabbitmq_manager = None
    
    if _supabase_manager:
        _supabase_manager = None
    
    logger.info("All connections cleaned up")
