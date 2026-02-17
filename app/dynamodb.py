"""
DynamoDB integration for conversation persistence.

Stores conversation history keyed by a hash of the user's API key + session ID.
Uses TTL for automatic cleanup of old conversations.
"""

import hashlib
import os
import time
from typing import Optional

import boto3
from botocore.exceptions import ClientError

TABLE_NAME = os.environ.get("DYNAMODB_TABLE", "friendly-advice-conversations")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-2")
CONVERSATION_TTL_DAYS = int(os.environ.get("CONVERSATION_TTL_DAYS", "90"))

# Lazy-initialized DynamoDB resource
_table = None


def _get_table():
    """Get the DynamoDB table resource (lazy-initialized)."""
    global _table
    if _table is None:
        dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
        _table = dynamodb.Table(TABLE_NAME)
    return _table


def _user_hash(api_key: str) -> str:
    """Create a consistent hash from an API key to use as user identifier."""
    return hashlib.sha256(api_key.encode()).hexdigest()[:16]


def save_conversation(
    api_key: str,
    session_id: str,
    question: str,
    response: str,
    preview: str,
) -> Optional[dict]:
    """
    Save or update a conversation in DynamoDB.

    Args:
        api_key: User's OpenAI API key (hashed for storage)
        session_id: Client-generated session ID
        question: The user's question
        response: The advice response (HTML)
        preview: Short preview text for the sidebar

    Returns:
        The saved item dict, or None on failure
    """
    table = _get_table()
    user_id = _user_hash(api_key)
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    ttl = int(time.time()) + (CONVERSATION_TTL_DAYS * 86400)

    item = {
        "user_id": user_id,
        "session_id": session_id,
        "question": question,
        "response": response,
        "preview": preview,
        "updated_at": now_iso,
        "ttl": ttl,
    }

    try:
        table.put_item(Item=item)
        return item
    except ClientError as e:
        print(f"DynamoDB put_item error: {e}")
        return None


def get_conversations(api_key: str, limit: int = 50) -> list[dict]:
    """
    Get all conversations for a user, sorted by most recent.

    Args:
        api_key: User's OpenAI API key (hashed for lookup)
        limit: Maximum number of conversations to return

    Returns:
        List of conversation dicts
    """
    table = _get_table()
    user_id = _user_hash(api_key)

    try:
        response = table.query(
            KeyConditionExpression=boto3.dynamodb.conditions.Key("user_id").eq(user_id),
            ScanIndexForward=False,  # Most recent first
            Limit=limit,
        )
        return response.get("Items", [])
    except ClientError as e:
        print(f"DynamoDB query error: {e}")
        return []


def get_conversation(api_key: str, session_id: str) -> Optional[dict]:
    """
    Get a specific conversation.

    Args:
        api_key: User's OpenAI API key (hashed for lookup)
        session_id: The session ID to retrieve

    Returns:
        Conversation dict or None
    """
    table = _get_table()
    user_id = _user_hash(api_key)

    try:
        response = table.get_item(Key={"user_id": user_id, "session_id": session_id})
        return response.get("Item")
    except ClientError as e:
        print(f"DynamoDB get_item error: {e}")
        return None


def delete_conversation(api_key: str, session_id: str) -> bool:
    """
    Delete a specific conversation.

    Args:
        api_key: User's OpenAI API key (hashed for lookup)
        session_id: The session ID to delete

    Returns:
        True if successful, False otherwise
    """
    table = _get_table()
    user_id = _user_hash(api_key)

    try:
        table.delete_item(Key={"user_id": user_id, "session_id": session_id})
        return True
    except ClientError as e:
        print(f"DynamoDB delete_item error: {e}")
        return False


def delete_all_conversations(api_key: str) -> bool:
    """
    Delete all conversations for a user.

    Args:
        api_key: User's OpenAI API key (hashed for lookup)

    Returns:
        True if successful, False otherwise
    """
    table = _get_table()
    user_id = _user_hash(api_key)

    try:
        # Query all items for this user
        response = table.query(
            KeyConditionExpression=boto3.dynamodb.conditions.Key("user_id").eq(user_id),
            ProjectionExpression="user_id, session_id",
        )

        # Batch delete
        with table.batch_writer() as batch:
            for item in response.get("Items", []):
                batch.delete_item(Key={"user_id": item["user_id"], "session_id": item["session_id"]})

        return True
    except ClientError as e:
        print(f"DynamoDB batch delete error: {e}")
        return False
