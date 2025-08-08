"""
Authentication module for bearer token validation
"""

import hashlib
import hmac
from typing import Optional
import structlog

from .config import settings

logger = structlog.get_logger()


def verify_token(token: str) -> bool:
    """
    Verify the bearer token against the expected token
    
    Args:
        token: The token to verify
        
    Returns:
        bool: True if token is valid, False otherwise
    """
    if not token:
        logger.warning("Empty token provided")
        return False
    
    # Check against the expected token from specs
    expected_token = settings.VALID_BEARER_TOKEN
    
    if not expected_token:
        logger.error("No valid bearer token configured")
        return False
    
    # Use secure comparison to prevent timing attacks
    is_valid = hmac.compare_digest(token, expected_token)
    
    if not is_valid:
        logger.warning("Invalid token provided", token_prefix=token[:10] + "...")
    else:
        logger.info("Valid token authenticated", token_prefix=token[:10] + "...")
    
    return is_valid


def hash_token(token: str) -> str:
    """
    Create a hash of the token for logging purposes
    
    Args:
        token: The token to hash
        
    Returns:
        str: Hashed token
    """
    return hashlib.sha256(token.encode()).hexdigest()[:16]


class TokenValidator:
    """Token validation utility class"""
    
    def __init__(self):
        self.valid_tokens = {settings.VALID_BEARER_TOKEN}
    
    def is_valid(self, token: str) -> bool:
        """Check if token is in the set of valid tokens"""
        return token in self.valid_tokens
    
    def add_token(self, token: str) -> None:
        """Add a new valid token"""
        self.valid_tokens.add(token)
    
    def remove_token(self, token: str) -> None:
        """Remove a token from valid tokens"""
        self.valid_tokens.discard(token)


# Global token validator instance
token_validator = TokenValidator()
