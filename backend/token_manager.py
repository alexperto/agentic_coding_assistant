"""
Token Manager for dynamic OAuth token retrieval and caching.

This module handles OAuth token fetching from Okta, caching tokens with
expiration tracking, and automatic token refresh before expiration.
"""

import os
import time
import threading
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import requests
from dataclasses import dataclass


@dataclass
class TokenData:
    """Represents a cached OAuth token with expiration tracking"""
    access_token: str
    expires_at: float  # Unix timestamp


class TokenManager:
    """
    Manages OAuth token lifecycle with caching and automatic refresh.

    This class handles:
    - Fetching OAuth tokens from Okta endpoint
    - Caching tokens with expiration tracking
    - Thread-safe token access
    - Automatic token refresh before expiration
    """

    # Refresh token 5 minutes before expiration (safety buffer)
    REFRESH_BUFFER_SECONDS = 300

    def __init__(self, client_id: str, client_secret: str, token_url: str):
        """
        Initialize TokenManager with OAuth credentials.

        Args:
            client_id: OAuth client ID from VERSA_CLIENT_ID
            client_secret: OAuth client secret from VERSA_CLIENT_SECRET
            token_url: Okta OAuth token endpoint URL

        Raises:
            ValueError: If client_id or client_secret is empty
        """
        if not client_id or not client_secret:
            raise ValueError(
                "TokenManager requires valid client_id and client_secret. "
                "Ensure VERSA_CLIENT_ID and VERSA_CLIENT_SECRET are set in .env"
            )

        self.client_id = client_id
        self.client_secret = client_secret
        self.token_url = token_url

        # Thread-safe token cache
        self._lock = threading.Lock()
        self._cached_token: Optional[TokenData] = None

    def get_token(self) -> str:
        """
        Get a valid OAuth token, fetching or refreshing as needed.

        This method is thread-safe and handles:
        - Returning cached token if still valid
        - Fetching new token if none cached
        - Refreshing token if close to expiration

        Returns:
            Valid OAuth access token string

        Raises:
            RuntimeError: If token fetch fails
        """
        with self._lock:
            # Check if we need to fetch/refresh token
            if self._is_token_expired():
                self._fetch_and_cache_token()

            # Return cached token
            if self._cached_token:
                return self._cached_token.access_token

            # Should not reach here if _fetch_and_cache_token works correctly
            raise RuntimeError("Failed to obtain valid token")

    def _is_token_expired(self) -> bool:
        """
        Check if cached token is expired or needs refresh.

        Returns:
            True if token needs refresh, False if still valid
        """
        if not self._cached_token:
            return True

        # Calculate refresh time (expires_at - buffer)
        refresh_time = self._cached_token.expires_at - self.REFRESH_BUFFER_SECONDS
        current_time = time.time()

        return current_time >= refresh_time

    def _fetch_and_cache_token(self) -> None:
        """
        Fetch new token from Okta and cache it.

        Makes OAuth request to token endpoint and caches the result
        with calculated expiration time.

        Raises:
            RuntimeError: If OAuth request fails or returns invalid response
        """
        try:
            # Prepare OAuth request
            headers = {
                "Content-Type": "application/x-www-form-urlencoded"
            }

            data = {
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "grant_type": "client_credentials",
                "scope": "versa.web versa.chat versa.assistant"
            }

            # Make OAuth request
            response = requests.post(
                self.token_url,
                headers=headers,
                data=data,
                timeout=10  # 10 second timeout
            )

            # Check for HTTP errors
            response.raise_for_status()

            # Parse response
            token_response = response.json()

            # Extract access token
            access_token = token_response.get("access_token")
            if not access_token:
                raise RuntimeError(
                    f"OAuth response missing 'access_token' field. "
                    f"Response: {token_response}"
                )

            # Extract expiration time (default to 3600 seconds = 1 hour)
            expires_in = token_response.get("expires_in", 3600)
            expires_at = time.time() + expires_in

            # Cache token
            self._cached_token = TokenData(
                access_token=access_token,
                expires_at=expires_at
            )

            # Log success (without exposing token)
            expiry_time = datetime.fromtimestamp(expires_at).strftime('%Y-%m-%d %H:%M:%S')
            print(f"[TokenManager] Successfully fetched new OAuth token (expires at {expiry_time})")

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"OAuth token request failed: {str(e)}") from e
        except (KeyError, ValueError) as e:
            raise RuntimeError(f"Invalid OAuth response format: {str(e)}") from e

    def clear_cache(self) -> None:
        """
        Clear cached token (useful for testing or forcing refresh).

        This method is thread-safe.
        """
        with self._lock:
            self._cached_token = None
            print("[TokenManager] Token cache cleared")


def create_token_manager_from_env() -> Optional[TokenManager]:
    """
    Factory function to create TokenManager from environment variables.

    Returns:
        TokenManager instance if OAuth credentials are configured,
        None if credentials are not available (for backward compatibility)
    """
    client_id = os.getenv("VERSA_CLIENT_ID", "")
    client_secret = os.getenv("VERSA_CLIENT_SECRET", "")
    token_url = os.getenv(
        "OKTA_TOKEN_URL",
        "https://uc-sf.okta.com/oauth2/ausnwf6tyaq6v47QF5d7/v1/token"
    )

    # Only create TokenManager if credentials are provided
    if client_id and client_secret:
        try:
            return TokenManager(client_id, client_secret, token_url)
        except ValueError as e:
            print(f"[TokenManager] Warning: {e}")
            return None

    # No credentials configured - return None for backward compatibility
    return None
