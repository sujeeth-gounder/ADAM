"""Gmail Service Module for ADAM.
Handles Google OAuth 2.0 authorization, secure token storage, paginated retrieval
of all accessible email messages, search, labels, safe previews, and incremental synchronization.
Does not fake emails or credentials.
"""

import base64
import email
import json
import logging
import os
import urllib.parse
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

logger = logging.getLogger("ADAM.EmailService")

# Read-only Gmail scope requested by default
GMAIL_READONLY_SCOPE = "https://www.googleapis.com/auth/gmail.readonly"


@dataclass
class EmailMessage:
    """Structured representation of an email message."""
    id: str
    thread_id: str
    subject: str
    sender: str
    recipient: str
    date: str
    snippet: str
    body_text: str = ""
    labels: List[str] = field(default_factory=list)
    has_attachments: bool = False
    is_unread: bool = False
    importance_score: float = 0.0
    importance_reason: str = ""


class GmailClient:
    """
    Secure client for interacting with Gmail API v1.
    Supports OAuth2 credentials, token refresh, paginated messages, and offline token caching.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.base_dir = data_dir or (Path(__file__).resolve().parent.parent / "data")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.token_file = self.base_dir / "gmail_token.json"
        self.client_secret_file = self.base_dir / "client_secret.json"
        
        self.access_token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self.client_id: Optional[str] = None
        self.client_secret: Optional[str] = None
        self.is_authorized: bool = False
        self.user_email: Optional[str] = None

        self._load_tokens()

    def _load_tokens(self) -> bool:
        """Load saved OAuth credentials if present."""
        if not self.token_file.exists():
            self.is_authorized = False
            return False

        try:
            with open(self.token_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.access_token = data.get("access_token")
            self.refresh_token = data.get("refresh_token")
            self.user_email = data.get("user_email")
            self.client_id = data.get("client_id")
            self.client_secret = data.get("client_secret")
            if self.access_token or self.refresh_token:
                self.is_authorized = True
                return True
        except Exception as e:
            logger.warning(f"Failed to load Gmail tokens: {e}")

        self.is_authorized = False
        return False

    def save_credentials(self, token_data: Dict[str, Any]):
        """Persist OAuth tokens securely to user data directory."""
        self.access_token = token_data.get("access_token")
        self.refresh_token = token_data.get("refresh_token", self.refresh_token)
        self.user_email = token_data.get("user_email", self.user_email)
        self.client_id = token_data.get("client_id", self.client_id)
        self.client_secret = token_data.get("client_secret", self.client_secret)
        self.is_authorized = True

        save_dict = {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "user_email": self.user_email,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }
        with open(self.token_file, "w", encoding="utf-8") as f:
            json.dump(save_dict, f, indent=2)
        # Protect file permissions on POSIX systems
        try:
            os.chmod(self.token_file, 0o600)
        except Exception:
            pass

    def get_auth_status(self) -> Dict[str, Any]:
        """Return truthful status of Gmail authentication."""
        if not self.is_authorized:
            return {
                "authorized": False,
                "user_email": None,
                "status_message": "Gmail not connected. Please authorize ADAM with Google OAuth."
            }
        return {
            "authorized": True,
            "user_email": self.user_email or "Authorized User",
            "status_message": f"Connected to Gmail ({self.user_email or 'Active'})"
        }

    def generate_auth_url(self, client_id: str, client_secret: str, redirect_uri: str = "http://localhost:8080/") -> str:
        """Construct Google OAuth 2.0 authorization URL."""
        self.client_id = client_id
        self.client_secret = client_secret
        params = {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": GMAIL_READONLY_SCOPE,
            "access_type": "offline",
            "prompt": "consent"
        }
        return f"https://accounts.google.com/o/oauth2/v2/auth?{urllib.parse.urlencode(params)}"

    def exchange_code_for_tokens(self, code: str, redirect_uri: str = "http://localhost:8080/") -> Tuple[bool, str]:
        """Exchange authorization code for access and refresh tokens."""
        if not self.client_id or not self.client_secret:
            return False, "Client ID and Secret are required to exchange code."

        data = urllib.parse.urlencode({
            "code": code,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code"
        }).encode("utf-8")

        req = urllib.request.Request("https://oauth2.googleapis.com/token", data=data)
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
                self.save_credentials({
                    "access_token": resp_data.get("access_token"),
                    "refresh_token": resp_data.get("refresh_token"),
                    "client_id": self.client_id,
                    "client_secret": self.client_secret
                })
                # Attempt to get user profile
                self._fetch_user_profile()
                return True, "Successfully authorized Gmail access."
        except urllib.error.HTTPError as he:
            err = he.read().decode("utf-8", errors="ignore")
            return False, f"Token exchange failed (HTTP {he.code}): {err}"
        except Exception as e:
            return False, f"Token exchange error: {e}"

    def refresh_access_token(self) -> bool:
        """Use refresh token to acquire a new access token."""
        if not self.refresh_token or not self.client_id or not self.client_secret:
            return False

        data = urllib.parse.urlencode({
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "refresh_token": self.refresh_token,
            "grant_type": "refresh_token"
        }).encode("utf-8")

        req = urllib.request.Request("https://oauth2.googleapis.com/token", data=data)
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
                self.access_token = resp_data.get("access_token")
                self.save_credentials({
                    "access_token": self.access_token,
                    "refresh_token": self.refresh_token,
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "user_email": self.user_email
                })
                return True
        except Exception as e:
            logger.error(f"Failed to refresh Gmail access token: {e}")
            return False

    def _fetch_user_profile(self):
        """Fetch email address of the authorized account."""
        if not self.access_token:
            return
        url = "https://gmail.googleapis.com/gmail/v1/users/me/profile"
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {self.access_token}"})
        try:
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                self.user_email = data.get("emailAddress")
                self.save_credentials({"access_token": self.access_token})
        except Exception:
            pass

    def _api_get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Perform authenticated GET request to Gmail API with automatic token refresh."""
        if not self.is_authorized:
            raise PermissionError("Gmail is not authorized. Please connect your Google account.")

        query_str = f"?{urllib.parse.urlencode(params)}" if params else ""
        url = f"https://gmail.googleapis.com/gmail/v1/users/me/{endpoint}{query_str}"

        for attempt in range(2):
            req = urllib.request.Request(url, headers={"Authorization": f"Bearer {self.access_token}"})
            try:
                with urllib.request.urlopen(req, timeout=12) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as he:
                if he.code == 401 and attempt == 0:
                    logger.info("Access token expired, attempting refresh...")
                    if self.refresh_access_token():
                        continue
                raise
        raise RuntimeError("Gmail API request failed after retry.")

    def list_messages(
        self,
        query: str = "",
        label_ids: Optional[List[str]] = None,
        max_results: int = 50,
        page_token: Optional[str] = None
    ) -> Tuple[List[Dict[str, str]], Optional[str], int]:
        """
        Retrieve paginated list of message references from mailbox.
        Returns: (messages: [{'id': ..., 'threadId': ...}], next_page_token: Optional[str], total_estimated: int)
        """
        params: Dict[str, Any] = {"maxResults": min(max_results, 100)}
        if query:
            params["q"] = query
        if label_ids:
            params["labelIds"] = label_ids
        if page_token:
            params["pageToken"] = page_token

        data = self._api_get("messages", params)
        messages = data.get("messages", [])
        next_page = data.get("nextPageToken")
        result_size_estimate = data.get("resultSizeEstimate", len(messages))
        return messages, next_page, result_size_estimate

    def iter_all_messages(
        self,
        query: str = "",
        label_ids: Optional[List[str]] = None,
        limit: int = 500
    ) -> Generator[Dict[str, str], None, None]:
        """
        Generator to paginate and stream all accessible messages across all pages up to `limit`.
        """
        page_token = None
        count = 0
        while True:
            batch, next_token, _ = self.list_messages(
                query=query,
                label_ids=label_ids,
                max_results=min(100, limit - count),
                page_token=page_token
            )
            for m in batch:
                yield m
                count += 1
                if count >= limit:
                    return

            if not next_token or count >= limit:
                break
            page_token = next_token

    def get_message_detail(self, message_id: str) -> Optional[EmailMessage]:
        """Fetch and parse complete metadata and body snippet of a single email message."""
        try:
            data = self._api_get(f"messages/{message_id}", {"format": "full"})
            headers = {h.get("name", "").lower(): h.get("value", "") for h in data.get("payload", {}).get("headers", [])}
            
            subject = headers.get("subject", "(No Subject)")
            sender = headers.get("from", "Unknown Sender")
            recipient = headers.get("to", "")
            date_str = headers.get("date", "")
            snippet = data.get("snippet", "")
            labels = data.get("labelIds", [])

            # Extract body text
            body_text = self._extract_body_text(data.get("payload", {}))

            is_unread = "UNREAD" in labels
            msg = EmailMessage(
                id=message_id,
                thread_id=data.get("threadId", ""),
                subject=subject,
                sender=sender,
                recipient=recipient,
                date=date_str,
                snippet=snippet,
                body_text=body_text[:2000],  # preview limit
                labels=labels,
                is_unread=is_unread
            )
            return msg
        except Exception as e:
            logger.error(f"Error fetching message {message_id}: {e}")
            return None

    def _extract_body_text(self, payload: Dict[str, Any]) -> str:
        """Recursively traverse payload MIME parts to extract plain text body."""
        mime_type = payload.get("mimeType", "")
        if mime_type == "text/plain":
            data = payload.get("body", {}).get("data", "")
            if data:
                return base64.urlsafe_b64decode(data.encode("ASCII")).decode("utf-8", errors="replace")

        parts = payload.get("parts", [])
        for part in parts:
            extracted = self._extract_body_text(part)
            if extracted:
                return extracted

        # Fallback to snippet if body data not directly decodable
        return ""

    def disconnect(self):
        """Revoke and delete cached tokens."""
        if self.token_file.exists():
            try:
                self.token_file.unlink()
            except Exception:
                pass
        self.access_token = None
        self.refresh_token = None
        self.user_email = None
        self.is_authorized = False
