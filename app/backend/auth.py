from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os
import secrets
import warnings

_PLACEHOLDER_KEYS = {"your-secret-key", "your_own_generated_secret_key_here", ""}
_env_secret = os.getenv("SECRET_KEY", "")

if _env_secret in _PLACEHOLDER_KEYS:
    # Never sign tokens with a hardcoded/placeholder secret - anyone who
    # read the source (or .env.example) could forge a valid JWT for any
    # username. Generate a random key for this process instead so tokens
    # are at least unforgeable, and warn loudly since it also means
    # existing sessions won't survive a restart until a real SECRET_KEY
    # is set.
    SECRET_KEY = secrets.token_hex(32)
    warnings.warn(
        "SECRET_KEY is not set (or is still the .env.example placeholder). "
        "Using a random key for this process - all sessions will be "
        "invalidated on restart. Set a real SECRET_KEY in .env before "
        "deploying (generate one with: python -c \"import secrets; "
        "print(secrets.token_hex(32))\").",
        stacklevel=2,
    )
else:
    SECRET_KEY = _env_secret

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

class AuthManager:
    """Authentication and authorization manager."""

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password."""
        return pwd_context.hash(password)

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return pwd_context.verify(plain_password, hashed_password)

    @staticmethod
    def create_access_token(data: dict, expires_delta: timedelta = None):
        """Create JWT access token."""
        to_encode = data.copy()
        expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
        to_encode.update({"exp": expire})
        return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    @staticmethod
    def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
        """Verify JWT token."""
        token = credentials.credentials
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username: str = payload.get("sub")
            if username is None:
                raise HTTPException(status_code=401, detail="Invalid token")
            return username
        except JWTError:
            raise HTTPException(status_code=401, detail="Invalid token")

    @staticmethod
    def get_current_user(token: str = Depends(verify_token)):
        """Get current authenticated user."""
        return {"username": token}
