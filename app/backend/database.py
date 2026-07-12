from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, JSON
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import os
import uuid

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./aimaster.db")

Base = declarative_base()
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    conversation_id = Column(String, index=True)
    message = Column(Text)
    response = Column(Text)
    model_used = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    meta_data = Column(JSON, default={})

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

class DatabaseManager:
    """Database connection manager."""

    @staticmethod
    async def connect():
        """Create database tables."""
        Base.metadata.create_all(bind=engine)

    @staticmethod
    async def disconnect():
        """Close database connections."""
        engine.dispose()

    @staticmethod
    def get_db():
        """Get database session."""
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

    # ---------------- Users ----------------

    @staticmethod
    def create_user(username, email, hashed_password):
        db = SessionLocal()
        try:
            user = User(username=username, email=email, hashed_password=hashed_password)
            db.add(user)
            db.commit()
            db.refresh(user)
            return user
        finally:
            db.close()

    @staticmethod
    def get_user_by_username(username):
        db = SessionLocal()
        try:
            return db.query(User).filter(User.username == username).first()
        finally:
            db.close()

    # ---------------- Conversations ----------------

    @staticmethod
    def save_conversation(user_id, message, response, model_used, conversation_id=None, metadata=None):
        """Save conversation to database. Generates a conversation_id if none given."""
        db = SessionLocal()
        try:
            conv = Conversation(
                user_id=user_id,
                conversation_id=conversation_id or uuid.uuid4().hex[:12],
                message=message,
                response=response,
                model_used=model_used,
                meta_data=metadata or {}
            )
            db.add(conv)
            db.commit()
            db.refresh(conv)
            return conv.conversation_id
        finally:
            db.close()

    @staticmethod
    def list_conversations(user_id):
        """One row per conversation_id, with a title from the first message,
        ordered by most recently active."""
        db = SessionLocal()
        try:
            rows = (
                db.query(Conversation)
                .filter(Conversation.user_id == user_id)
                .order_by(Conversation.timestamp.asc())
                .all()
            )
            grouped = {}
            for r in rows:
                cid = r.conversation_id
                if cid not in grouped:
                    grouped[cid] = {
                        "conversation_id": cid,
                        "title": (r.message[:40] + "…") if len(r.message) > 40 else r.message,
                        "last_active": r.timestamp,
                    }
                else:
                    grouped[cid]["last_active"] = r.timestamp
            result = list(grouped.values())
            result.sort(key=lambda c: c["last_active"], reverse=True)
            for r in result:
                r["last_active"] = r["last_active"].isoformat()
            return result
        finally:
            db.close()

    @staticmethod
    def get_conversation_messages(user_id, conversation_id):
        db = SessionLocal()
        try:
            rows = (
                db.query(Conversation)
                .filter(Conversation.user_id == user_id, Conversation.conversation_id == conversation_id)
                .order_by(Conversation.timestamp.asc())
                .all()
            )
            return [
                {
                    "message": r.message,
                    "response": r.response,
                    "model_used": r.model_used,
                    "timestamp": r.timestamp.isoformat(),
                }
                for r in rows
            ]
        finally:
            db.close()
