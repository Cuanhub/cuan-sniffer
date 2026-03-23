from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

# SQLite database file stored locally
DATABASE_URL = "sqlite:///sol_flow.db"

# Create database engine
engine = create_engine(DATABASE_URL, echo=False, future=True)

# Session factory for database operations
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

Base = declarative_base()

# === Wallet Balance Table ===
class WalletBalance(Base):
    __tablename__ = "wallet_balances"

    id = Column(Integer, primary_key=True, index=True)
    address = Column(String, unique=True, index=True)
    sol_balance = Column(Float, default=0.0)
    updated_at = Column(DateTime, default=datetime.utcnow)


# === Flow Event Table (each major inflow/outflow) ===
class FlowEvent(Base):
    __tablename__ = "flow_events"

    id = Column(Integer, primary_key=True, index=True)
    address = Column(String, index=True)
    direction = Column(String)  # "IN" or "OUT"
    sol_amount = Column(Float)
    usd_value = Column(Float)
    signature = Column(String, index=True)
    slot = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)


# === Initialize database ===
def init_db():
    Base.metadata.create_all(bind=engine)
