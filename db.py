from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, text
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
    last_signature = Column(String, default="")
    updated_at = Column(DateTime, default=datetime.utcnow)


# === Flow Event Table (each major inflow/outflow) ===
class FlowEvent(Base):
    __tablename__ = "flow_events"

    id = Column(Integer, primary_key=True, index=True)
    address = Column(String, index=True)
    direction = Column(String)  # "IN" or "OUT"
    sol_amount = Column(Float)  # SOL amount for SOL events; USD value for token events
    usd_value = Column(Float)
    signature = Column(String, index=True)
    slot = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    # coin: 'SOL' for native SOL events, 'JTO'/'WIF'/'FARTCOIN' for SPL token events.
    # Added in v2 — existing rows default to 'SOL' via the migration below.
    coin = Column(String, default="SOL", index=True)


def _migrate_db(conn) -> None:
    """
    Safe incremental migration.  Adds columns that don't yet exist so the
    schema can evolve without dropping and recreating the database.
    SQLite supports ALTER TABLE ADD COLUMN for nullable / default columns.
    """
    # Check existing columns via PRAGMA
    result = conn.execute(text("PRAGMA table_info(flow_events)"))
    existing = {row[1] for row in result}  # row[1] = column name

    if "coin" not in existing:
        conn.execute(text("ALTER TABLE flow_events ADD COLUMN coin VARCHAR(20) DEFAULT 'SOL'"))
        conn.execute(text("UPDATE flow_events SET coin = 'SOL' WHERE coin IS NULL"))
        print("[DB] Migrated flow_events: added 'coin' column (backfilled as 'SOL')")

    # Create index on coin if it doesn't exist (CREATE INDEX IF NOT EXISTS is safe)
    conn.execute(text(
        "CREATE INDEX IF NOT EXISTS ix_flow_events_coin ON flow_events (coin)"
    ))

    wallet_result = conn.execute(text("PRAGMA table_info(wallet_balances)"))
    wallet_existing = {row[1] for row in wallet_result}
    if "last_signature" not in wallet_existing:
        conn.execute(text("ALTER TABLE wallet_balances ADD COLUMN last_signature VARCHAR DEFAULT ''"))
        conn.execute(text("UPDATE wallet_balances SET last_signature = '' WHERE last_signature IS NULL"))
        print("[DB] Migrated wallet_balances: added 'last_signature' column")


# === Initialize database ===
def init_db():
    Base.metadata.create_all(bind=engine)
    with engine.connect() as conn:
        _migrate_db(conn)
        conn.commit()
