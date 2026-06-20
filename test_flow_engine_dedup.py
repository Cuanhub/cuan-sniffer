import unittest

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db import Base, FlowEvent, WalletBalance
from engine import SolFlowEngine


class TestFlowEngineDedup(unittest.TestCase):
    def setUp(self):
        db_engine = create_engine("sqlite:///:memory:", future=True)
        Base.metadata.create_all(bind=db_engine)
        self.Session = sessionmaker(bind=db_engine, autoflush=False, autocommit=False)

    def test_flow_event_insert_is_idempotent_per_wallet_signature_coin_direction(self):
        session = self.Session()
        engine = SolFlowEngine(["wallet_1"])
        try:
            first = engine._add_flow_event_once(
                session,
                address="wallet_1",
                direction="OUT",
                amount=123.0,
                usd_value=123.0,
                signature="sig_1",
                slot=42,
                coin="JTO",
            )
            session.commit()

            second = engine._add_flow_event_once(
                session,
                address="wallet_1",
                direction="OUT",
                amount=123.0,
                usd_value=123.0,
                signature="sig_1",
                slot=42,
                coin="JTO",
            )
            session.commit()

            self.assertTrue(first)
            self.assertFalse(second)
            self.assertEqual(session.query(FlowEvent).count(), 1)
        finally:
            session.close()

    def test_wallet_balance_persists_last_signature(self):
        session = self.Session()
        try:
            wallet = WalletBalance(
                address="wallet_1",
                sol_balance=1.0,
                last_signature="sig_latest",
            )
            session.add(wallet)
            session.commit()

            loaded = session.query(WalletBalance).filter_by(address="wallet_1").one()
            self.assertEqual(loaded.last_signature, "sig_latest")
        finally:
            session.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
