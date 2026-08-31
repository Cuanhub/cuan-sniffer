"""
Tests for smc_zones.py — Order Block / FVG detection and invalidation.

No test file existed for this module before (2026-08-25). Covers the zone
invalidation fix: an Order Block or FVG must die the moment price CLOSES
beyond its far edge (standard SMC "broken OB" / "filled FVG" rule), not
just after max_zone_age_bars.
"""

import unittest

import pandas as pd

from smc_zones import add_smc_zones


def _bar(o, h, l, c, atr=1.0):
    return {"open": o, "high": h, "low": l, "close": c, "atr_14": atr}


class TestOrderBlockInvalidation(unittest.TestCase):

    def _build_bull_ob_sequence(self):
        rows = [_bar(100, 100.3, 99.7, 100.0) for _ in range(5)]        # 0-4: quiet
        rows.append(_bar(100.0, 100.2, 98.8, 99.0))                     # 5: bearish OB source candle
        rows.append(_bar(99.0, 103.2, 98.9, 103.0))                     # 6: bullish displacement
        rows.append(_bar(102.0, 102.6, 101.5, 102.3))                   # 7: holds above the zone
        return rows

    def test_bull_ob_forms_after_displacement(self):
        rows = self._build_bull_ob_sequence()
        df = add_smc_zones(pd.DataFrame(rows))
        self.assertFalse(pd.isna(df["bull_ob_low"].iloc[6]))
        self.assertAlmostEqual(float(df["bull_ob_low"].iloc[6]), 98.8, places=3)
        self.assertAlmostEqual(float(df["bull_ob_high"].iloc[6]), 100.2, places=3)
        # Still alive one bar later, well within max_zone_age_bars.
        self.assertFalse(pd.isna(df["bull_ob_low"].iloc[7]))

    def test_bull_ob_invalidated_on_close_below_low(self):
        rows = self._build_bull_ob_sequence()
        rows.append(_bar(99.0, 99.2, 97.8, 97.9))  # 8: closes below OB low (98.8) -- broken
        rows.append(_bar(97.9, 98.0, 97.0, 97.5))  # 9: well after the break
        df = add_smc_zones(pd.DataFrame(rows))

        self.assertFalse(pd.isna(df["bull_ob_low"].iloc[7]), "zone must exist before the break")
        self.assertTrue(pd.isna(df["bull_ob_low"].iloc[8]), "zone must die on the SAME bar its close breaks it")
        self.assertTrue(pd.isna(df["bull_ob_low"].iloc[9]), "zone must stay dead, not reappear")
        self.assertEqual(int(df["in_bull_ob"].iloc[9]), 0)
        self.assertEqual(int(df["ob_bull"].iloc[9]), 0)

    def test_bear_ob_invalidated_on_close_above_high(self):
        rows = [_bar(100, 100.3, 99.7, 100.0) for _ in range(5)]
        rows.append(_bar(100.0, 101.2, 99.8, 101.0))   # 5: bullish OB source candle
        rows.append(_bar(101.0, 101.1, 96.8, 97.0))    # 6: bearish displacement
        rows.append(_bar(98.0, 98.5, 97.5, 98.2))      # 7: holds below the zone
        rows.append(_bar(99.0, 101.5, 98.9, 101.3))    # 8: closes above OB high (101.2) -- broken
        df = add_smc_zones(pd.DataFrame(rows))

        self.assertFalse(pd.isna(df["bear_ob_high"].iloc[7]), "zone must exist before the break")
        self.assertTrue(pd.isna(df["bear_ob_high"].iloc[8]), "zone must die on the SAME bar its close breaks it")


class TestFvgInvalidation(unittest.TestCase):

    def _build_bull_fvg_sequence(self):
        # Bullish FVG: low[i] > high[i-2]. Bar 0 high=100.2, bar 2 low=102.0
        # -> gap size 1.8, well above min_fvg_atr_frac(0.10)*atr(1.0)=0.10.
        rows = [
            _bar(99.8, 100.2, 99.6, 100.0),   # 0
            _bar(100.5, 101.5, 100.3, 101.2),  # 1
            _bar(102.0, 102.8, 102.0, 102.5),  # 2: low=102.0 > high[0]=100.2 -> bull FVG at bar 2
        ]
        return rows

    def test_bull_fvg_forms(self):
        rows = self._build_bull_fvg_sequence()
        df = add_smc_zones(pd.DataFrame(rows))
        self.assertFalse(pd.isna(df["bull_fvg_low"].iloc[2]))
        self.assertAlmostEqual(float(df["bull_fvg_low"].iloc[2]), 100.2, places=3)
        self.assertAlmostEqual(float(df["bull_fvg_high"].iloc[2]), 102.0, places=3)

    def test_bull_fvg_invalidated_when_fully_filled(self):
        rows = self._build_bull_fvg_sequence()
        rows.append(_bar(102.0, 102.2, 101.0, 101.5))  # 3: still above gap low (100.2), alive
        rows.append(_bar(101.5, 101.6, 99.0, 99.5))    # 4: closes below gap low (100.2) -- filled
        df = add_smc_zones(pd.DataFrame(rows))

        self.assertFalse(pd.isna(df["bull_fvg_low"].iloc[3]), "gap must still exist before being filled")
        self.assertTrue(pd.isna(df["bull_fvg_low"].iloc[4]), "gap must die on the SAME bar it's fully filled")
        self.assertEqual(int(df["in_bull_fvg"].iloc[4]), 0)
        self.assertEqual(int(df["fvg_bull"].iloc[4]), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
