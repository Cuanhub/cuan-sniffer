"""Labelling many calls from one scan.

cmd_label re-scans the universe per label -- 45 coins across two timeframes,
about 90 venue requests EACH. Sixteen verdicts in one sitting cost ~1,440
requests against an endpoint that has been returning rate limits, which is how
this got written: the batch was done by hand as a one-off script first.

Matched on (coin, level) rather than episode_id deliberately. An id carries its
bar timestamp, so the ones a board prints go stale as soon as the next bar
opens -- which is exactly when somebody sits down to label them.
"""

import csv
import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from research.levels import call as C
from research.levels import storage


def ep(coin, level, tf="1H"):
    return {"episode_id": f"{coin}-{tf}_1790827200000_{level:g}",
            "coin": coin, "level_price": level, "suggested_entry": level * 1.003,
            "suggested_stop": level * 0.99, "suggested_tp": level * 1.08,
            "suggested_rr": 3.0, "timeframe": tf.lower(), "_px": level * 1.05}


class Batch(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(self.id().replace(".", "_") + "_batch.txt")
        self.labels, self.episodes = [], []
        self._al, self._ae, self._fc, self._su, self._cl, self._lb = (
            storage.append_label, storage.append_episode, C.find_candidates,
            C.select_universe, C.HyperliquidReadOnlyClient, C._labelled)
        self.scans = 0

        def fake_find(client, universe, **kw):
            self.scans += 1
            return [ep("AAVE", 142.84), ep("AAVE", 135.52), ep("HYPE", 86.747),
                    ep("ZEC", 1392.7)]

        storage.append_label = lambda row: (self.labels.append(row) or True)
        storage.append_episode = lambda row: (self.episodes.append(row) or True)
        C.find_candidates = fake_find
        C.select_universe = lambda c: [{"symbol": "AAVE"}]
        C.HyperliquidReadOnlyClient = lambda *a, **k: object()
        C._labelled = lambda: {}

    def tearDown(self):
        (storage.append_label, storage.append_episode, C.find_candidates,
         C.select_universe, C.HyperliquidReadOnlyClient, C._labelled) = (
            self._al, self._ae, self._fc, self._su, self._cl, self._lb)
        self.tmp.unlink(missing_ok=True)

    def run_batch(self, text):
        self.tmp.write_text(text)
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = C.cmd_batch(str(self.tmp))
        return rc, buf.getvalue()

    # -- the point of the whole thing --------------------------------------

    def test_many_labels_cost_exactly_one_scan(self):
        rc, _ = self.run_batch("AAVE 142.84 take\nAAVE 135.52 take\n"
                               "HYPE 86.747 skip\nZEC 1392.7 take\n")
        self.assertEqual(rc, 0)
        self.assertEqual(len(self.labels), 4)
        self.assertEqual(self.scans, 1, "a batch must not re-scan per label")

    def test_the_episode_snapshot_is_written_before_the_label(self):
        """The label has to have something to join to."""
        self.run_batch("AAVE 142.84 take\n")
        self.assertEqual(len(self.episodes), 1)
        self.assertEqual(self.labels[0]["episode_id"],
                         self.episodes[0]["episode_id"])

    def test_private_keys_are_stripped_from_the_snapshot(self):
        self.run_batch("AAVE 142.84 take\n")
        self.assertFalse([k for k in self.episodes[0] if k.startswith("_")])

    # -- matching -----------------------------------------------------------

    def test_a_level_read_off_a_chart_still_matches(self):
        """142.8 for 142.84 must work; an operator reads these off a chart."""
        self.run_batch("AAVE 142.8 take\n")
        self.assertEqual(len(self.labels), 1)

    def test_a_price_that_is_a_different_level_is_refused_not_guessed(self):
        rc, out = self.run_batch("AAVE 120 take\n")
        self.assertEqual(rc, 1)
        self.assertEqual(self.labels, [])
        self.assertIn("within 2%", out)
        self.assertIn("nearest", out)

    def test_an_unknown_coin_is_refused(self):
        rc, out = self.run_batch("NOTACOIN 100 take\n")
        self.assertEqual(rc, 1)
        self.assertIn("no levels detected", out)

    def test_a_match_that_moved_is_reported(self):
        _, out = self.run_batch("ZEC 1379 take\n")
        self.assertIn("matched 1392.7", out)
        self.assertIn("+1.0%", out)

    def test_one_bad_line_does_not_discard_the_good_ones(self):
        rc, _ = self.run_batch("AAVE 142.84 take\nAAVE 999 take\nHYPE 86.747 skip\n")
        self.assertEqual(rc, 1, "an unmatched line must be reported in the exit code")
        self.assertEqual(len(self.labels), 2, "the matched lines must still land")

    # -- parsing ------------------------------------------------------------

    def test_notes_run_to_the_end_of_the_line(self):
        self.run_batch("AAVE 142.84 take  their blue zone, from the 1h chart\n")
        self.assertEqual(self.labels[0]["note"],
                         "their blue zone, from the 1h chart")

    def test_blanks_and_comments_are_ignored(self):
        rc, _ = self.run_batch("# a header\n\n  \nAAVE 142.84 take\n")
        self.assertEqual(rc, 0)
        self.assertEqual(len(self.labels), 1)

    def test_coin_and_verdict_are_case_insensitive(self):
        self.run_batch("aave 142.84 TAKE\n")
        self.assertEqual(self.labels[0]["verdict"], "take")

    def test_all_three_verdicts_are_accepted(self):
        self.run_batch("AAVE 142.84 take\nAAVE 135.52 skip\nHYPE 86.747 maybe\n")
        self.assertEqual([r["verdict"] for r in self.labels],
                         ["take", "skip", "maybe"])

    def test_a_bad_verdict_stops_before_any_venue_call(self):
        """Validation happens first, so a typo cannot cost a 90-call scan."""
        rc, out = self.run_batch("AAVE 142.84 buy\n")
        self.assertEqual(rc, 1)
        self.assertEqual(self.scans, 0, "it scanned despite a bad verdict")
        self.assertIn("take/skip/maybe", out)

    def test_a_malformed_line_stops_before_any_venue_call(self):
        for bad in ("AAVE take\n", "AAVE notanumber take\n"):
            self.scans = 0
            rc, _ = self.run_batch(bad)
            self.assertEqual(rc, 1)
            self.assertEqual(self.scans, 0)

    def test_an_empty_file_does_nothing(self):
        rc, out = self.run_batch("# only a comment\n")
        self.assertEqual(rc, 1)
        self.assertEqual(self.scans, 0)
        self.assertIn("nothing to label", out)


class SharedWithTheSingleLabelPath(unittest.TestCase):
    def test_both_paths_write_through_the_same_recorder(self):
        """So a batch cannot drift from a single label -- the same mistake the
        stop rule and the target rule each made by being written twice."""
        import inspect
        for fn in (C.cmd_batch, C.cmd_label):
            self.assertIn("_record(", inspect.getsource(fn))


if __name__ == "__main__":
    unittest.main()
