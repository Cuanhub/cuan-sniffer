"""Cuan Lab: the machinery that turns an LLM's research into evidence.

The division of labour is the safety model. The agent proposes hypotheses,
writes pre-registrations and implements features. This package -- fixed,
tested, and not the agent's to edit casually -- fetches the data, computes
every statistic, applies the decision rule and keeps the ledger. An
experiment module returns RECORDS, never p-values, so no verdict here was
ever computed by the thing that wanted it to come out a particular way.
"""
