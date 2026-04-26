"""Gate tests."""

import numpy as np

from combat_tracker_recognizer.bank import PrototypeBank
from combat_tracker_recognizer.config import GateConfig
from combat_tracker_recognizer.gate import NoveltyGate
from combat_tracker_recognizer.types import GateDecision


def _vec(seed, d=8):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(d).astype(np.float32)
    return v / np.linalg.norm(v)


def test_empty_bank_routes_unknown():
    gate = NoveltyGate(PrototypeBank(), GateConfig())
    decision, sc, conf, top = gate.route("punch", _vec(0))
    assert decision == GateDecision.UNKNOWN
    assert sc is None
    assert top == []


def test_exact_match_routes_known():
    bank = PrototypeBank()
    e = _vec(0)
    bank.add("punch", "jab", e)
    bank.add("punch", "hook", _vec(99))  # something far away to satisfy margin
    gate = NoveltyGate(bank, GateConfig())
    decision, sc, conf, top = gate.route("punch", e)
    assert decision == GateDecision.KNOWN
    assert sc == "jab"
    assert conf > 0.9


def test_two_close_matches_route_ambiguous():
    bank = PrototypeBank()
    e = _vec(0)
    near = e + 0.1 * _vec(1)
    near = near / np.linalg.norm(near)
    bank.add("punch", "jab", e)
    bank.add("punch", "lead_jab", near)
    gate = NoveltyGate(bank, GateConfig(min_margin_ratio=2.0))
    probe = (e + near) / 2.0
    probe = probe / np.linalg.norm(probe)
    decision, sc, conf, top = gate.route("punch", probe)
    assert decision in {GateDecision.AMBIGUOUS, GateDecision.UNKNOWN}


def test_zero_vector_routes_noise():
    bank = PrototypeBank()
    bank.add("punch", "jab", _vec(0))
    gate = NoveltyGate(bank, GateConfig(noise_magnitude_floor=0.5))
    decision, sc, _, _ = gate.route("punch", np.zeros(8, dtype=np.float32))
    assert decision == GateDecision.NOISE


def test_thresholds_respected_at_boundary():
    bank = PrototypeBank()
    bank.add("punch", "jab", _vec(0))
    # Force a known threshold of 0.05 so any tiny perturbation falls outside.
    cfg = GateConfig(known_distance_threshold=0.05, ambiguous_distance_threshold=0.5)
    gate = NoveltyGate(bank, cfg)
    e_far = _vec(50)
    decision, _, _, _ = gate.route("punch", e_far)
    assert decision in {GateDecision.AMBIGUOUS, GateDecision.UNKNOWN}
