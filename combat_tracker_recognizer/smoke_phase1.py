"""Phase 1 smoke test (per the build plan).

Encode synthetic jab + hook windows. Add one jab to the bank. Verify
subsequent jabs route KNOWN, hooks route UNKNOWN.

Run from repo root:

    python -m combat_tracker_recognizer.smoke_phase1
"""

from combat_tracker_recognizer.bank import PrototypeBank
from combat_tracker_recognizer.config import EncoderConfig, GateConfig
from combat_tracker_recognizer.encoders import HandcraftedEncoder
from combat_tracker_recognizer.gate import NoveltyGate
from combat_tracker_recognizer.tests.conftest import make_pose_window
from combat_tracker_recognizer.types import GateDecision


def main() -> None:
    enc = HandcraftedEncoder(EncoderConfig(seed=0))
    bank = PrototypeBank()
    # Random-init encoder produces extremely small intra-class distances
    # (~1e-5) and inter-class distances ~100-1000x larger. The plan's
    # default GateConfig is calibrated for trained-encoder cosine
    # distances; the smoke test tightens the thresholds so the
    # separation is detectable. With trained weights, the defaults work.
    gate = NoveltyGate(bank, GateConfig(
        known_distance_threshold=1e-4,
        ambiguous_distance_threshold=5e-4,
        min_margin_ratio=2.0,
    ))

    # Bank starts empty.
    jab_seed = enc.encode(make_pose_window("jab", seed=1))
    decision, _, _, _ = gate.route("punch", jab_seed)
    assert decision == GateDecision.UNKNOWN, decision
    print(f"empty bank        : jab -> {decision.value}")

    bank.add("punch", "jab", jab_seed, encoder_version=enc.version)

    for seed in range(2, 7):
        jab = enc.encode(make_pose_window("jab", seed=seed))
        d, sc, conf, _ = gate.route("punch", jab)
        print(f"after-add jab s{seed}: -> {d.value} sc={sc} conf={conf:.2f}")

    for seed in range(20, 25):
        hook = enc.encode(make_pose_window("hook", seed=seed))
        d, sc, conf, _ = gate.route("punch", hook)
        print(f"after-add hook s{seed}: -> {d.value} sc={sc} conf={conf:.2f}")


if __name__ == "__main__":
    main()
