"""Phase 0 smoke test — package imports cleanly."""


def test_package_imports():
    import tracking

    assert hasattr(tracking, "PvETracker")
    assert hasattr(tracking, "PvPTracker")
    assert hasattr(tracking, "ActionOwnership")


def test_config_defaults():
    from tracking.config import DEFAULT

    assert DEFAULT.iou_match_threshold == 0.5
    assert DEFAULT.feature_bank_size == 20
    assert "gloves_L" in DEFAULT.reid_region_weights
