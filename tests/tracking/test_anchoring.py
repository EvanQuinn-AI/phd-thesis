"""Phase 3 tests: IdentityAnchor."""

from tracking.anchoring import IdentityAnchor


def test_anchor_assigns_left_to_slot_1_right_to_slot_2(two_rect_frame, two_rect_bboxes, fake_landmarks_two_people):
    anchor = IdentityAnchor()
    for _ in range(anchor.window):
        anchor.observe(two_rect_frame, list(two_rect_bboxes), list(fake_landmarks_two_people))
    fighters = anchor.finalize()
    assert set(fighters) == {"1", "2"}
    assert fighters["1"].start_region == "left_half"
    assert fighters["2"].start_region == "right_half"
    assert fighters["1"].mean_bbox_centre_x() < fighters["2"].mean_bbox_centre_x()


def test_anchor_populates_feature_banks(two_rect_frame, two_rect_bboxes, fake_landmarks_two_people):
    anchor = IdentityAnchor()
    for _ in range(anchor.window):
        anchor.observe(two_rect_frame, list(two_rect_bboxes), list(fake_landmarks_two_people))
    fighters = anchor.finalize()
    bank_1 = fighters["1"].feature_bank
    bank_2 = fighters["2"].feature_bank
    assert bank_1.has_region("torso")
    assert bank_2.has_region("torso")
    # Different colours; the two banks should NOT match each other's feature.
    from tracking.features import PartExtractor
    ext = PartExtractor()
    f_left = ext.extract(two_rect_frame, two_rect_bboxes[0], fake_landmarks_two_people[0])
    f_right = ext.extract(two_rect_frame, two_rect_bboxes[1], fake_landmarks_two_people[1])
    assert bank_1.score(f_left) < bank_1.score(f_right)


def test_anchor_handles_single_person_frames(two_rect_frame, two_rect_bboxes, fake_landmarks_two_people):
    anchor = IdentityAnchor()
    for i in range(anchor.window):
        if i % 3 == 0:
            anchor.observe(two_rect_frame, [two_rect_bboxes[0]], [fake_landmarks_two_people[0]])
        else:
            anchor.observe(two_rect_frame, list(two_rect_bboxes), list(fake_landmarks_two_people))
    fighters = anchor.finalize()
    assert set(fighters) == {"1", "2"}


def test_anchor_returns_empty_when_never_two_people(two_rect_frame, two_rect_bboxes, fake_landmarks_two_people):
    anchor = IdentityAnchor()
    for _ in range(anchor.window):
        anchor.observe(two_rect_frame, [two_rect_bboxes[0]], [fake_landmarks_two_people[0]])
    fighters = anchor.finalize()
    assert fighters == {}


def test_anchor_window_is_finite(two_rect_frame, two_rect_bboxes, fake_landmarks_two_people):
    anchor = IdentityAnchor()
    for _ in range(anchor.window * 3):
        anchor.observe(two_rect_frame, list(two_rect_bboxes), list(fake_landmarks_two_people))
    assert anchor.is_full
    assert len(anchor._frames) == anchor.window
