import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aic_system.eval.local_scorer import GTRow, score_query  # noqa: E402


def test_kis_scoring_matches_worked_example_from_spec():
    # From the competition doc's worked example:
    # rank:      1   2   3   4   5   ...   20 ... 100
    # R-Score:   0   0   0   1   0   ...   0  ...  0
    # R@1=0 R@5=1 R@20=1 R@50=1 R@100=1 -> score = 0.8
    gt = GTRow(query_id="q1", task="kis", video="L00_V000", interval=(1000, 1010))
    rows = []
    for i in range(1, 101):
        if i == 4:
            rows.append(["L00_V000", "1005"])  # inside interval -> correct
        else:
            rows.append(["L00_V000", "999999"])  # outside interval -> wrong
    result = score_query(rows, gt)
    assert result["R@1"] == 0.0
    assert result["R@5"] == 1.0
    assert result["R@20"] == 1.0
    assert result["R@50"] == 1.0
    assert result["R@100"] == 1.0
    assert result["score"] == 0.8


def test_kis_wrong_video_never_scores():
    gt = GTRow(query_id="q1", task="kis", video="L00_V000", interval=(1000, 1010))
    rows = [["L99_V999", "1005"]] * 100
    result = score_query(rows, gt)
    assert result["score"] == 0.0


def test_qa_requires_video_frame_and_answer():
    gt = GTRow(
        query_id="q3", task="qa", video="L01_V028",
        interval=(3400, 3500), answer="5",
    )
    # correct video+frame but wrong answer -> 0
    rows = [["L01_V028", "3450", "6"]]
    assert score_query(rows, gt)["R@1"] == 0.0

    # everything correct -> 1
    rows = [["L01_V028", "3450", "5"]]
    assert score_query(rows, gt)["R@1"] == 1.0


def test_qa_answer_match_is_whitespace_and_case_insensitive():
    gt = GTRow(
        query_id="q3", task="qa", video="L01_V028",
        interval=(3400, 3500), answer="Năm người",
    )
    rows = [["L01_V028", "3450", "  năm   người  "]]
    assert score_query(rows, gt)["R@1"] == 1.0


def test_trake_fractional_scoring_matches_worked_example():
    # From the spec's worked example: 3/4 events correct -> RScore = 0.75
    gt = GTRow(
        query_id="q4", task="trake", video="L10_V001",
        intervals=[(1000, 1200), (2000, 2200), (3000, 3200), (4000, 4200)],
    )
    rows = [["L10_V001", "1100", "2100", "3500", "4100"]]
    result = score_query(rows, gt)
    assert result["R@1"] == 0.75


def test_trake_wrong_video_zeroes_regardless_of_frame_quality():
    gt = GTRow(
        query_id="q4", task="trake", video="L10_V001",
        intervals=[(1000, 1200), (2000, 2200)],
    )
    rows = [["L99_V999", "1100", "2100"]]
    assert score_query(rows, gt)["R@1"] == 0.0


def test_trake_wrong_event_count_scores_zero_not_crash():
    gt = GTRow(
        query_id="q4", task="trake", video="L10_V001",
        intervals=[(1000, 1200), (2000, 2200), (3000, 3200)],
    )
    rows = [["L10_V001", "1100", "2100"]]  # only 2 frames, expected 3
    result = score_query(rows, gt)
    assert result["R@1"] == 0.0
