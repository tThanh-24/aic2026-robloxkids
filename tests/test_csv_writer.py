import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402
from aic_system.io.csv_writer import (  # noqa: E402
    KISCandidate,
    QACandidate,
    TrakeCandidate,
    write_kis_csv,
    write_qa_csv,
    write_trake_csv,
)


def test_kis_basic_format(tmp_path):
    out = tmp_path / "query-1-kis.csv"
    write_kis_csv(out, [KISCandidate("L00_V000.mp4", 1234), KISCandidate("L01_V028", 25300)])
    text = out.read_text(encoding="utf-8")
    lines = text.splitlines()
    assert lines[0] == "L00_V000,1234"
    assert lines[1] == "L01_V028,25300"
    # no header, no extension leakage
    assert ".mp4" not in text


def test_kis_truncates_to_100(tmp_path):
    out = tmp_path / "query-2-kis.csv"
    cands = [KISCandidate(f"L00_V{i:03d}", i) for i in range(150)]
    with pytest.warns(UserWarning):
        write_kis_csv(out, cands, query_id="query-2-kis")
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 100


def test_qa_answer_always_quoted(tmp_path):
    out = tmp_path / "query-3-qa.csv"
    write_qa_csv(out, [QACandidate("L01_V028", 3450, "5")])
    text = out.read_text(encoding="utf-8")
    assert text.startswith('L01_V028,3450,"5"')


def test_qa_answer_with_comma_is_quoted(tmp_path):
    out = tmp_path / "query-3-qa.csv"
    write_qa_csv(out, [QACandidate("L01_V028", 3450, "Có 3 người, bao gồm nam và nữ")])
    text = out.read_text(encoding="utf-8")
    assert '"Có 3 người, bao gồm nam và nữ"' in text


def test_qa_answer_with_embedded_quotes_doubled(tmp_path):
    out = tmp_path / "query-3-qa.csv"
    write_qa_csv(out, [QACandidate("L01_V028", 3450, 'Anh ấy nói "Xin chào"')])
    text = out.read_text(encoding="utf-8")
    assert '"Anh ấy nói ""Xin chào"""' in text
    # must not be quadruple-quoted (the double-quoting bug this module guards against)
    assert '""""' not in text


def test_qa_answer_over_100_chars_truncated_with_warning(tmp_path):
    out = tmp_path / "query-3-qa.csv"
    long_answer = "x" * 150
    with pytest.warns(UserWarning):
        write_qa_csv(out, [QACandidate("L01_V028", 1, long_answer)], query_id="q")
    text = out.read_text(encoding="utf-8")
    # 100 x's plus the two quote chars
    assert '"' + "x" * 100 + '"' in text


def test_trake_multiple_frame_columns(tmp_path):
    out = tmp_path / "query-4-trake.csv"
    write_trake_csv(out, [TrakeCandidate("L10_V001", [1200, 1850, 2100, 2450])])
    text = out.read_text(encoding="utf-8")
    assert text.startswith("L10_V001,1200,1850,2100,2450")


def test_video_extension_stripped_for_all_variants(tmp_path):
    out = tmp_path / "query-1-kis.csv"
    write_kis_csv(out, [KISCandidate("L00_V000.mp4", 1)])
    text = out.read_text(encoding="utf-8")
    assert text.startswith("L00_V000,1")
