from pathlib import Path

from examscrapers.raynormaritime.scrape_questions import RaynorMaritimeScraper


def _make_html(
    marker: str, *, question_text: str = "What is the correct action?"
) -> str:
    return f"""
    <html>
      <body>
        <h3>{marker}</h3>
        <table>
          <tr><td colspan="3"><b>{question_text}</b></td></tr>
          <tr><td><label><input type="radio" name="CheckAns" value="A"><b>A.</b> First option</label></td></tr>
          <tr><td><label><input type="radio" name="CheckAns" value="B" checked><b>B.</b> Second option</label></td></tr>
          <tr><td><label><input type="radio" name="CheckAns" value="C"><b>C.</b> Third option</label></td></tr>
          <tr><td><label><input type="radio" name="CheckAns" value="D"><b>D.</b> Fourth option</label></td></tr>
        </table>
      </body>
    </html>
    """


def _make_blank_html(marker: str) -> str:
    return f"""
    <html>
      <body>
        <h3>{marker}</h3>
        <table>
          <tr><td><label><input type="radio" name="CheckAns" value="A"><b>A.</b></label></td></tr>
          <tr><td><label><input type="radio" name="CheckAns" value="B"><b>B.</b></label></td></tr>
          <tr><td><label><input type="radio" name="CheckAns" value="C"><b>C.</b></label></td></tr>
          <tr><td><label><input type="radio" name="CheckAns" value="D"><b>D.</b></label></td></tr>
        </table>
      </body>
    </html>
    """


def test_parse_question_discards_blank_pages(tmp_path: Path) -> None:
    scraper = RaynorMaritimeScraper(output_file=tmp_path / "out.json")

    parsed = scraper.parse_question(_make_blank_html("INLAND ONLY"), "0003")

    assert parsed is None


def test_parse_question_classifies_inland(tmp_path: Path) -> None:
    scraper = RaynorMaritimeScraper(output_file=tmp_path / "out.json")

    parsed = scraper.parse_question(_make_html("INLAND ONLY"), "0004")

    assert parsed is not None
    assert parsed.zone_slug == "inland"
    assert parsed.question_text.startswith("US INLAND ONLY ")


def test_parse_question_classifies_both(tmp_path: Path) -> None:
    scraper = RaynorMaritimeScraper(output_file=tmp_path / "out.json")

    parsed = scraper.parse_question(_make_html("BOTH INTERNATIONAL AND INLAND"), "4585")

    assert parsed is not None
    assert parsed.zone_slug == "both"
    assert parsed.question_text.startswith("BOTH INTERNATIONAL & US INLAND ")


def test_parse_question_classifies_international_only_and_counts(
    tmp_path: Path,
) -> None:
    scraper = RaynorMaritimeScraper(output_file=tmp_path / "out.json")

    parsed = scraper.parse_question(_make_html("INTERNATIONAL ONLY"), "7777")

    assert parsed is not None
    assert parsed.zone_slug == "international-only"
    assert parsed.question_text.startswith("INTERNATIONAL ONLY ")
    assert scraper.international_only_count == 1


def test_parse_question_extracts_checked_answer_and_cleans_option_text(
    tmp_path: Path,
) -> None:
    scraper = RaynorMaritimeScraper(output_file=tmp_path / "out.json")

    parsed = scraper.parse_question(_make_html("INLAND ONLY"), "0004")

    assert parsed is not None
    assert parsed.correct_option_id == "B"
    assert parsed.answers["A"] == "First option"
    assert parsed.answers["B"] == "Second option"
