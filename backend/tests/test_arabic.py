from app.services.arabic import analyze_arabic, is_duplicate, normalize_arabic


def test_arabic_normalization_preserves_facts() -> None:
    text = "أَعْلَنَتِ الوزارةُ أن العدد ٥٠٠ ـ حالة؟"
    normalized = normalize_arabic(text)
    assert "500" in normalized
    assert "الوزاره" not in normalized  # taa marbuta is intentionally preserved
    assert "الوزارة" in normalized
    assert "ـ" not in normalized
    assert normalized.endswith("?")


def test_analysis_extracts_dates_numbers_and_urls() -> None:
    result = analyze_arabic("نُشر في ٤/٨/٢٠٢٦ ووصل إلى ٥٠٠: https://example.org/x")
    assert "4/8/2026" in result.dates
    assert "500" in result.numbers
    assert result.urls == ("https://example.org/x",)
    assert len(result.content_hash) == 64


def test_duplicate_detection_tolerates_arabic_variants() -> None:
    assert is_duplicate("إجمالي الإصابات ٥٠٠ حالة", "اجمالي الاصابات 500 حالة")
