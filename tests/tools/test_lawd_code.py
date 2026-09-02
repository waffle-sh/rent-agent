from rent_agent.tools.lawd_code import find_lawd_codes


def test_exact_gu_name():
    assert find_lawd_codes("강남구") == [("서울특별시 강남구", "11680")]


def test_partial_match_returns_multiple():
    results = find_lawd_codes("서울")
    assert len(results) == 25
    assert ("서울특별시 종로구", "11110") in results


def test_dong_name_not_supported_returns_empty():
    assert find_lawd_codes("역삼동") == []


def test_whitespace_and_suffix_tolerant():
    assert find_lawd_codes(" 강남 ") == [("서울특별시 강남구", "11680")]
