from rent_agent.tools.lawd_code import find_lawd_codes


def test_exact_gu_name():
    assert find_lawd_codes("강남구") == [("서울특별시 강남구", "11680")]


def test_partial_match_returns_multiple():
    results = find_lawd_codes("서울")
    assert len(results) == 25
    assert ("서울특별시 종로구", "11110") in results


def test_dong_name_not_supported_returns_empty():
    assert find_lawd_codes("역삼동") == []


def test_whitespace_tolerant():
    assert find_lawd_codes(" 강남 ") == [("서울특별시 강남구", "11680")]


def test_multi_token_query_requires_all_tokens():
    # 사용자는 "서울 강남구", "성남 분당"처럼 띄어 쓴다 — 모든 토큰이 이름에 포함되면 매칭
    assert find_lawd_codes("서울 강남구") == [("서울특별시 강남구", "11680")]
    assert find_lawd_codes("성남 분당") == [("경기도 성남시 분당구", "41135")]
    assert find_lawd_codes("부산 중구") == []


def test_city_with_districts_returns_all_districts():
    assert len(find_lawd_codes("성남시")) == 3
    assert len(find_lawd_codes("부천")) == 3
    assert len(find_lawd_codes("화성")) == 4


def test_codes_are_unique_five_digits():
    from rent_agent.tools.lawd_code import LAWD_CODES

    codes = list(LAWD_CODES.values())
    assert len(codes) == len(set(codes)) == 50
    assert all(len(c) == 5 and c.isdigit() for c in codes)
