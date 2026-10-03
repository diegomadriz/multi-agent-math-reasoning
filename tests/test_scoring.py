from multiagent_math.scoring import extract_final_number, is_correct


def test_extracts_first_marker():
    assert extract_final_number("steps\n#### 42\n#### 7") == "42"


def test_signed_and_decimal():
    assert extract_final_number("#### -3.5") == "-3.5"


def test_missing_marker():
    assert extract_final_number("the answer is 42") is None
    assert extract_final_number("") is None
    assert extract_final_number(None) is None


def test_commas_are_a_known_limitation():
    # "1,000" parses as "1": documented in the README limitations.
    assert extract_final_number("#### 1,000") == "1"


def test_numeric_equality():
    assert is_correct("8.0", "8")
    assert not is_correct("9", "8")
    assert not is_correct(None, "8")
