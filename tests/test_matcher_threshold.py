from a_memorix.core.utils.matcher import HAS_AHOCORASICK_RS, AhoCorasick


def test_native_matcher_threshold_keeps_small_pattern_sets_on_python_path() -> None:
    matcher = AhoCorasick(native_min_patterns=10)
    matcher.add_pattern("艾宝")
    matcher.add_pattern("稀疏检索")

    matcher.build()

    assert matcher._native_matcher is None
    assert matcher.find_all("艾宝 触发 稀疏检索") == {"艾宝": 1, "稀疏检索": 1}


def test_native_matcher_can_enable_when_threshold_is_met() -> None:
    matcher = AhoCorasick(native_min_patterns=1)
    matcher.add_pattern("艾宝")

    matcher.build()

    if HAS_AHOCORASICK_RS:
        assert matcher._native_matcher is not None
    assert matcher.find_all("艾宝 艾宝") == {"艾宝": 2}
