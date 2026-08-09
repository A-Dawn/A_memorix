from pathlib import Path

from a_memorix.core.runtime.sdk_memory_kernel import SDKMemoryKernel
from a_memorix.core.utils.profile_evidence import profile_evidence_type_from_source, profile_relation_content


def test_profile_evidence_type_from_source_uses_metadata_first() -> None:
    assert profile_evidence_type_from_source("manual:1", {"source_type": "person_fact"}) == "person_fact"
    assert profile_evidence_type_from_source("manual:1", {"source_type": "chat_summary"}) == "chat_summary"


def test_profile_evidence_type_from_source_falls_back_to_source_prefix() -> None:
    assert profile_evidence_type_from_source("person_fact:person-1") == "person_fact"
    assert profile_evidence_type_from_source("chat_summary:session-1") == "chat_summary"
    assert profile_evidence_type_from_source("manual:note-1") == "paragraph"


def test_profile_relation_content_formats_complete_relation() -> None:
    assert (
        profile_relation_content({"subject": "Alice", "predicate": "喜欢", "object": "绿茶"}) == "Alice -[喜欢]-> 绿茶"
    )
    assert profile_relation_content({"subject": "Alice", "object": "绿茶"}) == "Alice 绿茶"


def test_profile_evidence_kernel_and_service_compatibility_wrappers() -> None:
    kernel = SDKMemoryKernel(data_dir=Path.cwd(), config={})
    service = kernel._profile_admin_service

    relation = {"subject": "Alice", "predicate": "喜欢", "object": "绿茶"}
    assert kernel._profile_evidence_type_from_source("chat_summary:session-1") == "chat_summary"
    assert service._profile_evidence_type_from_source("person_fact:person-1") == "person_fact"
    assert kernel._profile_relation_content(relation) == "Alice -[喜欢]-> 绿茶"
    assert service._profile_relation_content(relation) == "Alice -[喜欢]-> 绿茶"
