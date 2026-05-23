from __future__ import annotations


def test_metadata_statistics_include_dashboard_counts(stores):
    metadata, _graph, _vector = stores

    metadata.upsert_episode(
        {
            "source": "notes",
            "title": "Planning",
            "summary": "Roadmap planning.",
            "evidence_ids": [],
        }
    )
    metadata.upsert_person_profile_snapshot(
        person_id="person-a",
        profile_text="A profile",
    )
    metadata.upsert_person_profile_snapshot(
        person_id="person-a",
        profile_text="A newer profile",
    )

    stats = metadata.get_statistics()

    assert stats["episode_count"] == 1
    assert stats["person_profile_count"] == 1


def test_async_task_summary_tracks_latest_import_status(stores):
    metadata, _graph, _vector = stores

    first = metadata.create_async_task(
        task_id="task-1",
        task_type="import",
        payload={"mode": "text"},
    )
    assert first["status"] == "queued"
    metadata.update_async_task("task-1", status="running", started_at=1.0)
    metadata.update_async_task("task-1", status="succeeded", result={"count": 2}, finished_at=2.0)

    metadata.create_async_task(
        task_id="task-2",
        task_type="import",
        payload={"mode": "json"},
    )

    summary = metadata.get_async_task_summary(task_type="import")

    assert summary["counts"]["queued"] == 1
    assert summary["counts"]["succeeded"] == 1
    assert summary["latest"]["task_id"] == "task-2"
