

def test_accept_alignment_recovery_removes_stale_turn_artifacts(
    tmp_path,
):
    from voice_dataset.review import (
        accept_alignment_recovery,
    )
    from voice_dataset.schema import TurnRecord
    from voice_dataset.storage import DatasetStorage

    storage = DatasetStorage(tmp_path)

    storage.add_turn(
        TurnRecord(
            id="turn_000001",
            source_id="source_001",
            source_start=1.0,
            source_end=2.0,
            source_regions=[
                "region_000001",
            ],
            transcript="test",
            metadata={
                "alignment_evidence": {
                    "status": "invalid",
                    "recovery": {
                        "status": "suggested",
                        "source_start": 0.8,
                        "source_end": 2.2,
                        "region_ids": [
                            "region_000001",
                        ],
                    },
                },
            },
        )
    )

    artifact_dir = (
        storage.root
        / "turns"
        / "turn_000001"
    )

    embedding_dir = (
        artifact_dir
        / "embeddings"
    )

    embedding_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (artifact_dir / "review.wav").write_bytes(
        b"stale"
    )
    (
        embedding_dir
        / "ecapa_speaker.npy"
    ).write_bytes(
        b"stale"
    )

    updated = accept_alignment_recovery(
        storage,
        "turn_000001",
    )

    assert updated["source_start"] == 0.8
    assert updated["source_end"] == 2.2

    assert (
        updated["metadata"][
            "alignment_evidence"
        ]["recovery"]["status"]
        == "accepted"
    )

    assert not artifact_dir.exists()
