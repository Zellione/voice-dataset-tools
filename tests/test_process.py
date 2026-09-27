from pathlib import Path
from types import SimpleNamespace
from voice_dataset.media import AudioStream

import voice_dataset.process as process_module


def test_process_source_runs_pipeline_in_order(
    tmp_path: Path,
    monkeypatch,
):
    source = tmp_path / "episode.mkv"
    source.touch()

    output = tmp_path / "dataset"

    calls = []

    class FakeStorage:
        def __init__(self, root):
            self.root = root
            self.regions = SimpleNamespace(
                load=lambda: [],
            )

        def get_source(self, source_id):
            return None

    monkeypatch.setattr(
        process_module,
        "DatasetStorage",
        FakeStorage,
    )

    def fake_ingest(
        source,
        output,
        source_id,
        audio_stream=None,
        language=None,
        channel="auto",
        start=None,
        duration=None,
    ):
        calls.append(
            (
                "ingest",
                {
                    "source": source,
                    "output": output,
                    "source_id": source_id,
                    "audio_stream": audio_stream,
                    "language": language,
                    "channel": channel,
                    "start": start,
                    "duration": duration,
                },
            )
        )

    monkeypatch.setattr(
        process_module,
        "ingest",
        fake_ingest,
    )

    def fake_resolve(
        storage,
        source_id,
        purpose,
    ):
        calls.append(("resolve", purpose))

        if purpose == "boundary_analysis":
            return (
                "center",
                {"kind": "center"},
                output / "center.wav",
            )

        if purpose == "asr":
            return (
                "center",
                {"kind": "center"},
                output / "center.wav",
            )

        raise AssertionError(
            f"Unexpected purpose: {purpose}"
        )

    monkeypatch.setattr(
        process_module,
        "resolve_source_representation_for_purpose",
        fake_resolve,
    )

    def fake_separate(
        storage,
        source_id,
        *,
        input_representation,
        output_representation,
    ):
        calls.append(
            (
                "separate",
                input_representation,
                output_representation,
            )
        )

    monkeypatch.setattr(
        process_module,
        "separate_source",
        fake_separate,
    )

    def fake_detect(
        storage,
        source_id,
        representation_name,
    ):
        calls.append(
            (
                "detect",
                representation_name,
            )
        )

    monkeypatch.setattr(
        process_module,
        "detect_source_regions",
        fake_detect,
    )

    def fake_transcribe_regions(
        storage,
        source_id,
        *,
        representation_name,
        transcript_name,
        language,
    ):
        calls.append(
            (
                "whisper",
                representation_name,
                transcript_name,
                language,
            )
        )

    monkeypatch.setattr(
        process_module,
        "transcribe_regions",
        fake_transcribe_regions,
    )

    monkeypatch.setattr(
        process_module,
        "_has_continuous_asr",
        lambda *args: False,
    )

    def fake_qwen(
        storage,
        source_id,
        *,
        representation_name,
        evidence_name,
        language,
    ):
        calls.append(
            (
                "qwen",
                representation_name,
                evidence_name,
                language,
            )
        )

    monkeypatch.setattr(
        process_module,
        "transcribe_source_qwen3",
        fake_qwen,
    )

    pipeline = object()

    def fake_build(
        storage,
        source_id,
        *,
        asr_evidence_name,
    ):
        calls.append(
            (
                "build",
                asr_evidence_name,
            )
        )
        return pipeline

    monkeypatch.setattr(
        process_module,
        "build_source_utterances",
        fake_build,
    )

    turns = SimpleNamespace(
        created=3,
        skipped=4,
        review=1,
    )

    def fake_apply(
        storage,
        source_id,
        result,
        *,
        language,
        asr_evidence_name,
    ):
        assert result is pipeline

        calls.append(
            (
                "apply",
                language,
                asr_evidence_name,
            )
        )

        return turns

    monkeypatch.setattr(
        process_module,
        "apply_source_utterance_turns",
        fake_apply,
    )

    def fake_prepare_review(
        storage,
        source_id,
    ):
        calls.append(("review",))

    monkeypatch.setattr(
        process_module,
        "prepare_source_review_audio",
        fake_prepare_review,
    )

    def fake_prepare(
        storage,
        source_id,
    ):
        calls.append(("speakers",))

    monkeypatch.setattr(
        process_module,
        "prepare_source_speaker_evidence",
        fake_prepare,
    )

    result = process_module.process_source(
        source,
        output,
        source_id="episode",
        audio_stream=2,
        audio_language="eng",
        language="English",
        channel="center",
        start=1200.0,
        duration=300.0,
    )

    assert calls == [
        (
            "ingest",
            {
                "source": source.resolve(),
                "output": output.resolve(),
                "source_id": "episode",
                "audio_stream": 2,
                "language": "eng",
                "channel": "center",
                "start": 1200.0,
                "duration": 300.0,
            },
        ),
        ("resolve", "boundary_analysis"),
        ("separate", "center", "speech"),
        ("resolve", "asr"),
        ("detect", "center"),
        (
            "whisper",
            "center",
            "whisper",
            "English",
        ),
        (
            "qwen",
            "center",
            "qwen3",
            "English",
        ),
        ("build", "qwen3"),
        ("apply", "English", "qwen3"),
        ("review",),
        ("speakers",),
    ]

    assert result == process_module.ProcessResult(
        source_id="episode",
        detector_ran=True,
        qwen_ran=True,
        turns_created=3,
        turns_skipped=4,
        turns_review=1,
    )


def test_process_source_skips_expensive_completed_steps(
    tmp_path: Path,
    monkeypatch,
):
    source = tmp_path / "episode.mkv"
    source.touch()

    output = tmp_path / "dataset"

    existing_source = {
        "id": "episode",
        "media_path": str(source.resolve()),
        "metadata": {
            "continuous_asr": {
                "qwen3": {
                    "model": "existing",
                },
            },
            "utterance_reconciliation": {
                "asr_evidence": "qwen3",
                "word_ranges": [
                    [0, 8],
                    [9, 12],
                    [13, 17],
                ],
            },
        },
    }

    class FakeStorage:
        def __init__(self, root):
            self.root = root
            self.regions = SimpleNamespace(
                load=lambda: [
                    {
                        "id": "region_000001",
                        "source_id": "episode",
                    },
                ],
            )
            self.turns = SimpleNamespace(
                load=lambda: [
                    {
                        "id": "turn_000001",
                        "source_id": "episode",
                        "metadata": {
                            "creation": {
                                "method": (
                                    "continuous_asr_utterance"
                                ),
                            },
                            "word_range": {
                                "start": 0,
                                "end": 8,
                            },
                        },
                    },
                    {
                        "id": "turn_000002",
                        "source_id": "episode",
                        "metadata": {
                            "creation": {
                                "method": (
                                    "continuous_asr_utterance"
                                ),
                            },
                            "word_range": {
                                "start": 9,
                                "end": 12,
                            },
                        },
                    },
                    {
                        "id": "turn_000003",
                        "source_id": "episode",
                        "metadata": {
                            "creation": {
                                "method": (
                                    "continuous_asr_utterance"
                                ),
                            },
                            "word_range": {
                                "start": 13,
                                "end": 17,
                            },
                        },
                    },
                ],
            )

        def get_source(self, source_id):
            return existing_source

    monkeypatch.setattr(
        process_module,
        "DatasetStorage",
        FakeStorage,
    )

    def must_not_run(*args, **kwargs):
        raise AssertionError(
            "completed expensive step ran again"
        )

    monkeypatch.setattr(
        process_module,
        "ingest",
        must_not_run,
    )

    monkeypatch.setattr(
        process_module,
        "detect_source_regions",
        must_not_run,
    )

    monkeypatch.setattr(
        process_module,
        "transcribe_source_qwen3",
        must_not_run,
    )

    def fake_resolve(
        storage,
        source_id,
        purpose,
    ):
        if purpose == "boundary_analysis":
            return (
                "center",
                {"kind": "center"},
                output / "center.wav",
            )

        if purpose == "asr":
            return (
                "center",
                {"kind": "center"},
                output / "center.wav",
            )

        raise AssertionError(
            f"Unexpected purpose: {purpose}"
        )

    monkeypatch.setattr(
        process_module,
        "resolve_source_representation_for_purpose",
        fake_resolve,
    )

    monkeypatch.setattr(
        process_module,
        "separate_source",
        lambda *args, **kwargs: None,
    )

    monkeypatch.setattr(
        process_module,
        "transcribe_regions",
        lambda *args, **kwargs: None,
    )

    monkeypatch.setattr(
        process_module,
        "build_source_utterances",
        must_not_run,
    )

    monkeypatch.setattr(
        process_module,
        "apply_source_utterance_turns",
        must_not_run,
    )

    monkeypatch.setattr(
        process_module,
        "prepare_source_speaker_evidence",
        lambda *args, **kwargs: None,
    )

    review_calls = []

    def fake_prepare_review(
        storage,
        source_id,
    ):
        review_calls.append(source_id)

    monkeypatch.setattr(
        process_module,
        "prepare_source_review_audio",
        fake_prepare_review,
    )

    result = process_module.process_source(
        source,
        output,
        source_id="episode",
        language="English",
    )

    assert review_calls == ["episode"]

    assert result.detector_ran is False
    assert result.qwen_ran is False
    assert result.turns_created == 0
    assert result.turns_skipped == 0


def test_process_source_rejects_existing_source_from_different_media(
    tmp_path: Path,
    monkeypatch,
):
    source = tmp_path / "episode.mkv"
    source.touch()

    other_source = tmp_path / "other-episode.mkv"
    other_source.touch()

    output = tmp_path / "dataset"

    existing_source = {
        "id": "episode",
        "media_path": str(other_source.resolve()),
        "representations": {
            "center": {
                "path": "audio/episode/center.wav",
                "kind": "center",
                "media_start": 0.0,
                "duration": 300.0,
                "stream_index": 1,
                "channel_mode": "center",
                "purposes": [
                    "asr",
                    "speaker_embedding",
                    "boundary_analysis",
                    "context",
                ],
                "metadata": {
                    "audio_stream": 0,
                    "source_language": "eng",
                },
            },
        },
        "metadata": {},
    }

    class FakeStorage:
        def __init__(self, root):
            self.root = root

        def get_source(self, source_id):
            return existing_source

    monkeypatch.setattr(
        process_module,
        "DatasetStorage",
        FakeStorage,
    )

    try:
        process_module.process_source(
            source,
            output,
            source_id="episode",
        )
    except ValueError as error:
        assert "different media" in str(error)
    else:
        raise AssertionError(
            "existing source from different media "
            "was accepted"
        )


def test_process_source_rejects_different_requested_start(
    tmp_path: Path,
    monkeypatch,
):
    source = tmp_path / "episode.mkv"
    source.touch()

    output = tmp_path / "dataset"

    existing_source = {
        "id": "episode",
        "media_path": str(source.resolve()),
        "representations": {},
        "metadata": {
            "ingest": {
                "requested_audio_stream": None,
                "requested_audio_language": None,
                "requested_channel": "auto",
                "requested_start": 1200.0,
                "requested_duration": 300.0,
                "audio_stream": 0,
                "stream_index": 1,
                "channel_mode": "center",
                "source_language": "eng",
            },
        },
    }

    class FakeStorage:
        def __init__(self, root):
            self.root = root

        def get_source(self, source_id):
            return existing_source

    monkeypatch.setattr(
        process_module,
        "DatasetStorage",
        FakeStorage,
    )

    try:
        process_module.process_source(
            source,
            output,
            source_id="episode",
            start=1300.0,
            duration=300.0,
        )
    except ValueError as error:
        assert "requested_start" in str(error)
    else:
        raise AssertionError(
            "different requested start was accepted"
        )


def test_validate_existing_source_accepts_equivalent_audio_selection(
    tmp_path: Path,
    monkeypatch,
):
    source = tmp_path / "episode.mkv"
    source.touch()

    stream = AudioStream(
        index=4,
        codec="aac",
        sample_rate=48000,
        channels=6,
        channel_layout="5.1",
        language="eng",
        title="English",
        is_default=False,
    )

    record = {
        "id": "episode",
        "media_path": str(source.resolve()),
        "metadata": {
            "ingest": {
                "requested_audio_stream": None,
                "requested_audio_language": "eng",
                "requested_channel": "auto",
                "requested_start": 1200.0,
                "requested_duration": 300.0,
                "audio_stream": 1,
                "stream_index": 4,
                "channel_mode": "center",
                "source_language": "eng",
            },
        },
    }

    monkeypatch.setattr(
        process_module,
        "probe_audio_streams",
        lambda path: [
            AudioStream(
                index=1,
                codec="aac",
                sample_rate=48000,
                channels=2,
                channel_layout="stereo",
                language="deu",
                title=None,
                is_default=True,
            ),
            stream,
        ],
    )

    process_module._validate_existing_source(
        record,
        source=source,
        audio_stream=1,
        audio_language=None,
        channel="auto",
        start=1200.0,
        duration=300.0,
    )


def test_validate_existing_source_rejects_different_effective_audio_stream(
    tmp_path: Path,
    monkeypatch,
):
    source = tmp_path / "episode.mkv"
    source.touch()

    record = {
        "id": "episode",
        "media_path": str(source.resolve()),
        "metadata": {
            "ingest": {
                "requested_audio_stream": None,
                "requested_audio_language": "eng",
                "requested_channel": "auto",
                "requested_start": 1200.0,
                "requested_duration": 300.0,
                "audio_stream": 1,
                "stream_index": 4,
                "channel_mode": "center",
                "source_language": "eng",
            },
        },
    }

    monkeypatch.setattr(
        process_module,
        "probe_audio_streams",
        lambda path: [
            AudioStream(
                index=1,
                codec="aac",
                sample_rate=48000,
                channels=2,
                channel_layout="stereo",
                language="deu",
                title=None,
                is_default=True,
            ),
            AudioStream(
                index=4,
                codec="aac",
                sample_rate=48000,
                channels=6,
                channel_layout="5.1",
                language="eng",
                title="English",
                is_default=False,
            ),
        ],
    )

    try:
        process_module._validate_existing_source(
            record,
            source=source,
            audio_stream=0,
            audio_language=None,
            channel="auto",
            start=1200.0,
            duration=300.0,
        )
    except ValueError as error:
        assert "audio stream" in str(error)
    else:
        raise AssertionError(
            "different effective audio stream was accepted"
        )
