from dataclasses import dataclass

from voice_dataset.cuda import preload_cuda_libraries

# ctranslate2/faster-whisper loads CUDA libraries dynamically.
# Preload the CUDA runtime dependencies installed in the virtualenv
# before importing/using WhisperModel.
preload_cuda_libraries()

from faster_whisper import WhisperModel


@dataclass
class Segment:
    start: float
    end: float
    text: str


class Transcriber:
    def __init__(self, model_name: str = "large-v3"):
        print(f"Loading Whisper model {model_name}...", flush=True)

        self.model = WhisperModel(
            model_name,
            device="cuda",
            compute_type="float16",
        )

    def transcribe(
        self,
        path: str,
        language: str | None = None,
    ) -> tuple[str, list[Segment]]:
        print(f"Transcribing {path}...", flush=True)

        segments, info = self.model.transcribe(
            path,
            language=language,
            beam_size=5,
            vad_filter=True,
        )

        result = [
            Segment(
                start=s.start,
                end=s.end,
                text=s.text.strip(),
            )
            for s in segments
        ]

        return info.language, result
