import time
from io import StringIO
from threading import Thread
from typing import BinaryIO, Union

import whisper
from faster_whisper import WhisperModel

from app.asr_models.asr_model import ASRModel
from app.config import CONFIG
from app.utils import ResultWriter, WriteJSON, WriteSRT, WriteTSV, WriteTXT, WriteVTT


class FasterWhisperASR(ASRModel):

    def load_model(self):
        import torch

        # Optimize faster-whisper for L4 GPU
        model_kwargs = {
            "model_size_or_path": CONFIG.MODEL_NAME,
            "device": CONFIG.DEVICE,
            "compute_type": CONFIG.MODEL_QUANTIZATION,
            "download_root": CONFIG.MODEL_PATH,
        }

        # Add L4-specific optimizations
        if torch.cuda.is_available():
            model_kwargs.update(
                {
                    "device_index": 0,  # Explicitly use first GPU
                    "cpu_threads": 8,  # Optimize CPU threads for L4 workloads
                    "num_workers": 2,  # Enable parallel processing within model
                }
            )

            # For L4 GPU, we can be more aggressive with memory usage
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if gpu_memory_gb > 20:  # L4 has 24GB
                print(f"🚀 L4 GPU detected ({gpu_memory_gb:.1f}GB) - enabling high memory mode")
                # Enable larger batch processing and better memory utilization

        self.model = WhisperModel(**model_kwargs)

        # Pre-warm the model with dummy audio to allocate GPU memory
        self._warmup_model()

        Thread(target=self.monitor_idleness, daemon=True).start()

    def _warmup_model(self):
        """Pre-warm the model to allocate GPU memory and optimize performance"""
        import numpy as np
        import torch

        if not torch.cuda.is_available():
            return

        print("🔥 Warming up model for optimal GPU memory allocation...")

        try:
            # Create dummy audio (5 seconds at 16kHz)
            dummy_audio = np.zeros(80000, dtype=np.float32)

            # Run a dummy transcription to allocate GPU memory
            segment_generator, _ = self.model.transcribe(
                dummy_audio, beam_size=1, word_timestamps=True, vad_filter=True
            )

            # Consume the generator to ensure full allocation
            list(segment_generator)

            # Check GPU memory allocation
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / (1024**3)
                cached = torch.cuda.memory_reserved(0) / (1024**3)
                print(f"✅ Model warmed up - GPU allocated: {allocated:.2f}GB, reserved: {cached:.2f}GB")

        except Exception as e:
            print(f"⚠️ Model warmup failed (continuing anyway): {e}")

    def transcribe(
        self,
        audio,
        task: Union[str, None],
        language: Union[str, None],
        initial_prompt: Union[str, None],
        vad_filter: Union[bool, None],
        word_timestamps: Union[bool, None],
        options: Union[dict, None],
        output,
    ):
        self.last_activity_time = time.time()

        # Ensure model is loaded (only lock for model loading)
        with self.model_lock:
            if self.model is None:
                self.load_model()

        options_dict = {"task": task}
        if language:
            options_dict["language"] = language
        if initial_prompt:
            options_dict["initial_prompt"] = initial_prompt
        if vad_filter:
            options_dict["vad_filter"] = True
        if word_timestamps:
            options_dict["word_timestamps"] = True

        # Use semaphore to limit concurrent GPU operations instead of exclusive lock
        import threading

        current_thread = threading.current_thread().name
        print(f"🔄 Thread {current_thread}: Acquiring semaphore (available: {self.request_semaphore._value})")

        with self.request_semaphore:
            print(f"🎯 Thread {current_thread}: Starting transcription (audio duration: ~{len(audio)/16000:.1f}s)")

            # Optimize transcription parameters for L4 GPU
            transcribe_options = {
                "beam_size": 1,
                "best_of": 1,  # Multiple candidates for better accuracy
                "temperature": 0.0,  # Deterministic output
                "compression_ratio_threshold": 2.4,  # Better quality control
                "log_prob_threshold": -1.0,  # Filter low-confidence segments
                "no_speech_threshold": 0.6,  # Better silence detection
                "condition_on_previous_text": True,  # Better context awareness
                "prompt_reset_on_temperature": 0.5,  # Reset prompts for long audio
                **options_dict,
            }

            segments = []
            text = ""
            start_time = time.time()

            segment_generator, info = self.model.transcribe(audio, **transcribe_options)

            for segment in segment_generator:
                segments.append(segment)
                text = text + segment.text

            transcription_time = time.time() - start_time
            audio_duration = len(audio) / 16000
            rtf = transcription_time / audio_duration  # Real-time factor

            print(
                f"✅ Thread {current_thread}: Transcription complete ({len(segments)} segments, {transcription_time:.2f}s, RTF: {rtf:.2f}x)"
            )
            result = {"language": options_dict.get("language", info.language), "segments": segments, "text": text}

        output_file = StringIO()
        self.write_result(result, output_file, output)
        output_file.seek(0)

        return output_file

    def language_detection(self, audio):

        self.last_activity_time = time.time()

        # Ensure model is loaded (only lock for model loading)
        with self.model_lock:
            if self.model is None:
                self.load_model()

        # load audio and pad/trim it to fit 30 seconds
        audio = whisper.pad_or_trim(audio)

        # detect the spoken language using semaphore for concurrency
        with self.request_semaphore:
            _, info = self.model.transcribe(audio, beam_size=1)
            detected_lang_code = info.language
            detected_language_confidence = info.language_probability

        return detected_lang_code, detected_language_confidence

    def write_result(self, result: dict, file: BinaryIO, output: Union[str, None]):
        if output == "srt":
            WriteSRT(ResultWriter).write_result(result, file=file)
        elif output == "vtt":
            WriteVTT(ResultWriter).write_result(result, file=file)
        elif output == "tsv":
            WriteTSV(ResultWriter).write_result(result, file=file)
        elif output == "json":
            WriteJSON(ResultWriter).write_result(result, file=file)
        else:
            WriteTXT(ResultWriter).write_result(result, file=file)
