import asyncio
import gc
import time
from abc import ABC, abstractmethod
from threading import Lock, Semaphore
from typing import Union

import torch

from app.config import CONFIG


class ASRModel(ABC):
    """
    Abstract base class for ASR (Automatic Speech Recognition) models.
    """

    model = None
    model_lock = Lock()  # Only for model loading/unloading
    last_activity_time = time.time()

    # Dynamic GPU concurrency control based on available memory
    _max_concurrent_requests = None
    _request_semaphore = None

    @classmethod
    def get_max_concurrent_requests(cls):
        """Calculate optimal concurrent requests based on GPU memory"""
        if cls._max_concurrent_requests is None:
            if torch.cuda.is_available():
                # Get GPU memory info
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

                # # Estimate based on model size and available memory
                # # Base model ~2GB, Small ~4GB, Medium ~8GB, Large ~12GB
                # model_memory_gb = {
                #     'tiny': 1, 'base': 2, 'small': 4,
                #     'medium': 8, 'large': 12, 'large-v2': 12, 'large-v3': 12
                # }.get(CONFIG.MODEL_NAME.lower(), 4)  # Default to 4GB

                # # Reserve 2GB for system, use 80% of remaining for models
                # available_memory = (gpu_memory_gb - 2) * 0.8
                # max_models = max(1, int(available_memory / model_memory_gb))

                # # L4 optimization: Use more aggressive concurrency for better memory utilization
                # # L4 has 24GB, base model ~2GB, but faster-whisper uses less memory
                # if gpu_memory_gb > 20:  # L4 GPU detected
                #     # Be more aggressive with L4's large memory
                #     cls._max_concurrent_requests = min(max_models, 8)
                #     print(f"🎯 L4 GPU: Enabling high concurrency mode ({cls._max_concurrent_requests} concurrent)")
                # else:
                #     cls._max_concurrent_requests = min(max_models, 4)
                cls._max_concurrent_requests = 3
                print(f"🚀 GPU Memory: {gpu_memory_gb:.1f}GB, Model: {CONFIG.MODEL_NAME}")
                print(f"🎯 Max concurrent requests: {cls._max_concurrent_requests}")
            else:
                cls._max_concurrent_requests = 1  # CPU fallback
        return cls._max_concurrent_requests

    @classmethod
    def get_request_semaphore(cls):
        """Get or create the request semaphore with dynamic limit"""
        if cls._request_semaphore is None:
            max_requests = cls.get_max_concurrent_requests()
            cls._request_semaphore = Semaphore(max_requests)
        return cls._request_semaphore

    @property
    def request_semaphore(self):
        return self.get_request_semaphore()

    def __init__(self):
        pass

    @abstractmethod
    def load_model(self):
        """
        Loads the model from the specified path.
        """
        pass

    @abstractmethod
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
        """
        Perform transcription on the given audio file.
        """
        pass

    @abstractmethod
    def language_detection(self, audio):
        """
        Perform language detection on the given audio file.
        """
        pass

    def monitor_idleness(self):
        """
        Monitors the idleness of the ASR model and releases the model if it has been idle for too long.
        """
        if CONFIG.MODEL_IDLE_TIMEOUT <= 0:
            return
        while True:
            time.sleep(15)
            if time.time() - self.last_activity_time > CONFIG.MODEL_IDLE_TIMEOUT:
                with self.model_lock:
                    self.release_model()
                    break

    def release_model(self):
        """
        Unloads the model from memory and clears any cached GPU memory.
        """
        del self.model
        torch.cuda.empty_cache()
        gc.collect()
        self.model = None
        print("Model unloaded due to timeout")
