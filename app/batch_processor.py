"""
Batch processing for optimal GPU utilization.
Groups multiple requests together to maximize GPU throughput.
"""
import asyncio
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

@dataclass
class BatchRequest:
    """A single request in a batch"""
    request_id: str
    audio_data: Any
    params: Dict[str, Any]
    future: asyncio.Future
    created_at: float

class BatchProcessor:
    """
    Intelligent batch processor for GPU operations.
    Groups requests together for optimal GPU utilization.
    """
    
    def __init__(self, 
                 max_batch_size: int = 4,
                 max_wait_time: float = 0.1,  # 100ms max wait
                 process_func: Optional[Callable] = None):
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.process_func = process_func
        
        self.pending_requests: deque = deque()
        self.processing_lock = asyncio.Lock()
        self.batch_counter = 0
        
        # Start the batch processing loop
        asyncio.create_task(self._batch_processing_loop())
    
    async def add_request(self, audio_data: Any, params: Dict[str, Any]) -> Any:
        """Add a request to the batch queue and wait for result"""
        request_id = f"req_{self.batch_counter}_{time.time()}"
        self.batch_counter += 1
        
        future = asyncio.Future()
        request = BatchRequest(
            request_id=request_id,
            audio_data=audio_data,
            params=params,
            future=future,
            created_at=time.time()
        )
        
        self.pending_requests.append(request)
        
        # Wake up the batch processor if it's waiting
        if hasattr(self, '_batch_event'):
            self._batch_event.set()
        
        # Wait for the result
        return await future
    
    async def _batch_processing_loop(self):
        """Main batch processing loop"""
        while True:
            try:
                # Wait for requests or timeout
                await asyncio.sleep(0.01)  # 10ms check interval
                
                if not self.pending_requests:
                    continue
                
                async with self.processing_lock:
                    if not self.pending_requests:
                        continue
                    
                    # Collect requests for batching
                    batch_requests = []
                    current_time = time.time()
                    
                    # Strategy 1: Collect until max batch size
                    while (len(batch_requests) < self.max_batch_size and 
                           self.pending_requests):
                        request = self.pending_requests[0]
                        
                        # Strategy 2: Process if oldest request is too old
                        if (current_time - request.created_at) > self.max_wait_time:
                            batch_requests.append(self.pending_requests.popleft())
                            break
                        
                        batch_requests.append(self.pending_requests.popleft())
                    
                    # Process the batch if we have requests
                    if batch_requests:
                        await self._process_batch(batch_requests)
                        
            except Exception as e:
                print(f"❌ Batch processing error: {e}")
                # Fail any pending requests
                for request in batch_requests:
                    if not request.future.done():
                        request.future.set_exception(e)
    
    async def _process_batch(self, requests: List[BatchRequest]):
        """Process a batch of requests"""
        batch_size = len(requests)
        start_time = time.time()
        
        print(f"🔄 Processing batch of {batch_size} requests")
        
        try:
            # For now, process each request individually
            # TODO: Implement true batch processing for models that support it
            for request in requests:
                try:
                    if self.process_func:
                        result = await self._call_process_func(request)
                        request.future.set_result(result)
                    else:
                        request.future.set_exception(
                            RuntimeError("No process function configured")
                        )
                except Exception as e:
                    request.future.set_exception(e)
            
            end_time = time.time()
            processing_time = end_time - start_time
            print(f"✅ Batch completed in {processing_time:.2f}s ({batch_size} requests)")
            
        except Exception as e:
            print(f"❌ Batch processing failed: {e}")
            for request in requests:
                if not request.future.done():
                    request.future.set_exception(e)
    
    async def _call_process_func(self, request: BatchRequest):
        """Call the processing function for a single request"""
        if asyncio.iscoroutinefunction(self.process_func):
            return await self.process_func(
                request.audio_data, 
                **request.params
            )
        else:
            return self.process_func(
                request.audio_data,
                **request.params
            )