# Problem 1 Solution: Service Boundaries - 3 Implementation Alternatives

## Current Problem
`TranscriptionProcessor` directly imports and calls:
- `backend.transcribe.transcribe()` function
- `backend.variants.executor` functions (create_variant_preprocess_runner, etc.)
- `TranscriptionDatabase` class directly
- `shutil.move()` for file operations
- Direct file I/O for output writing

**Impact:** Hard to swap implementations, test cleanly, or benchmark different engines.

---

## Alternative 1: Protocol-Based Interfaces (Cleanest)

**Approach:** Use Python `typing.Protocol` for structural typing. No inheritance required, duck-typing with type safety.

### Implementation

```python
# backend/services/interfaces.py
from typing import Protocol, runtime_checkable
from pathlib import Path
from dataclasses import dataclass

@dataclass
class TranscriptionRequest:
    audio_path: str
    output_path: str
    preset: str
    language: str | None = None

@dataclass
class TranscriptionResult:
    metrics: TranscriptionMetrics | None
    payload: dict[str, Any]

@runtime_checkable
class TranscriptionService(Protocol):
    """Service for transcribing audio files."""
    
    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        """Transcribe audio and return result with metrics."""
        ...

@runtime_checkable
class PreprocessService(Protocol):
    """Service for preprocessing audio files."""
    
    def preprocess(
        self, 
        audio_path: str, 
        config: PreprocessConfig
    ) -> PreprocessResult:
        """Preprocess audio file."""
        ...

@runtime_checkable
class StateStore(Protocol):
    """Service for managing transcription state."""
    
    def update_status(
        self, 
        file_path: str, 
        status: str, 
        error_message: str | None = None
    ) -> None:
        """Update file processing status."""
        ...
    
    def record_run(self, run_record: RunRecord) -> int:
        """Record a processing run and return run_id."""
        ...
    
    def record_file_metric(self, file_record: FileMetricRecord) -> None:
        """Record file-level metrics."""
        ...

@runtime_checkable
class FileMover(Protocol):
    """Service for moving files."""
    
    def move(self, source: str, destination: str) -> str | None:
        """Move file from source to destination."""
        ...

@runtime_checkable
class OutputWriter(Protocol):
    """Service for writing transcription output."""
    
    def write(
        self, 
        output_path: str, 
        payload: dict[str, Any], 
        format: str
    ) -> list[Path]:
        """Write transcription output in specified format. Returns list of created files."""
        ...
```

### Concrete Implementations

```python
# backend/services/whisper_transcription_service.py
from backend.services.interfaces import TranscriptionService, TranscriptionRequest, TranscriptionResult
from backend.transcribe import transcribe
from backend.variants.executor import transcribe_with_minimal_params, is_minimal_config

class WhisperTranscriptionService:
    """Concrete implementation using faster-whisper."""
    
    def __init__(
        self,
        preprocess_config_provider: Callable[[], PreprocessConfig],
        preprocess_runner: Callable[[str, PreprocessConfig], PreprocessResult],
        transcription_config_provider: Callable[[], TranscriptionConfig],
    ):
        self._preprocess_config_provider = preprocess_config_provider
        self._preprocess_runner = preprocess_runner
        self._transcription_config_provider = transcription_config_provider
    
    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        metrics_container: dict[str, TranscriptionMetrics] = {}
        
        def _collect(metrics: TranscriptionMetrics) -> None:
            metrics_container["value"] = metrics
        
        transcription_config = self._transcription_config_provider()
        is_minimal = is_minimal_config(transcription_config)
        
        if is_minimal:
            payload = transcribe_with_minimal_params(
                path=request.audio_path,
                preset=request.preset,
                language=request.language,
                preprocess_config=self._preprocess_config_provider(),
                preprocess_runner=self._preprocess_runner,
                transcription_config=transcription_config,
                metrics_collector=_collect,
            )
        else:
            payload = transcribe(
                path=request.audio_path,
                preset=request.preset,
                language=request.language,
                preprocess_config_provider=self._preprocess_config_provider,
                preprocess_runner=self._preprocess_runner,
                transcription_config_provider=self._transcription_config_provider,
                metrics_collector=_collect,
            )
        
        return TranscriptionResult(
            metrics=metrics_container.get("value"),
            payload=payload
        )

# backend/services/duckdb_state_store.py
from backend.services.interfaces import StateStore
from backend.database import TranscriptionDatabase, RunRecord, FileMetricRecord

class DuckDBStateStore:
    """Concrete implementation using DuckDB."""
    
    def __init__(self, db: TranscriptionDatabase):
        self._db = db
    
    def update_status(self, file_path: str, status: str, error_message: str | None = None) -> None:
        self._db.update_status(file_path, status, error_message)
    
    def record_run(self, run_record: RunRecord) -> int:
        return self._db.record_run(run_record)
    
    def record_file_metric(self, file_record: FileMetricRecord) -> None:
        self._db.record_file_metric(file_record)

# backend/services/shutil_file_mover.py
from backend.services.interfaces import FileMover
import shutil

class ShutilFileMover:
    """Concrete implementation using shutil."""
    
    def move(self, source: str, destination: str) -> str | None:
        return shutil.move(source, destination)

# backend/services/json_output_writer.py
from backend.services.interfaces import OutputWriter
import json
from pathlib import Path

class JsonOutputWriter:
    """Concrete implementation for writing transcription output."""
    
    def write(self, output_path: str, payload: dict[str, Any], format: str) -> list[Path]:
        created_files = []
        base_path = Path(output_path)
        
        if format in ("txt", "both"):
            txt_path = base_path.with_suffix(".txt")
            segments = payload.get("segments", [])
            with open(txt_path, "w", encoding="utf-8") as f:
                for segment in segments:
                    f.write(segment["text"] + "\n")
            created_files.append(txt_path)
        
        if format in ("json", "both"):
            json_path = base_path.with_suffix(".json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            created_files.append(json_path)
        
        return created_files
```

### Usage in Processor

```python
# backend/processor.py
from backend.services.interfaces import (
    TranscriptionService,
    StateStore,
    FileMover,
    OutputWriter,
    TranscriptionRequest,
)

class TranscriptionProcessor:
    def __init__(
        self,
        transcription_service: TranscriptionService,
        state_store: StateStore,
        file_mover: FileMover,
        output_writer: OutputWriter,
        input_folder: str | Path,
        output_format: str = DEFAULT_OUTPUT_FORMAT,
        # ... other params
    ):
        self._transcription_service = transcription_service
        self._state_store = state_store
        self._file_mover = file_mover
        self._output_writer = output_writer
        # ...
    
    def process_file(self, file_path: str) -> FileProcessingStats:
        # ...
        request = TranscriptionRequest(
            audio_path=str(file_path_obj),
            output_path=str(output_path),
            preset=self.preset,
            language=self.language,
        )
        
        result = self._transcription_service.transcribe(request)
        created_files = self._output_writer.write(
            str(output_path),
            result.payload,
            self.output_format
        )
        
        self._state_store.update_status(file_path, "completed")
        # ...
```

### Pros
- ✅ **Cleanest**: No inheritance, pure structural typing
- ✅ **Pythonic**: Uses modern Python typing features
- ✅ **Flexible**: Any object matching the protocol works
- ✅ **Type-safe**: Full type checking with mypy/pyright
- ✅ **Test-friendly**: Easy to create mock implementations

### Cons
- ⚠️ Runtime checks optional (use `@runtime_checkable` if needed)
- ⚠️ Less familiar to developers from OOP backgrounds

---

## Alternative 2: Abstract Base Classes with Dependency Injection (Traditional)

**Approach:** Use `abc.ABC` for explicit interface contracts with runtime validation. More traditional OOP pattern.

### Implementation

```python
# backend/services/interfaces.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

@dataclass
class TranscriptionRequest:
    audio_path: str
    output_path: str
    preset: str
    language: str | None = None

@dataclass
class TranscriptionResult:
    metrics: TranscriptionMetrics | None
    payload: dict[str, Any]

class TranscriptionService(ABC):
    """Abstract service for transcribing audio files."""
    
    @abstractmethod
    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        """Transcribe audio and return result with metrics."""
        pass

class PreprocessService(ABC):
    """Abstract service for preprocessing audio files."""
    
    @abstractmethod
    def preprocess(
        self, 
        audio_path: str, 
        config: PreprocessConfig
    ) -> PreprocessResult:
        """Preprocess audio file."""
        pass

class StateStore(ABC):
    """Abstract service for managing transcription state."""
    
    @abstractmethod
    def update_status(
        self, 
        file_path: str, 
        status: str, 
        error_message: str | None = None
    ) -> None:
        """Update file processing status."""
        pass
    
    @abstractmethod
    def record_run(self, run_record: RunRecord) -> int:
        """Record a processing run and return run_id."""
        pass
    
    @abstractmethod
    def record_file_metric(self, file_record: FileMetricRecord) -> None:
        """Record file-level metrics."""
        pass

class FileMover(ABC):
    """Abstract service for moving files."""
    
    @abstractmethod
    def move(self, source: str, destination: str) -> str | None:
        """Move file from source to destination."""
        pass

class OutputWriter(ABC):
    """Abstract service for writing transcription output."""
    
    @abstractmethod
    def write(
        self, 
        output_path: str, 
        payload: dict[str, Any], 
        format: str
    ) -> list[Path]:
        """Write transcription output in specified format."""
        pass
```

### Concrete Implementations

```python
# backend/services/whisper_transcription_service.py
from backend.services.interfaces import TranscriptionService, TranscriptionRequest, TranscriptionResult

class WhisperTranscriptionService(TranscriptionService):
    """Concrete implementation using faster-whisper."""
    
    def __init__(
        self,
        preprocess_config_provider: Callable[[], PreprocessConfig],
        preprocess_runner: Callable[[str, PreprocessConfig], PreprocessResult],
        transcription_config_provider: Callable[[], TranscriptionConfig],
    ):
        self._preprocess_config_provider = preprocess_config_provider
        self._preprocess_runner = preprocess_runner
        self._transcription_config_provider = transcription_config_provider
    
    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        # Same implementation as Alternative 1
        ...

# backend/services/duckdb_state_store.py
from backend.services.interfaces import StateStore

class DuckDBStateStore(StateStore):
    """Concrete implementation using DuckDB."""
    
    def __init__(self, db: TranscriptionDatabase):
        self._db = db
    
    def update_status(self, file_path: str, status: str, error_message: str | None = None) -> None:
        self._db.update_status(file_path, status, error_message)
    
    def record_run(self, run_record: RunRecord) -> int:
        return self._db.record_run(run_record)
    
    def record_file_metric(self, file_record: FileMetricRecord) -> None:
        self._db.record_file_metric(file_record)
```

### Service Factory (Optional)

```python
# backend/services/factory.py
from backend.services.interfaces import (
    TranscriptionService,
    StateStore,
    FileMover,
    OutputWriter,
)
from backend.services.whisper_transcription_service import WhisperTranscriptionService
from backend.services.duckdb_state_store import DuckDBStateStore
from backend.services.shutil_file_mover import ShutilFileMover
from backend.services.json_output_writer import JsonOutputWriter
from backend.database import TranscriptionDatabase

class ServiceFactory:
    """Factory for creating service instances."""
    
    @staticmethod
    def create_transcription_service(
        preprocess_config_provider: Callable[[], PreprocessConfig],
        preprocess_runner: Callable[[str, PreprocessConfig], PreprocessResult],
        transcription_config_provider: Callable[[], TranscriptionConfig],
    ) -> TranscriptionService:
        return WhisperTranscriptionService(
            preprocess_config_provider,
            preprocess_runner,
            transcription_config_provider,
        )
    
    @staticmethod
    def create_state_store(db_path: str | Path | None = None) -> StateStore:
        db = TranscriptionDatabase(db_path)
        return DuckDBStateStore(db)
    
    @staticmethod
    def create_file_mover() -> FileMover:
        return ShutilFileMover()
    
    @staticmethod
    def create_output_writer() -> OutputWriter:
        return JsonOutputWriter()
```

### Usage in Processor

```python
# backend/processor.py
from backend.services.interfaces import (
    TranscriptionService,
    StateStore,
    FileMover,
    OutputWriter,
)

class TranscriptionProcessor:
    def __init__(
        self,
        transcription_service: TranscriptionService,
        state_store: StateStore,
        file_mover: FileMover,
        output_writer: OutputWriter,
        # ... same as Alternative 1
    ):
        # Same as Alternative 1
        ...
```

### Pros
- ✅ **Explicit contracts**: Clear inheritance hierarchy
- ✅ **Runtime validation**: ABC enforces method implementation
- ✅ **Familiar pattern**: Well-understood OOP approach
- ✅ **Type-safe**: Full type checking support
- ✅ **Test-friendly**: Easy to create test doubles

### Cons
- ⚠️ Requires inheritance (more ceremony)
- ⚠️ Less flexible than protocols (must inherit)

---

## Alternative 3: Service Layer with Explicit Dependencies (Most Structured)

**Approach:** Create a dedicated service layer with explicit dependency injection container. Most enterprise-ready, supports complex scenarios.

### Implementation

```python
# backend/services/container.py
from dataclasses import dataclass
from typing import Protocol, Callable
from pathlib import Path

@dataclass
class ServiceContainer:
    """Container holding all service dependencies."""
    
    transcription_service: "TranscriptionService"
    preprocess_service: "PreprocessService"
    state_store: "StateStore"
    file_mover: "FileMover"
    output_writer: "OutputWriter"
    
    @classmethod
    def create_default(
        cls,
        db_path: str | Path | None = None,
        preprocess_config_provider: Callable[[], PreprocessConfig] | None = None,
        preprocess_runner: Callable[[str, PreprocessConfig], PreprocessResult] | None = None,
        transcription_config_provider: Callable[[], TranscriptionConfig] | None = None,
    ) -> "ServiceContainer":
        """Create container with default implementations."""
        from backend.services.whisper_transcription_service import WhisperTranscriptionService
        from backend.services.duckdb_state_store import DuckDBStateStore
        from backend.services.shutil_file_mover import ShutilFileMover
        from backend.services.json_output_writer import JsonOutputWriter
        from backend.database import TranscriptionDatabase
        from backend.preprocess.audio import preprocess_audio
        from backend.preprocess.config import PreprocessConfig
        from backend.transcribe.config import TranscriptionConfig
        
        db = TranscriptionDatabase(db_path)
        
        return cls(
            transcription_service=WhisperTranscriptionService(
                preprocess_config_provider or PreprocessConfig.from_env,
                preprocess_runner or preprocess_audio,
                transcription_config_provider or TranscriptionConfig.from_env,
            ),
            preprocess_service=PreprocessServiceAdapter(preprocess_runner or preprocess_audio),
            state_store=DuckDBStateStore(db),
            file_mover=ShutilFileMover(),
            output_writer=JsonOutputWriter(),
        )

# backend/services/interfaces.py (same as Alternative 2, using ABC)
# ... (same abstract classes)

# backend/services/whisper_transcription_service.py (same as Alternative 2)
# ... (same implementations)
```

### Service Adapters

```python
# backend/services/preprocess_service_adapter.py
from backend.services.interfaces import PreprocessService
from backend.preprocess.audio import preprocess_audio

class PreprocessServiceAdapter(PreprocessService):
    """Adapter wrapping existing preprocess_audio function."""
    
    def __init__(
        self,
        preprocess_runner: Callable[[str, PreprocessConfig], PreprocessResult] = preprocess_audio,
    ):
        self._preprocess_runner = preprocess_runner
    
    def preprocess(self, audio_path: str, config: PreprocessConfig) -> PreprocessResult:
        return self._preprocess_runner(audio_path, config)
```

### Usage in Processor

```python
# backend/processor.py
from backend.services.container import ServiceContainer

class TranscriptionProcessor:
    def __init__(
        self,
        services: ServiceContainer,
        input_folder: str | Path,
        preset: str = "et-large",
        language: str | None = None,
        output_format: str = DEFAULT_OUTPUT_FORMAT,
        disable_file_moving: bool = False,
    ):
        self._services = services
        self.input_folder = Path(input_folder)
        self.preset = preset
        self.language = language
        self.output_format = output_format
        self._disable_file_moving = disable_file_moving
        # ...
    
    def process_file(self, file_path: str) -> FileProcessingStats:
        # ...
        request = TranscriptionRequest(
            audio_path=str(file_path_obj),
            output_path=str(output_path),
            preset=self.preset,
            language=self.language,
        )
        
        result = self._services.transcription_service.transcribe(request)
        created_files = self._services.output_writer.write(
            str(output_path),
            result.payload,
            self.output_format
        )
        
        self._services.state_store.update_status(file_path, "completed")
        # ...
```

### CLI Integration

```python
# backend/cli/transcription_commands.py
from backend.services.container import ServiceContainer

def cmd_process(args: argparse.Namespace) -> int:
    """Process audio files in input folder."""
    services = ServiceContainer.create_default(
        db_path=args.db_path,
    )
    
    processor = TranscriptionProcessor(
        services=services,
        input_folder=args.input_folder,
        preset=args.preset,
        language=args.language,
        output_format=args.output_format,
    )
    
    results = processor.process_folder()
    # ...
```

### Pros
- ✅ **Most structured**: Clear separation of concerns
- ✅ **Centralized configuration**: All services in one place
- ✅ **Easy to extend**: Add new services without changing processor
- ✅ **Test-friendly**: Swap entire container for testing
- ✅ **Enterprise-ready**: Supports complex dependency graphs
- ✅ **Versioning support**: Can version service interfaces

### Cons
- ⚠️ More boilerplate (container setup)
- ⚠️ Slightly more complex for simple use cases
- ⚠️ Requires understanding of DI patterns

---

## Comparison Matrix

| Feature | Alternative 1 (Protocols) | Alternative 2 (ABC) | Alternative 3 (Service Layer) |
|---------|---------------------------|---------------------|-------------------------------|
| **Cleanliness** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Pythonic** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Type Safety** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Testability** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Flexibility** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Familiarity** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Structure** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Boilerplate** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Enterprise Ready** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## Alternative 4: FastAPI Dependency Injection Framework

**Approach:** Use FastAPI's built-in dependency injection system to manage service boundaries. This leverages FastAPI's mature DI container instead of building custom protocols or ABCs.

### Implementation

```python
# backend/services/fastapi_app.py
from fastapi import FastAPI, Depends
from typing import Callable, Any
from pathlib import Path

# Create a FastAPI app instance for dependency management
app = FastAPI(title="STT-Faster Services")

# Service dependency providers
def get_transcription_service(
    preprocess_config_provider: Callable[[], PreprocessConfig] = Depends(PreprocessConfig.from_env),
    preprocess_runner: Callable[[str, PreprocessConfig], PreprocessResult] = Depends(lambda: preprocess_audio),
    transcription_config_provider: Callable[[], TranscriptionConfig] = Depends(TranscriptionConfig.from_env),
) -> "TranscriptionService":
    """Dependency provider for transcription service."""
    return WhisperTranscriptionService(
        preprocess_config_provider=preprocess_config_provider,
        preprocess_runner=preprocess_runner,
        transcription_config_provider=transcription_config_provider,
    )

def get_state_store(db_path: str | Path | None = None) -> "StateStore":
    """Dependency provider for state store."""
    db = TranscriptionDatabase(db_path)
    return DuckDBStateStore(db)

def get_file_mover() -> "FileMover":
    """Dependency provider for file mover."""
    return ShutilFileMover()

def get_output_writer() -> "OutputWriter":
    """Dependency provider for output writer."""
    return JsonOutputWriter()

# Service implementations (same as Alternative 1, but registered with FastAPI)
class WhisperTranscriptionService:
    """Concrete implementation using faster-whisper."""
    # ... same implementation as Alternative 1

class DuckDBStateStore:
    """Concrete implementation using DuckDB."""
    # ... same implementation as Alternative 1
```

### FastAPI Router for Services

```python
# backend/services/transcription_router.py
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
from pathlib import Path

router = APIRouter(prefix="/transcription", tags=["transcription"])

class TranscriptionRequest(BaseModel):
    audio_path: str
    output_path: str
    preset: str = "et-large"
    language: Optional[str] = None
    output_format: str = "txt"

class TranscriptionResponse(BaseModel):
    success: bool
    metrics: Optional[dict] = None
    created_files: list[str] = []
    error_message: Optional[str] = None

@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    request: TranscriptionRequest,
    transcription_service: TranscriptionService = Depends(get_transcription_service),
    output_writer: OutputWriter = Depends(get_output_writer),
    state_store: StateStore = Depends(get_state_store),
) -> TranscriptionResponse:
    """Transcribe a single audio file."""
    try:
        # Convert request to internal format
        transcribe_request = TranscriptionRequest(
            audio_path=request.audio_path,
            output_path=request.output_path,
            preset=request.preset,
            language=request.language,
        )

        # Perform transcription
        result = transcription_service.transcribe(transcribe_request)

        # Write output
        created_files = output_writer.write(
            request.output_path,
            result.payload,
            request.output_format
        )

        # Update state
        state_store.update_status(request.audio_path, "completed")

        return TranscriptionResponse(
            success=True,
            metrics=result.metrics.__dict__ if result.metrics else None,
            created_files=[str(f) for f in created_files],
        )

    except Exception as e:
        # Update state with error
        state_store.update_status(request.audio_path, "failed", str(e))
        raise HTTPException(status_code=500, detail=str(e))

# backend/services/processor_router.py
@router.post("/process-batch")
async def process_batch(
    input_folder: str,
    preset: str = "et-large",
    language: Optional[str] = None,
    output_format: str = "txt",
    transcription_service: TranscriptionService = Depends(get_transcription_service),
    state_store: StateStore = Depends(get_state_store),
    file_mover: FileMover = Depends(get_file_mover),
    output_writer: OutputWriter = Depends(get_output_writer),
) -> dict[str, Any]:
    """Process all files in a folder."""
    # Create processor with injected services
    processor = TranscriptionProcessor(
        transcription_service=transcription_service,
        state_store=state_store,
        file_mover=file_mover,
        output_writer=output_writer,
        input_folder=input_folder,
        preset=preset,
        language=language,
        output_format=output_format,
    )

    return processor.process_folder()
```

### Updated Processor

```python
# backend/processor.py
from fastapi import Depends

class TranscriptionProcessor:
    def __init__(
        self,
        transcription_service: TranscriptionService = Depends(get_transcription_service),
        state_store: StateStore = Depends(get_state_store),
        file_mover: FileMover = Depends(get_file_mover),
        output_writer: OutputWriter = Depends(get_output_writer),
        input_folder: str | Path,
        # ... other params
    ):
        self._transcription_service = transcription_service
        self._state_store = state_store
        self._file_mover = file_mover
        self._output_writer = output_writer
        # ... rest same as Alternative 1
```

### CLI Integration with FastAPI TestClient

```python
# backend/cli/transcription_commands.py
from fastapi.testclient import TestClient
from backend.services.fastapi_app import app

def cmd_process(args: argparse.Namespace) -> int:
    """Process audio files in input folder."""
    client = TestClient(app)

    response = client.post("/transcription/process-batch", json={
        "input_folder": args.input_folder,
        "preset": args.preset,
        "language": args.language,
        "output_format": args.output_format,
    })

    if response.status_code == 200:
        results = response.json()
        # Handle results...
        return 0
    else:
        print(f"Error: {response.text}")
        return 1
```

### Testing with FastAPI

```python
# tests/test_services.py
from fastapi.testclient import TestClient
from backend.services.fastapi_app import app
import pytest

@pytest.fixture
def client():
    return TestClient(app)

def test_transcription_service(client):
    # Test via HTTP API
    response = client.post("/transcription/transcribe", json={
        "audio_path": "/path/to/audio.wav",
        "output_path": "/path/to/output.txt",
        "preset": "et-large"
    })
    assert response.status_code == 200

def test_processor_integration(client):
    # Test batch processing via HTTP API
    response = client.post("/transcription/process-batch", json={
        "input_folder": "/tmp/test_input",
        "preset": "et-large"
    })
    assert response.status_code == 200
```

### Pros
- ✅ **Mature DI system**: FastAPI's dependency injection is battle-tested
- ✅ **HTTP interfaces**: Services can be called via HTTP (useful for distributed setups)
- ✅ **Built-in testing**: FastAPI TestClient for integration testing
- ✅ **OpenAPI docs**: Automatic API documentation
- ✅ **Request validation**: Pydantic models for request/response validation
- ✅ **Async support**: Built-in async/await support

### Cons
- ⚠️ **Overkill for internal services**: HTTP overhead for local calls
- ⚠️ **Web framework dependency**: Adds FastAPI as a core dependency
- ⚠️ **More complex**: HTTP serialization/deserialization adds complexity
- ⚠️ **Not traditional CLI**: Changes the application architecture

---

## Updated Comparison Matrix

| Feature | Alt 1 (Protocols) | Alt 2 (ABC) | Alt 3 (Service Layer) | Alt 4 (FastAPI) |
|---------|-------------------|-------------|----------------------|-----------------|
| **Cleanliness** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Pythonic** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Type Safety** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Testability** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Flexibility** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **HTTP APIs** | ❌ | ❌ | ❌ | ✅ |
| **Boilerplate** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Enterprise Ready** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **CLI Focus** | ✅ | ✅ | ✅ | ⚠️ |

## Recommendation

**For this codebase, still recommend Alternative 1 (Protocols)** because:
1. Most Pythonic and clean for a CLI application
2. Minimal boilerplate
3. Excellent type safety
4. Easy to test
5. No unnecessary web framework dependencies

**Consider Alternative 4 (FastAPI)** if:
- You want to expose transcription as web APIs
- You plan to build a web interface or distributed system
- You want automatic OpenAPI documentation
- You need request/response validation
- Your team is familiar with FastAPI patterns

**Alternative 4** would be most appropriate if you're planning to evolve this from a CLI tool into a web service or microservice architecture.

