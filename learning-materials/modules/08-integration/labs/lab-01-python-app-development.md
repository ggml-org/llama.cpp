# Lab 8.1: Python Application Development

**Estimated Time**: 2-3 hours
**Difficulty**: Intermediate
**Prerequisites**: Lesson 8.1 (Python Bindings)

## Objective

Build a complete Python application using llama-cpp-python with proper error handling, configuration management, and production-ready patterns.

## Learning Outcomes

- Set up a Python project with llama-cpp-python
- Implement robust error handling
- Create reusable components
- Add logging and monitoring
- Build a command-line interface
- Write unit tests

---

## Part 1: Project Setup (30 minutes)

### Task 1.1: Create Project Structure

Create a new Python project with the following structure:

```
llama-chat-app/
├── src/
│   ├── __init__.py
│   ├── llama_wrapper.py
│   ├── config.py
│   ├── utils.py
│   └── cli.py
├── tests/
│   ├── __init__.py
│   └── test_llama_wrapper.py
├── requirements.txt
├── setup.py
├── README.md
└── .env.example
```

**Deliverables:**
- Project directory structure
- Basic `requirements.txt` with dependencies
- `.env.example` file for configuration

### Task 1.2: Install Dependencies

Create `requirements.txt`:

```
llama-cpp-python>=0.2.0
python-dotenv>=1.0.0
click>=8.0.0
pydantic>=2.0.0
pytest>=7.0.0
```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Part 2: Core Components (45 minutes)

### Task 2.1: Configuration Management

Create `src/config.py`:

```python
from pydantic import BaseModel, Field
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

class ModelConfig(BaseModel):
    """Model configuration."""
    model_path: Path = Field(description="Path to GGUF model")
    n_ctx: int = Field(default=2048, description="Context window size")
    n_threads: int = Field(default=4, description="CPU threads")
    n_gpu_layers: int = Field(default=0, description="GPU layers")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, gt=0)

    class Config:
        validate_assignment = True

    @classmethod
    def from_env(cls) -> 'ModelConfig':
        """Load configuration from environment variables."""
        return cls(
            model_path=os.getenv('MODEL_PATH', './models/model.gguf'),
            n_ctx=int(os.getenv('N_CTX', '2048')),
            n_threads=int(os.getenv('N_THREADS', '4')),
            n_gpu_layers=int(os.getenv('N_GPU_LAYERS', '0')),
            temperature=float(os.getenv('TEMPERATURE', '0.7')),
            max_tokens=int(os.getenv('MAX_TOKENS', '512'))
        )
```

**Test your config:**
```python
config = ModelConfig.from_env()
print(config)
```

### Task 2.2: LLaMA Wrapper Class

Create `src/llama_wrapper.py`:

```python
from llama_cpp import Llama
from typing import Iterator, List, Dict
import logging
from .config import ModelConfig

logger = logging.getLogger(__name__)

class LlamaWrapper:
    """Thread-safe wrapper for llama.cpp with error handling."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.llm = None
        self._initialize()

    def _initialize(self):
        """Initialize the model with error handling."""
        try:
            logger.info(f"Loading model from {self.config.model_path}")

            self.llm = Llama(
                model_path=str(self.config.model_path),
                n_ctx=self.config.n_ctx,
                n_threads=self.config.n_threads,
                n_gpu_layers=self.config.n_gpu_layers,
                verbose=False
            )

            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def generate(
        self,
        prompt: str,
        stream: bool = False,
        **kwargs
    ) -> Iterator[str] | str:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt
            stream: Whether to stream output
            **kwargs: Additional generation parameters

        Returns:
            Generated text or iterator of tokens
        """
        if not self.llm:
            raise RuntimeError("Model not initialized")

        try:
            params = {
                'max_tokens': kwargs.get('max_tokens', self.config.max_tokens),
                'temperature': kwargs.get('temperature', self.config.temperature),
                'stream': stream
            }

            response = self.llm(prompt, **params)

            if stream:
                def token_generator():
                    for chunk in response:
                        token = chunk['choices'][0]['text']
                        yield token
                return token_generator()
            else:
                return response['choices'][0]['text']

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

    def chat(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
        **kwargs
    ) -> Iterator[str] | str:
        """
        Chat completion.

        Args:
            messages: Conversation messages
            stream: Whether to stream output
            **kwargs: Additional parameters

        Returns:
            Response text or token iterator
        """
        if not self.llm:
            raise RuntimeError("Model not initialized")

        try:
            params = {
                'messages': messages,
                'max_tokens': kwargs.get('max_tokens', self.config.max_tokens),
                'temperature': kwargs.get('temperature', self.config.temperature),
                'stream': stream
            }

            response = self.llm.create_chat_completion(**params)

            if stream:
                def token_generator():
                    full_response = ""
                    for chunk in response:
                        delta = chunk['choices'][0]['delta']
                        if 'content' in delta:
                            token = delta['content']
                            full_response += token
                            yield token
                return token_generator()
            else:
                return response['choices'][0]['message']['content']

        except Exception as e:
            logger.error(f"Chat failed: {e}")
            raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.llm:
            del self.llm
        return False
```

---

## Part 3: CLI Application (45 minutes)

### Task 3.1: Build CLI with Click

Create `src/cli.py`:

```python
import click
import logging
from pathlib import Path
from .config import ModelConfig
from .llama_wrapper import LlamaWrapper

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

@click.group()
def cli():
    """LLaMA Chat Application CLI."""
    pass

@cli.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.option('--n-ctx', default=2048, help='Context size')
@click.option('--n-gpu-layers', default=0, help='GPU layers')
def chat(model_path, n_ctx, n_gpu_layers):
    """Start interactive chat."""
    config = ModelConfig(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers
    )

    with LlamaWrapper(config) as llm:
        messages = []

        click.echo("Chat started. Type 'quit' to exit.\n")

        while True:
            user_input = click.prompt("You", type=str)

            if user_input.lower() in ['quit', 'exit']:
                break

            messages.append({
                "role": "user",
                "content": user_input
            })

            click.echo("\nAssistant: ", nl=False)

            full_response = ""
            for token in llm.chat(messages, stream=True):
                click.echo(token, nl=False)
                full_response += token

            click.echo("\n")

            messages.append({
                "role": "assistant",
                "content": full_response
            })

@cli.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.argument('prompt')
@click.option('--max-tokens', default=256)
def generate(model_path, prompt, max_tokens):
    """Generate single response."""
    config = ModelConfig(model_path=model_path, max_tokens=max_tokens)

    with LlamaWrapper(config) as llm:
        response = llm.generate(prompt)
        click.echo(response)

if __name__ == '__main__':
    cli()
```

---

## Part 4: Testing (30 minutes)

### Task 4.1: Write Unit Tests

Create `tests/test_llama_wrapper.py`:

```python
import pytest
from src.config import ModelConfig
from src.llama_wrapper import LlamaWrapper

@pytest.fixture
def config():
    """Test configuration."""
    return ModelConfig(
        model_path="./models/test-model.gguf",
        n_ctx=512,
        n_threads=2,
        n_gpu_layers=0
    )

def test_config_validation():
    """Test configuration validation."""
    config = ModelConfig(model_path="./model.gguf")
    assert config.n_ctx == 2048

    # Test invalid temperature
    with pytest.raises(ValueError):
        ModelConfig(model_path="./model.gguf", temperature=3.0)

def test_model_initialization(config, monkeypatch):
    """Test model initialization."""
    # Mock Llama class
    class MockLlama:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __call__(self, *args, **kwargs):
            return {'choices': [{'text': 'test response'}]}

    monkeypatch.setattr('src.llama_wrapper.Llama', MockLlama)

    wrapper = LlamaWrapper(config)
    assert wrapper.llm is not None

def test_generate(config, monkeypatch):
    """Test text generation."""
    class MockLlama:
        def __call__(self, *args, **kwargs):
            return {'choices': [{'text': 'Generated text'}]}

    monkeypatch.setattr('src.llama_wrapper.Llama', MockLlama)

    wrapper = LlamaWrapper(config)
    response = wrapper.generate("Test prompt")

    assert isinstance(response, str)
    assert len(response) > 0
```

Run tests:
```bash
pytest tests/
```

---

## Part 5: Documentation and Packaging (30 minutes)

### Task 5.1: Create README

Create `README.md`:

```markdown
# LLaMA Chat Application

Production-ready chat application using llama.cpp Python bindings.

## Installation

pip install -r requirements.txt

## Configuration

Copy `.env.example` to `.env` and configure:

MODEL_PATH=./models/llama-2-7b-chat.Q4_K_M.gguf
N_CTX=2048
N_GPU_LAYERS=35
TEMPERATURE=0.7

## Usage

### Interactive Chat

python -m src.cli chat ./models/model.gguf --n-ctx 2048

### Single Generation

python -m src.cli generate ./models/model.gguf "What is Python?"

## Testing

pytest tests/

## License

MIT
```

### Task 5.2: Create setup.py

```python
from setuptools import setup, find_packages

setup(
    name='llama-chat-app',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'llama-cpp-python>=0.2.0',
        'click>=8.0.0',
        'pydantic>=2.0.0',
        'python-dotenv>=1.0.0',
    ],
    entry_points={
        'console_scripts': [
            'llama-chat=src.cli:cli',
        ],
    },
)
```

---

## Challenges

### Challenge 1: Add Conversation History Persistence

Implement saving and loading conversation history to/from JSON files.

### Challenge 2: Add Progress Indicators

Use `tqdm` or `click.progressbar` to show progress during long generations.

### Challenge 3: Implement Retry Logic

Add automatic retry with exponential backoff for failed generations.

### Challenge 4: Add Metrics Collection

Track tokens/sec, response times, and error rates.

## Success Criteria

- [X] Project structure created
- [X] Configuration management working
- [X] LLaMA wrapper with error handling
- [X] CLI commands functional
- [X] Unit tests passing
- [X] Documentation complete

## Submission

Submit:
1. Complete source code
2. Test results screenshot
3. Sample conversation output
4. README.md

---

**Lab**: 8.1 - Python Application Development
**Module**: 08 - Integration & Applications
**Version**: 1.0
