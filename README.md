# Clinical Evidence Copilot

An agentic AI system that provides clinicians with real-time, evidence-backed medical information by continuously ingesting and analyzing live medical research, clinical trials, and treatment guidelines.

## Project Structure

```
clinical-evidence-copilot/
├── src/
│   ├── api/           # FastAPI REST endpoints and WebSocket handlers
│   ├── ingestion/     # Document processing and Pathway streaming
│   ├── models/        # Data models and validation
│   ├── reasoning/     # Agentic reasoning engine and medical intelligence
│   ├── services/      # Business logic and external service integrations
│   └── config.py      # Configuration management
├── data/
│   ├── documents/     # Medical literature input folder
│   └── samples/       # Sample documents for testing
├── tests/             # Unit and property-based tests
├── requirements.txt   # Python dependencies
└── .env.example       # Environment variables template
```

## Setup Instructions

### Prerequisites

- Python 3.10 or higher
- Virtual environment (already created in `venv/`)

### Installation

1. **Activate the virtual environment:**
   ```bash
   # On Windows
   venv\Scripts\activate
   
   # On Linux/MacOS
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your actual API keys and settings
   ```

### Important Note: Pathway Framework Limitation

⚠️ **Windows Compatibility**: The Pathway framework (core streaming engine) currently only supports Linux and MacOS. Windows users have the following options:

1. **Use Windows Subsystem for Linux (WSL)** - Recommended
2. **Use Docker** with Linux containers
3. **Use a Linux VM**
4. **Develop on a Linux/MacOS machine**

For development on Windows without WSL, the project structure and other components can be developed, but the Pathway streaming functionality will need to be run in a Linux environment.

## Configuration

The application uses environment variables for configuration. Key settings include:

- `OPENAI_API_KEY`: Your OpenAI API key for embeddings and LLM calls
- `DOCUMENTS_PATH`: Path to folder containing medical literature
- `RESPONSE_TIMEOUT_SECONDS`: Maximum time for query responses (default: 30s)
- `ENABLE_HIPAA_LOGGING`: Enable HIPAA-compliant logging (default: true)

See `.env.example` for all available configuration options.

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run property-based tests with verbose output
pytest -v tests/ -k "property"

# Run with coverage
pytest --cov=src tests/
```

### Starting the Development Server

```bash
# Start FastAPI server
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

## Architecture

The system follows a microservices architecture with:

1. **Data Ingestion Layer**: Real-time document processing with Pathway
2. **Agentic Reasoning Layer**: Multi-step query processing and medical intelligence
3. **API Layer**: REST endpoints and WebSocket connections
4. **Client Layer**: Web interface and EHR integrations
