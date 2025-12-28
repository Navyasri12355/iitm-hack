# Docker Setup for Clinical Evidence Copilot

This document explains how to set up and run the Clinical Evidence Copilot with Pathway integration using Docker.

## Prerequisites

- Docker Desktop installed and running
- Docker Compose v2.0 or higher

## Quick Start

1. **Build and start the services:**
   ```bash
   python docker-dev.py build
   python docker-dev.py start
   ```

2. **Create sample documents for testing:**
   ```bash
   python docker-dev.py samples
   ```

3. **Monitor document processing:**
   ```bash
   python docker-dev.py monitor
   ```

## Services

### clinical-evidence-copilot
- Main application service
- Runs the FastAPI web server
- Accessible at http://localhost:8000

### pathway-worker
- Dedicated service for Pathway document monitoring
- Monitors the `data/documents` folder for changes
- Processes documents in real-time using Pathway streaming

## Development Workflow

### 1. Initial Setup
```bash
# Build containers
python docker-dev.py build

# Start services
python docker-dev.py start

# Create sample documents
python docker-dev.py samples
```

### 2. Testing Document Processing
```bash
# Monitor logs to see document processing
python docker-dev.py logs pathway-worker

# Add a new document to data/documents/ and watch it get processed
# The system should detect and process it within 60 seconds
```

### 3. Running Tests
```bash
# Run tests inside the container
python docker-dev.py test

# Or run specific tests
docker-compose exec clinical-evidence-copilot python -m pytest tests/test_pathway_connector.py -v
```

### 4. Development Shell
```bash
# Open a shell in the container for debugging
python docker-dev.py shell
```

## File Structure

```
.
├── docker-compose.yml          # Service definitions
├── Dockerfile                  # Container build instructions
├── docker-dev.py              # Development helper script
├── test_docker_pathway.py     # Pathway testing script
├── data/
│   ├── documents/             # Monitored folder (mounted volume)
│   └── samples/               # Sample documents
└── src/
    └── ingestion/
        └── pathway_connector.py  # Pathway integration
```

## Environment Variables

Set these in your `.env` file:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Document Processing
DOCUMENTS_PATH=/app/data/documents
SUPPORTED_FORMATS=["pdf", "html", "xml", "txt"]

# Logging
LOG_LEVEL=INFO
```

## Troubleshooting

### Container Build Issues
```bash
# Clean build (removes cache)
docker-compose build --no-cache

# Check for build errors
docker-compose build --progress=plain
```

### Pathway Connection Issues
```bash
# Check pathway-worker logs
python docker-dev.py logs pathway-worker

# Test Pathway functionality
docker-compose exec pathway-worker python test_docker_pathway.py --test-only
```

### Volume Mount Issues
```bash
# Ensure directories exist
mkdir -p data/documents data/samples

# Check volume mounts
docker-compose exec clinical-evidence-copilot ls -la /app/data/
```

### Permission Issues
```bash
# Fix permissions on data directories
chmod -R 755 data/
```

## Monitoring Document Processing

The Pathway connector monitors the `data/documents` folder and processes:

1. **New documents** - Detected and indexed within 60 seconds
2. **Modified documents** - Re-indexed immediately (incremental)
3. **Deleted documents** - Removed from index immediately

### Testing Real-time Processing

1. Start monitoring:
   ```bash
   python docker-dev.py monitor
   ```

2. In another terminal, add a document:
   ```bash
   echo "Test medical document content" > data/documents/test.txt
   ```

3. Watch the logs for processing confirmation

4. Modify the document:
   ```bash
   echo "Updated content" >> data/documents/test.txt
   ```

5. Delete the document:
   ```bash
   rm data/documents/test.txt
   ```

Each action should be detected and processed in real-time.

## Production Considerations

For production deployment:

1. Use proper secrets management for API keys
2. Set up persistent volumes for document storage
3. Configure proper logging and monitoring
4. Use health checks for service availability
5. Consider scaling the pathway-worker service

## Next Steps

After verifying the Pathway connector works:

1. Implement task 3.2: Document validation and indexing pipeline
2. Add vector embedding generation
3. Integrate with OpenAI embeddings API
4. Set up vector store for search functionality