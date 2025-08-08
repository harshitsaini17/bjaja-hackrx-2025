# LLM Document Retrieval System

## Overview

This is a high-performance LLM-powered document retrieval system designed for processing insurance, legal, HR, and compliance documents. The system achieves 20-25 second response times through optimized parallel processing and real-time document analysis.

## Features

- **Fast Document Processing**: 5-second PDF processing with parallel text extraction and embedding generation
- **Parallel Question Processing**: Up to 10 concurrent questions processed simultaneously
- **Multiple Vector Store Support**: Pinecone and FAISS backends
- **Domain-Specific Optimization**: Specialized for insurance, legal, HR, and compliance domains
- **Real-time Performance Monitoring**: Comprehensive metrics and bottleneck analysis
- **Robust Error Handling**: Timeout management and graceful degradation

## API Specifications

- **Endpoint**: `POST /api/v1/hackrx/run`
- **Authentication**: Bearer token required
- **Target Response Time**: 20-25 seconds
- **Maximum Questions**: 10 per request
- **Document Support**: PDF files via Azure Blob Storage URLs

### Sample Request

```json
{
  "documents": "https://hackrx.blob.core.windows.net/documents/policy.pdf",
  "questions": [
    "What is the grace period for premium payment?",
    "What is the waiting period for pre-existing diseases?",
    "Does this policy cover maternity expenses?"
  ]
}
```

### Sample Response

```json
{
  "answers": [
    "Grace period of thirty days is provided for premium payment...",
    "Waiting period of thirty-six months applies for pre-existing diseases...",
    "Yes, this policy covers maternity expenses after waiting period..."
  ],
  "metadata": {
    "processing_time": 23.4,
    "document_processed": true,
    "questions_count": 3
  }
}
```

## Quick Start

### Prerequisites

- Python 3.9+
- Azure OpenAI account with GPT and embedding models
- Pinecone account (or use FAISS locally)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd bjaja-hackrx-6.0
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment:
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

4. Start the server:
```bash
python main.py
```

The API will be available at `http://localhost:8000`

### Docker Deployment

```bash
docker build -t llm-document-retrieval .
docker run -p 8000:8000 --env-file .env llm-document-retrieval
```

## Configuration

Key configuration options in `.env`:

- **AZURE_OPENAI_ENDPOINT**: Your Azure OpenAI endpoint
- **AZURE_OPENAI_API_KEY**: Azure OpenAI API key  
- **PINECONE_API_KEY**: Pinecone API key (if using Pinecone)
- **USE_FAISS**: Set to `true` to use FAISS instead of Pinecone
- **MAX_CONCURRENT_QUESTIONS**: Maximum parallel question processing (default: 10)

## Performance Targets

| Phase | Target Time | Description |
|-------|-------------|-------------|
| Request Validation | 0.5s | Authentication and input validation |
| Document Processing | 5.0s | Download, extract, chunk, and embed |
| Question Processing | 15.0s | Parallel processing of all questions |
| Response Compilation | 2.5s | Format and validate response |
| **Total Response** | **23.0s** | **Complete end-to-end processing** |

## Architecture

### Core Components

1. **Document Processor**: Handles PDF download, text extraction, and chunking
2. **Vector Store Service**: Manages embeddings and similarity search
3. **LLM Service**: Interfaces with Azure OpenAI for answer generation
4. **Question Processor**: Orchestrates parallel question processing
5. **Performance Monitor**: Tracks metrics and bottlenecks

### Processing Pipeline

```
Request → Validation → Document Processing → Question Processing → Response
   0.5s      5.0s           15.0s              2.5s
```

## API Documentation

Once running, access interactive documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Monitoring

### Health Check
```bash
GET /health
```

### Performance Metrics
```bash
GET /api/v1/metrics
```

### Request-Specific Metrics
```bash
GET /api/v1/metrics/request/{request_id}
```

## Domain Specialization

The system is optimized for specific domains:

- **Insurance**: Premium calculations, coverage details, waiting periods
- **Legal**: Contract terms, obligations, compliance requirements  
- **HR**: Employee policies, benefits, procedures
- **Compliance**: Regulatory requirements, audit procedures, reporting

## Error Handling

The system provides robust error handling with:
- Automatic retries for transient failures
- Graceful degradation for partial failures
- Detailed error reporting and logging
- Timeout management at all levels

## Development

### Project Structure

```
src/
├── core/           # Configuration and authentication
├── models/         # Pydantic models and schemas
├── services/       # Core business logic services
└── utils/          # Utilities and helpers
```

### Running Tests

```bash
python -m pytest tests/
```

### Code Quality

```bash
# Format code
python -m black .

# Lint code  
python -m flake8 .
```

## Production Deployment

For production deployment:

1. Use a production WSGI server (e.g., Gunicorn)
2. Set up proper logging and monitoring
3. Configure load balancing for high availability
4. Use a managed vector database (Pinecone recommended)
5. Implement proper security measures
6. Set up automated backups

## Support

For issues and questions, please refer to the project documentation or contact the development team.

## License

MIT License - see LICENSE file for details.
