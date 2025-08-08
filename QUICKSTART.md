# Quick Start Guide - LLM Document Retrieval System

## 🚀 Get Started in 5 Minutes

### Step 1: Prerequisites
- Python 3.9+
- Azure OpenAI account with GPT and embedding models
- Pinecone account (or use FAISS locally)

### Step 2: Setup
```bash
# Clone and navigate to the project
cd bjaja-hackrx-6.0

# Run setup script
python setup.py

# Edit .env file with your API keys
# (The setup script will create .env from .env.example)
```

### Step 3: Configuration
Edit `.env` file with your credentials:
```env
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_OPENAI_API_DEPLOYMENT_NAME=gpt-5-mini
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=text-embedding-3-small
PINECONE_API_KEY=your-pinecone-api-key
```

### Step 4: Start the System
```bash
# Option 1: Use startup script
start.bat      # Windows
./start.sh     # Unix/Mac

# Option 2: Manual start
python main.py

# Option 3: With auto-reload
python -m uvicorn main:app --reload
```

### Step 5: Test the System
```bash
# Run comprehensive tests
python test_system.py

# Or visit the interactive docs
# http://localhost:8000/docs
```

## 📡 API Usage

### Authentication
All requests require a Bearer token in the header:
```
Authorization: Bearer 45373a6a634dff62b774382b7d01e401e8d1c711debf1d1d6c465b7a898e7620
```

### Sample Request
```bash
curl -X POST "http://localhost:8000/api/v1/hackrx/run" \
  -H "Authorization: Bearer 45373a6a634dff62b774382b7d01e401e8d1c711debf1d1d6c465b7a898e7620" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://hackrx.blob.core.windows.net/documents/policy.pdf",
    "questions": [
      "What is the grace period for premium payment?",
      "What is the waiting period for pre-existing diseases?",
      "Does this policy cover maternity expenses?"
    ]
  }'
```

### Expected Response
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

## 🔧 Development

### VS Code Tasks
- **Ctrl+Shift+P** → "Tasks: Run Task"
- Available tasks:
  - Setup Development Environment
  - Start LLM Document Retrieval System
  - Test System
  - Install Dependencies
  - Format Code
  - Lint Code

### Debugging
- Use F5 to debug with the configured launch profiles
- Available debug configs:
  - Debug LLM Document Retrieval System
  - Debug Test System
  - Run Setup

### Docker Deployment
```bash
# Build and run with Docker
docker-compose up --build

# Run with MongoDB cache
docker-compose --profile mongodb up

# Run with Redis cache
docker-compose --profile redis up
```

## 📊 Monitoring

### Health Check
```bash
curl http://localhost:8000/health
```

### Performance Metrics
```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
  http://localhost:8000/api/v1/metrics
```

### Interactive Documentation
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 🎯 Performance Targets

The system is designed to meet these performance targets:

| Phase | Target | Description |
|-------|---------|-------------|
| Validation | 0.5s | Request validation |
| Document Processing | 5.0s | PDF processing |
| Question Processing | 15.0s | Parallel questions |
| Response Compilation | 2.5s | Format response |
| **Total** | **23.0s** | **End-to-end** |

## 🆘 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   pip install -r requirements.txt
   ```

2. **Azure OpenAI Errors**
   - Check endpoint URL format
   - Verify API key and deployment names
   - Ensure quota is available

3. **Pinecone Errors**
   - Verify API key
   - Check index name and environment
   - Consider using FAISS: `USE_FAISS=true`

4. **Performance Issues**
   - Check system resources
   - Monitor logs for bottlenecks
   - Use `/api/v1/metrics` endpoint

### Logs
Check application logs for detailed error information:
```bash
tail -f logs/application.log
```

## 🎉 You're Ready!

The system is now running and ready to process documents. Start with the test script to verify everything is working, then integrate with your applications using the API endpoints.

For detailed documentation, see the full README.md file.
