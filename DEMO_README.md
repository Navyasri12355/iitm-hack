# Clinical Evidence Copilot Demonstration

This directory contains comprehensive demonstration scripts for the Clinical Evidence Copilot system. The demonstration showcases all key capabilities including real-time evidence processing, contradiction detection, agentic reasoning, and live recommendation updates.

## Quick Start

The easiest way to run the demonstration:

```bash
python setup_demo.py
```

This will:
1. Check all requirements
2. Start necessary services automatically
3. Present demo options
4. Clean up when finished

## Demo Components

### 1. Core Demo Scripts

- **`demo_script.py`** - Main demonstration logic with all capabilities
- **`run_demo.py`** - Demo runner with multiple modes (interactive, automated, scenario, benchmark)
- **`setup_demo.py`** - Automated setup and service management
- **`demo_scenarios.json`** - Comprehensive demo scenarios and test cases

### 2. Supporting Files

- **`mock_api_server.py`** - Mock API server for demonstration
- **`simple_server.py`** - Static file server for web interface
- **`static/index.html`** - Interactive web interface

## Demonstration Modes

### Interactive Mode (Recommended for Presentations)
```bash
python run_demo.py --mode interactive
```
- Guided walkthrough with explanations
- User prompts between phases
- 15-20 minute duration
- Best for first-time users and presentations

### Automated Mode (Quick Overview)
```bash
python run_demo.py --mode automated
```
- Complete demonstration running automatically
- No user interaction required
- 10-15 minute duration
- Best for quick overviews

### Scenario Mode (Focused Testing)
```bash
python run_demo.py --mode scenario --scenario contradiction_detection
```
Available scenarios:
- `basic_queries` - Basic clinical query processing
- `live_ingestion` - Document ingestion and indexing
- `contradiction_detection` - Conflicting evidence handling
- `agentic_reasoning` - Multi-step reasoning demonstration
- `recommendation_evolution` - Recommendation change tracking

### Benchmark Mode (Performance Testing)
```bash
python run_demo.py --mode benchmark
```
- Tests query response times
- Document indexing performance
- Success rates and system metrics
- 5-10 minute duration

### Web Interface Mode
```bash
# Start services first
python mock_api_server.py &
python simple_server.py &

# Then open http://localhost:8080 in browser
```
- Interactive web interface
- Manual query testing
- Real-time WebSocket updates
- Document upload testing

## Key Demonstration Features

### 1. Clinical Query Processing
- **Evidence-backed recommendations** with citations
- **Response time < 30 seconds** for clinical queries
- **Confidence scoring** for all recommendations
- **Multi-step reasoning** process visibility

### 2. Live Document Ingestion
- **Real-time indexing** of new medical literature
- **Automatic metadata extraction** from documents
- **Document credibility scoring** based on source and content
- **Immediate availability** for query processing

### 3. Recommendation Updates
- **Live updates** when new evidence affects existing recommendations
- **WebSocket notifications** to connected clinicians
- **Change reason explanations** for all updates
- **Recommendation versioning** and history tracking

### 4. Contradiction Detection
- **Automatic identification** of conflicting evidence
- **Detailed explanations** of contradictions
- **Risk-benefit analysis** when evidence conflicts
- **Clinical decision support** for complex cases

### 5. Agentic Reasoning
- **Query decomposition** into logical sub-tasks
- **Multi-step processing** (search → filter → rank → summarize)
- **Ambiguity detection** and clarification requests
- **Evidence synthesis** across multiple conditions

## Sample Clinical Scenarios

The demonstration includes realistic clinical scenarios:

### Hypertension Management
- **Query**: "What is the recommended first-line treatment for hypertension in elderly patients over 65?"
- **Demonstrates**: Evidence-based ranking, age-specific recommendations
- **Expected Evidence**: ACE inhibitor effectiveness, safety profiles, dosing considerations

### Diabetes Guidelines
- **Query**: "What are the current ADA guidelines for HbA1c targets in adults with type 2 diabetes?"
- **Demonstrates**: Current guideline integration, personalized medicine
- **Expected Evidence**: ADA 2024 standards, individualized targets, monitoring strategies

### Aspirin Controversy
- **Query**: "Should I prescribe aspirin for primary prevention in a 55-year-old with 12% ASCVD risk?"
- **Demonstrates**: Contradiction detection, risk-benefit analysis
- **Expected Evidence**: Conflicting studies on bleeding risk vs cardiovascular benefit

### Complex Comorbidities
- **Query**: "Best glucose medication for 68-year-old with diabetes, heart failure, and kidney disease?"
- **Demonstrates**: Multi-step reasoning, comorbidity integration
- **Expected Evidence**: SGLT2 inhibitor benefits, cardiovascular outcomes, kidney protection

## Live Update Scenarios

### New Evidence Integration
The demo shows how new medical literature automatically updates recommendations:

1. **Baseline Query**: Ask about hypertension treatment in elderly
2. **Add New Evidence**: Upload study showing ARB superiority over ACE inhibitors
3. **Updated Recommendation**: System automatically revises recommendation
4. **Notification**: WebSocket alert about recommendation change
5. **Explanation**: Clear reasoning for the update

### Contradiction Detection
Demonstrates handling of conflicting evidence:

1. **Conflicting Studies**: Aspirin primary prevention benefits vs bleeding risks
2. **Automatic Detection**: System identifies contradictory findings
3. **Risk Analysis**: Weighs benefits against potential harms
4. **Clinical Guidance**: Provides decision support despite uncertainty

## Performance Metrics

The demonstration validates key performance requirements:

### Response Times
- **Query Processing**: < 30 seconds (typically 2-5 seconds)
- **Document Indexing**: < 60 seconds (typically 10-30 seconds)
- **Live Updates**: < 5 seconds for recommendation changes

### Accuracy Metrics
- **Citation Completeness**: All recommendations include source citations
- **Evidence Quality**: Systematic reviews and RCTs prioritized
- **Contradiction Detection**: Conflicting evidence automatically flagged
- **Reasoning Transparency**: Multi-step process fully visible

### System Reliability
- **Availability**: > 99% uptime during demonstration
- **WebSocket Stability**: Persistent connections for live updates
- **Error Handling**: Graceful degradation and recovery
- **Scalability**: Handles multiple concurrent queries

## Technical Requirements

### System Requirements
- Python 3.8+
- 4GB RAM minimum (8GB recommended)
- Network access for API calls
- Modern web browser for interface

### Python Dependencies
```bash
pip install fastapi uvicorn websockets requests pathlib asyncio
```

### Port Usage
- **8001**: Mock API server
- **8080**: Frontend web server
- **WebSocket**: Same as API server (8001)

## Troubleshooting

### Common Issues

**Services won't start:**
```bash
# Check if ports are in use
netstat -an | grep :8001
netstat -an | grep :8080

# Kill existing processes if needed
pkill -f mock_api_server.py
pkill -f simple_server.py
```

**WebSocket connection fails:**
- Ensure API server is running on port 8001
- Check firewall settings
- Verify WebSocket support in browser

**Demo scripts fail:**
- Verify all required files are present
- Check Python version (3.8+ required)
- Install missing dependencies
- Ensure sufficient disk space

### Performance Issues

**Slow query responses:**
- Check system resources (CPU, memory)
- Verify network connectivity
- Reduce concurrent queries
- Clear browser cache

**Document indexing delays:**
- Check available disk space
- Verify document format compatibility
- Reduce document size for testing
- Monitor system logs for errors

## Demo Customization

### Adding New Scenarios
Edit `demo_scenarios.json` to add custom clinical queries:

```json
{
  "id": "custom_scenario",
  "query_text": "Your clinical question here",
  "urgency_level": "routine",
  "demonstrates": ["capability1", "capability2"]
}
```

### Modifying Test Documents
Add new medical literature to `demo_documents` array:

```json
{
  "title": "Your Study Title",
  "content": "Study content...",
  "document_type": "systematic_review",
  "demonstrates": "Live updates, contradiction detection"
}
```

### Custom Performance Tests
Modify `_run_benchmark_demo()` in `run_demo.py` to add custom performance tests.

## Support and Documentation

For questions about the demonstration:

1. **Check logs** in console output for error details
2. **Verify setup** using `python setup_demo.py`
3. **Test components** individually using scenario mode
4. **Review requirements** in this README

The demonstration provides a comprehensive showcase of the Clinical Evidence Copilot's capabilities, suitable for presentations, testing, and validation of the system's clinical decision support features.