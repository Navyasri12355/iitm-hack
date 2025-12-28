# Implementation Plan

- [x] 1. Set up project structure and dependencies














  - Create Python project with virtual environment
  - Install core dependencies: pathway, openai, fastapi, uvicorn, python-multipart
  - Set up basic directory structure for components
  - Create configuration management for API keys and settings
  - _Requirements: All requirements depend on basic project setup_




- [x] 2. Implement core data models and document processing






  - [x] 2.1 Create data models for documents, queries, and recommendations

    - Define ParsedDocument, ClinicalQuery, Evidence, ClinicalRecommendation classes
    - Implement basic validation and serialization methods
    - _Requirements: 6.1, 6.2, 6.3_

  - [ ]* 2.2 Write property test for document parsing round-trip
    - **Property 22: Multi-format parsing accuracy**
    - **Validates: Requirements 6.1, 6.2**

  - [x] 2.3 Implement document parser with multi-format support



    - Create UnstructuredParser wrapper for PDF, HTML, XML parsing
    - Add metadata extraction for medical documents
    - Implement error handling and alternative parsing methods
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

  - [ ]* 2.4 Write property test for metadata extraction
    - **Property 23: Comprehensive metadata extraction**
    - **Validates: Requirements 6.3**



- [x] 3. Build Pathway streaming document ingestion




  - [x] 3.1 Set up Pathway file system connector






    - Configure folder monitoring for medical documents
    - Implement real-time document detection and processing
    - _Requirements: 2.1, 2.2, 2.3_

  - [x] 3.2 Create document validation and indexing pipeline
    - Implement document authenticity validation
    - Set up vector embedding generation using OpenAI embeddings
    - Create Pathway vector store for real-time indexing
    - _Requirements: 2.4, 2.5_

  - [ ]* 3.3 Write property test for indexing performance
    - **Property 6: Document indexing performance**
    - **Validates: Requirements 2.1**

  - [ ]* 3.4 Write property test for incremental updates
    - **Property 7: Incremental re-indexing**
    - **Validates: Requirements 2.2**




- [-] 4. Implement agentic reasoning engine




  - [x] 4.1 Create query decomposition and routing logic
    - Implement query analysis and sub-task breakdown
    - Create search, filter, rank, summarize pipeline
    - Add ambiguity detection and clarification requests
    - _Requirements: 3.1, 3.2_

  - [ ]* 4.2 Write property test for query decomposition
    - **Property 11: Query decomposition**
    - **Validates: Requirements 3.1**


  - [x] 4.3 Build evidence retrieval and ranking system

    - Implement vector similarity search with medical context
    - Create evidence hierarchy ranking (systematic reviews > RCTs > observational)
    - Add contradiction detection between studies
    - _Requirements: 1.3, 1.4, 3.5_

  - [ ]* 4.4 Write property test for evidence ranking

    - **Property 3: Evidence-based ranking**

    - **Validates: Requirements 1.3**

  - [x] 4.5 Create recommendation generation with reasoning

    - Implement LLM-based recommendation synthesis
    - Add reasoning explanation and citation generation
    - Include confidence scoring and change tracking
    - _Requirements: 1.2, 3.4, 4.2, 4.3_

  - [ ]* 4.6 Write property test for citation completeness
    - **Property 2: Citation completeness**

    - **Validates: Requirements 1.2**

- [ ] 5. Build FastAPI web service


  - [x] 5.1 Create REST API endpoints
    - Implement /query endpoint for clinical questions
    - Add /documents endpoint for document management
    - Create /recommendations endpoint for tracking changes
    - _Requirements: 5.1, 5.3_

  - [x] 5.2 Add real-time updates with WebSocket support
    - Implement WebSocket connections for live recommendation updates
    - Create notification system for evidence changes
    - Add user session management
    - _Requirements: 1.5, 4.1, 4.4_

  - [ ]* 5.3 Write property test for response time compliance
    - **Property 1: Query response time compliance**
    - **Validates: Requirements 1.1**


- [-] 6. Create dynamic, colourful and interactive web interface for demonstration


  - [x] 6.1 Build attractive HTML/JavaScript frontend
    - Create query input form and results display
    - Add real-time updates display for recommendation changes
    - Implement document upload interface for testing
    - _Requirements: Demo and testing purposes_

  - [x] 6.2 Add mock medical document samples
    - Create sample research papers, guidelines, and clinical trials
    - Include documents with contradictory findings for testing
    - Add various formats (PDF, HTML, XML) for parser testing
    - _Requirements: Testing and demonstration_


- [-] 7. Integration and testing checkpoint


  - [ ] 7.1 Ensure all tests pass, ask the user if questions arise
    - Run all property-based tests with 100+ iterations
    - Verify end-to-end workflow from document ingestion to query response
    - Test live update functionality with document changes
    - _Requirements: All requirements validation_

  - [x] 7.2 Create demonstration script
    - Prepare sample clinical queries for demo
    - Set up document addition/modification scenarios
    - Create script showing live evidence updates affecting recommendations
    - _Requirements: Demo preparation_

- [ ]* 8. Additional property tests for comprehensive coverage
  - [ ]* 8.1 Write property test for contradiction detection
    - **Property 4: Contradiction detection and explanation**
    - **Validates: Requirements 1.4**

  - [ ]* 8.2 Write property test for live updates
    - **Property 5: Live recommendation updates with notification**
    - **Validates: Requirements 1.5, 4.4**

  - [ ]* 8.3 Write property test for system availability
    - **Property 10: System availability during ingestion**
    - **Validates: Requirements 2.5**

  - [ ]* 8.4 Write property test for error handling
    - **Property 13: Graceful error handling**
    - **Validates: Requirements 3.3**