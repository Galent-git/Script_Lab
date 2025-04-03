// --- Constants and Configuration ---
ZOTERO_API_KEY = getUserInput("Enter Zotero API Key") // Or load from secure config
ZOTERO_USER_ID = getUserInput("Enter Zotero User/Group ID") // Or load from config
GOOGLE_API_KEY = getUserInput("Enter Google AI API Key") // LOAD SECURELY!
LOCAL_ZOTERO_STORAGE_PATH = getUserInput("Path to Zotero local storage/PDFs") // Or detect
CACHE_DIRECTORY = "./.zotero_agent_cache/"
VECTOR_STORE_PATH = CACHE_DIRECTORY + "vector_store"
SELECTED_GPU = detectOrSelectGPU() // Still useful for local embeddings, QA, summarization if kept local

// --- Initialization ---
Function initialize():
    createDirectoryIfNotExists(CACHE_DIRECTORY)
    zotero = initializeZoteroClient(ZOTERO_API_KEY, ZOTERO_USER_ID)
    
    // Initialize Google Generative AI Client
    google_ai_client = initializeGoogleAIClient(GOOGLE_API_KEY)
    // Specify Gemini model (e.g., 'gemini-1.5-flash' for speed/cost or 'gemini-pro' for capability)
    gemini_model_name = "gemini-1.5-flash" 

    // Load models for LOCAL processing (GPU acceleration useful here)
    embedding_model = loadEmbeddingModel("all-mpnet-base-v2", device=SELECTED_GPU) // For local retrieval
    // Keep local summarizer/QA optional, or rely on Gemini for these tasks too via specific prompts?
    // Decision Point: Keep local models for speed on single docs, or use Gemini for everything?
    // Let's assume we keep local embedding for RAG retrieval speed and control.
    // Summarizer/QA could be replaced by specific Gemini calls if desired.
    // summarizer_model = loadSummarizationModel("facebook/bart-large-cnn", device=SELECTED_GPU) // Optional local
    // qa_model = loadQuestionAnsweringModel("deepset/roberta-base-squad2", device=SELECTED_GPU) // Optional local
    ner_model = loadNerModel("en_core_sci_lg") // spaCy NER likely still useful locally

    // Initialize or load vector store (FAISS, ChromaDB)
    vector_store = initializeVectorStore(VECTOR_STORE_PATH, embedding_model.dimension)

    // Load cache index
    cache_index = loadCacheIndex(CACHE_DIRECTORY + "cache_index.json")

    print("Initialization Complete. Using Google Gemini API for generation.")
    # REMINDER: Inform user about potential API costs and data privacy (sending text snippets to Google).
    print("WARNING: Text snippets will be sent to Google Gemini API for analysis.")

    return zotero, google_ai_client, gemini_model_name, embedding_model, /*summarizer_model, qa_model,*/ ner_model, vector_store, cache_index


// --- Core Data Structures (remain similar) ---
// cache_index: Dict[item_key, {...}]
// paper_data: Dict[item_key, {...}]

// --- Zotero Interaction (remains the same) ---
Function getCollections(zotero_client): pass
Function getCollectionItems(zotero_client, collection_id): pass

// --- Caching (remains the same) ---
Function checkCache(item_key, data_type, cache_index): pass
Function saveToCache(item_key, data_type, data, cache_index): pass

// --- PDF Processing (remains the same) ---
Function extractTextFromPDF(pdf_path): pass

// --- LOCAL NLP Analysis (Embeddings, maybe NER) ---
Function generateEmbeddings(text_chunks, embedding_model, batch_size=32): pass // Uses local GPU
Function extractEntities(text, ner_model): pass // Uses local spaCy
Function chunkText(text, strategy='paragraph', chunk_size=500): pass

// --- Core Processing Pipeline (remains similar, focuses on local tasks) ---
Function processPaper(item_metadata, cache_index, force_reprocess=False):
    // ... (steps 1-4: get PDF path, extract text, chunk text - remain the same) ...
    
    // 5. Generate Embeddings locally (if chunks exist, Cache Check)
    cached_embeddings = checkCache(item_key, "embeddings", cache_index)
    if cached_embeddings and not force_reprocess:
        processed_data["embeddings"] = cached_embeddings
        addEmbeddingsToVectorStore(vector_store, item_key, processed_data["chunks"], cached_embeddings)
    elif processed_data["chunks"]:
        embeddings = generateEmbeddings(processed_data["chunks"], embedding_model) // Local GPU
        processed_data["embeddings"] = embeddings
        saveToCache(item_key, "embeddings", embeddings, cache_index)
        addEmbeddingsToVectorStore(vector_store, item_key, processed_data["chunks"], embeddings)

    // 6. NER Extraction (Optional, On-demand, Local)
    // ...

    return processed_data


// --- Vector Store Operations (remain the same) ---
Function addEmbeddingsToVectorStore(vector_store, item_key, chunks, embeddings): pass
Function searchVectorStore(query, vector_store, embedding_model, top_k=5): pass

// --- RAG using Google Gemini API ---
Function performRAGWithGemini(query, vector_store, embedding_model, google_ai_client, gemini_model_name, top_k=5):
    // 1. Retrieve relevant chunks LOCALLY
    print("Retrieving relevant context locally...")
    retrieved_chunks_metadata = searchVectorStore(query, vector_store, embedding_model, top_k=top_k)
    if not retrieved_chunks_metadata:
        return "Could not find relevant information in the processed documents.", []

    context = "\n---\n".join([f"Source: {chunk_meta['item_key']}, Chunk: {chunk_meta['text']}" for chunk_meta in retrieved_chunks_metadata])
    source_keys = list(set([chunk_meta['item_key'] for chunk_meta in retrieved_chunks_metadata]))

    // 2. Build Prompt for Gemini
    # Consider safety settings and instruction tuning
    prompt = f"""You are a scientific research assistant. Analyze the following context extracted from research papers ({', '.join(source_keys)}) and answer the user's query based *only* on this context. Do not use external knowledge. Cite the source paper key(s) relevant to your answer. If the context doesn't provide the answer, state that clearly.

--- Context Start ---
{context}
--- Context End ---

User Query: {query}

Answer:"""

    // 3. Call Google Gemini API
    print(f"Sending request to Google Gemini ({gemini_model_name})...")
    # !! PRIVACY WARNING: 'prompt' containing text snippets is sent externally !!
    try:
        gemini_model = google_ai_client.GenerativeModel(gemini_model_name)
        # Configure safety settings if needed
        # response = gemini_model.generate_content(prompt, safety_settings=...) 
        response = gemini_model.generate_content(prompt) 
        
        # Basic error checking/content filtering check
        if not response.candidates or not response.candidates[0].content.parts:
             # Handle cases where the API blocked the response or returned empty
             error_reason = response.prompt_feedback if response.prompt_feedback else "Unknown reason (possibly content filtering or API error)"
             print(f"Gemini API Error/Block: {error_reason}")
             return f"Failed to get response from Gemini API. Reason: {error_reason}", source_keys
             
        generated_text = response.text # Accessing the text part directly

    except Exception as e:
        print(f"Error calling Google Gemini API: {e}")
        return f"Error communicating with Gemini API: {e}", source_keys

    print("Received response from Gemini.")
    return generated_text, source_keys

// --- Specific Task Functions (Optionally using Gemini) ---
Function summarizeWithGemini(text, google_ai_client, gemini_model_name, item_key="Unknown"):
    prompt = f"""Summarize the key findings and conclusions from the following text excerpt from paper {item_key}. Be concise and accurate.

Text:
{text}

Summary:"""
    print(f"Sending summary request to Gemini for {item_key}...")
    # Call Gemini API similar to performRAGWithGemini
    try:
        gemini_model = google_ai_client.GenerativeModel(gemini_model_name)
        response = gemini_model.generate_content(prompt)
        if not response.candidates or not response.candidates[0].content.parts:
             error_reason = response.prompt_feedback if response.prompt_feedback else "Unknown reason"
             return f"Failed to get summary from Gemini API. Reason: {error_reason}"
        return response.text
    except Exception as e:
        return f"Error communicating with Gemini API for summary: {e}"


Function extractWithGemini(text, extraction_query, google_ai_client, gemini_model_name, item_key="Unknown"):
    prompt = f"""From the following text excerpt from paper {item_key}, extract the information relevant to the query: '{extraction_query}'. List the key points or provide a concise answer based *only* on the text.

Text:
{text}

Relevant Information for '{extraction_query}':"""
    print(f"Sending extraction request to Gemini for {item_key}...")
    # Call Gemini API similar to performRAGWithGemini
    try:
        gemini_model = google_ai_client.GenerativeModel(gemini_model_name)
        response = gemini_model.generate_content(prompt)
        if not response.candidates or not response.candidates[0].content.parts:
             error_reason = response.prompt_feedback if response.prompt_feedback else "Unknown reason"
             return f"Failed to get extraction from Gemini API. Reason: {error_reason}"
        return response.text
    except Exception as e:
        return f"Error communicating with Gemini API for extraction: {e}"

// --- Report Generation (remains the same conceptually) ---
Function generatePDFReport(data, template_name, output_path): pass

// --- Main Interaction Loop (Updated Calls) ---
Function main():
    // Initialize with Gemini client instead of local LLM
    zotero, google_ai_client, gemini_model_name, embedding_model, /*optional models,*/ ner_model, vector_store, cache_index = initialize()
    active_collection_items = [] 

    while True:
        displayInterface() 
        user_input = getUserInput("> ") 

        // ... (list_collections, select_collection, process_pdfs remain similar, using local processing) ...

        elif user_input.command == "summarize":
            target_items = selectItemsFromList(active_collection_items, user_input.item_keys_or_all)
            summaries = {}
            for item_data in target_items:
                item_key = item_data['metadata']['item_key']
                cached_summary = checkCache(item_key, "summary_gemini", cache_index) # Use different cache key?
                if cached_summary:
                    summaries[item_key] = cached_summary
                elif item_data.get("text"):
                    # Use Gemini for summarization
                    summary = summarizeWithGemini(item_data["text"], google_ai_client, gemini_model_name, item_key)
                    summaries[item_key] = summary
                    # Decide whether/how to cache API results (cost vs computation)
                    saveToCache(item_key, "summary_gemini", summary, cache_index) 
                else:
                    summaries[item_key] = "Error: No text available to summarize."
            displaySummaries(summaries)

        elif user_input.command == "extract_methods" or user_input.command == "extract_findings":
             target_items = selectItemsFromList(active_collection_items, user_input.item_keys) 
             extracted_info = {}
             extraction_query = "What methods were used?" if user_input.command == "extract_methods" else "What were the key findings reported?"
             for item_data in target_items:
                 item_key = item_data['metadata']['item_key']
                 if item_data.get("text"):
                     # Use Gemini for extraction
                     answer = extractWithGemini(item_data["text"], extraction_query, google_ai_client, gemini_model_name, item_key)
                     extracted_info[item_key] = answer
                     # Cache API result?
                 else:
                     extracted_info[item_key] = "Error: No text available."
             displayExtractionResults(extracted_info)

        elif user_input.command == "discuss" or user_input.command == "ask":
            query = user_input.query_text
            if not vector_store or vector_store.isEmpty():
                 print("Error: No papers processed and indexed for discussion. Please process PDFs first.")
                 continue
            # Use RAG with Gemini
            response, sources = performRAGWithGemini(query, vector_store, embedding_model, google_ai_client, gemini_model_name)
            displayDiscussion(query, response, sources) # Display API response

        elif user_input.command == "generate_report":
             report_type = user_input.report_type 
             target_items = selectItemsFromList(active_collection_items, user_input.item_keys_or_all)
             # Report data preparation might involve calling Gemini functions if summaries/extractions weren't cached
             report_data = prepareReportData(report_type, target_items, cache_index, google_ai_client, gemini_model_name) 
             output_filename = f"Report_{report_type}_{user_input.collection_id}.pdf"
             generatePDFReport(report_data, f"template_{report_type}.html", output_filename)

        elif user_input.command == "exit":
            break
        else:
            print("Unknown command.")

    print("Exiting Zotero Agent.")


// --- Helper Functions (Placeholders, add Google AI init) ---
Function initializeGoogleAIClient(api_key): pass // Use google.generativeai.configure(api_key=...)
// ... other helpers remain largely the same ...
Function prepareReportData(report_type, items, cache, google_client, model_name): pass // May need API client now

// --- Entry Point ---
if __name__ == "__main__":
    main()
