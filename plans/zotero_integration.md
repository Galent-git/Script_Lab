// --- Constants and Configuration ---
ZOTERO_API_KEY = getUserInput("Enter Zotero API Key") // Or load from secure config
ZOTERO_USER_ID = getUserInput("Enter Zotero User/Group ID") // Or load from config
LOCAL_ZOTERO_STORAGE_PATH = getUserInput("Path to Zotero local storage/PDFs") // Or detect
CACHE_DIRECTORY = "./.zotero_agent_cache/"
VECTOR_STORE_PATH = CACHE_DIRECTORY + "vector_store"
SELECTED_GPU = detectOrSelectGPU() // Logic to choose RTX 4060 if available, else 2060

// --- Initialization ---
Function initialize():
    createDirectoryIfNotExists(CACHE_DIRECTORY)
    zotero = initializeZoteroClient(ZOTERO_API_KEY, ZOTERO_USER_ID)
    
    // Load models onto GPU (adjust model names based on VRAM and desired quality)
    // Use libraries like Hugging Face transformers, sentence-transformers
    embedding_model = loadEmbeddingModel("all-mpnet-base-v2", device=SELECTED_GPU) // Good general model
    summarizer_model = loadSummarizationModel("facebook/bart-large-cnn", device=SELECTED_GPU) // Good quality summarizer
    qa_model = loadQuestionAnsweringModel("deepset/roberta-base-squad2", device=SELECTED_GPU) // QA model
    // Consider NER model like scispaCy if needed (might be CPU-bound or require specific GPU setup)
    ner_model = loadNerModel("en_core_sci_lg") // Example scispaCy model

    // Load Local LLM (using llama-cpp-python, ctransformers, ollama etc. with GPU offloading)
    // Choose model based on VRAM (e.g., Mistral-7B-Instruct-v0.2-GGUF with Q4_K_M quantization for 8GB VRAM)
    local_llm = loadLocalLLM("path/to/quantized_model.gguf", gpu_layers=-1) // Offload all layers to GPU if possible

    // Initialize or load vector store (FAISS, ChromaDB)
    vector_store = initializeVectorStore(VECTOR_STORE_PATH, embedding_model.dimension)

    // Load cache index (e.g., a dictionary mapping item_key to cached data paths/status)
    cache_index = loadCacheIndex(CACHE_DIRECTORY + "cache_index.json")

    print("Initialization Complete.")
    return zotero, embedding_model, summarizer_model, qa_model, ner_model, local_llm, vector_store, cache_index

// --- Core Data Structures ---
// cache_index: Dict[item_key, {"metadata": timestamp, "pdf_path": path, "extracted_text_path": path, "summary_path": path, "embeddings_path": path, "ner_results_path": path}]
// paper_data: Dict[item_key, {"metadata": {...}, "pdf_path": "...", "text": "...", "summary": "...", "entities": [...], "chunks": [...]}] // In-memory representation for active use

// --- Zotero Interaction ---
Function getCollections(zotero_client):
    collections = zotero_client.collections()
    return collections // List of collection names/IDs

Function getCollectionItems(zotero_client, collection_id):
    items_metadata = zotero_client.collection_items(collection_id)
    // Enhance metadata: try to find local PDF path based on Zotero data structure / linked attachments
    for item in items_metadata:
        item['local_pdf_path'] = findLocalPdfPath(item, LOCAL_ZOTERO_STORAGE_PATH)
        item['item_key'] = item['key'] // Use Zotero item key as unique ID
    return items_metadata // List of dicts with metadata and potential PDF path

// --- Caching ---
Function checkCache(item_key, data_type, cache_index): // data_type e.g., "text", "summary", "embeddings"
    if item_key in cache_index and data_type + "_path" in cache_index[item_key]:
        return loadDataFromCacheFile(cache_index[item_key][data_type + "_path"])
    return None

Function saveToCache(item_key, data_type, data, cache_index):
    filepath = CACHE_DIRECTORY + item_key + "_" + data_type + ".cache"
    saveDataToCacheFile(data, filepath)
    if item_key not in cache_index: cache_index[item_key] = {}
    cache_index[item_key][data_type + "_path"] = filepath
    saveCacheIndex(cache_index, CACHE_DIRECTORY + "cache_index.json")

// --- PDF Processing ---
Function extractTextFromPDF(pdf_path):
    if not pdf_path or not fileExists(pdf_path): return None
    try:
        text = usePyMuPDF(pdf_path) // Or other PDF library
        // Basic cleaning (remove headers/footers, weird line breaks) could go here
        // Check for OCR necessity if text is too short/garbled? (Advanced)
        // if needsOCR(text): text = runOCR(pdf_path) // Optional, Tesseract
        return text
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {e}")
        return None

// --- NLP Analysis (GPU Accelerated) ---
Function generateEmbeddings(text_chunks, embedding_model, batch_size=32):
    // Use sentence-transformers library with GPU
    embeddings = embedding_model.encode(text_chunks, batch_size=batch_size, show_progress_bar=True, device=SELECTED_GPU)
    return embeddings

Function summarizeText(text, summarizer_model, max_length=250, min_length=50):
    // Use Hugging Face pipeline with GPU
    summary = summarizer_model(text, max_length=max_length, min_length=min_length, truncation=True)[0]['summary_text']
    return summary

Function extractEntities(text, ner_model):
    // Use spaCy (may run mostly on CPU unless using specific GPU libraries like spacy-transformers)
    doc = ner_model(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

Function answerQuestion(question, context, qa_model):
    // Use Hugging Face pipeline with GPU
    result = qa_model(question=question, context=context)
    return result['answer'] // Potentially return score as well

Function chunkText(text, strategy='paragraph', chunk_size=500): // strategy='sentence', 'fixed_size' etc.
    // Simple splitting logic
    chunks = splitTextIntoChunks(text, strategy, chunk_size)
    return chunks

// --- Core Processing Pipeline ---
Function processPaper(item_metadata, cache_index, force_reprocess=False):
    item_key = item_metadata['item_key']
    processed_data = {"metadata": item_metadata}

    // 1. Get PDF Path
    processed_data["pdf_path"] = item_metadata.get('local_pdf_path')

    // 2. Extract Text (Cache Check)
    cached_text = checkCache(item_key, "extracted_text", cache_index)
    if cached_text and not force_reprocess:
        processed_data["text"] = cached_text
    else:
        extracted_text = extractTextFromPDF(processed_data["pdf_path"])
        if extracted_text:
            processed_data["text"] = extracted_text
            saveToCache(item_key, "extracted_text", extracted_text, cache_index)
        else:
            processed_data["text"] = None // Mark as failed/no text

    // 3. Generate Summary (Optional, On-demand or batch, Cache Check)
    // Could be triggered later by specific requests

    // 4. Chunk Text (if text exists)
    if processed_data["text"]:
        processed_data["chunks"] = chunkText(processed_data["text"])
    else:
        processed_data["chunks"] = []

    // 5. Generate Embeddings (if chunks exist, Cache Check)
    cached_embeddings = checkCache(item_key, "embeddings", cache_index)
    if cached_embeddings and not force_reprocess:
        processed_data["embeddings"] = cached_embeddings
        // Add to vector store if not already there (requires tracking state)
        addEmbeddingsToVectorStore(vector_store, item_key, processed_data["chunks"], cached_embeddings)
    elif processed_data["chunks"]:
        embeddings = generateEmbeddings(processed_data["chunks"], embedding_model)
        processed_data["embeddings"] = embeddings
        saveToCache(item_key, "embeddings", embeddings, cache_index)
        // Add to vector store
        addEmbeddingsToVectorStore(vector_store, item_key, processed_data["chunks"], embeddings)

    // 6. NER Extraction (Optional, On-demand, Cache Check)
    // Could be triggered later

    return processed_data

// --- Vector Store Operations ---
Function addEmbeddingsToVectorStore(vector_store, item_key, chunks, embeddings):
    // Add embeddings with metadata (item_key, chunk_index, chunk_text)
    vector_store.add(embeddings, metadatas=[{"item_key": item_key, "chunk_idx": i, "text": chunk} for i, chunk in enumerate(chunks)])

Function searchVectorStore(query, vector_store, embedding_model, top_k=5):
    query_embedding = generateEmbeddings([query], embedding_model)[0]
    results = vector_store.search(query_embedding, k=top_k) // Returns indices/distances + metadata
    return results // List of relevant chunks with metadata

// --- RAG for Discussion/Inference ---
Function performRAG(query, vector_store, embedding_model, local_llm, top_k=5):
    // 1. Retrieve relevant chunks
    retrieved_chunks_metadata = searchVectorStore(query, vector_store, embedding_model, top_k=top_k)
    context = "\n".join([chunk_meta['text'] for chunk_meta in retrieved_chunks_metadata])
    source_keys = list(set([chunk_meta['item_key'] for chunk_meta in retrieved_chunks_metadata]))

    // 2. Build Prompt
    prompt = f"""Based on the following excerpts from research papers ({', '.join(source_keys)}):
--- Context Start ---
{context}
--- Context End ---

User Query: {query}

Answer the user's query drawing *only* from the provided context. Be concise and cite the source paper keys if possible. If the context doesn't contain the answer, state that clearly.
Answer:"""

    // 3. Generate Response using Local LLM (GPU accelerated)
    response = local_llm.generate(prompt, max_tokens=500) // Adjust parameters

    return response, source_keys

// --- Report Generation ---
Function generatePDFReport(data, template_name, output_path):
    // Use Jinja2 to render data into an HTML/Markdown template
    html_content = renderTemplate(template_name, data)
    // Use WeasyPrint or ReportLab to convert HTML/Markdown to PDF
    convertHtmlToPdf(html_content, output_path)
    print(f"Report generated: {output_path}")

// --- Main Interaction Loop (Simplified) ---
Function main():
    zotero, embedding_model, summarizer_model, qa_model, ner_model, local_llm, vector_store, cache_index = initialize()
    active_collection_items = [] // List of full paper_data dicts for the selected collection

    while True:
        displayInterface() // Show collections, active papers, chat history
        user_input = getUserInput("> ") // From GUI or CLI

        if user_input.command == "list_collections":
            collections = getCollections(zotero)
            displayCollections(collections)

        elif user_input.command == "select_collection":
            collection_id = user_input.collection_id
            print(f"Loading collection {collection_id}...")
            metadata_list = getCollectionItems(zotero, collection_id)
            active_collection_items = []
            print(f"Processing {len(metadata_list)} items (checking cache)...")
            for item_meta in metadata_list:
                // Pre-process basic info or trigger full processing later
                processed_data = processPaper(item_meta, cache_index) // At least get text and maybe embeddings
                active_collection_items.append(processed_data)
            print("Collection loaded. Embeddings indexed (if available/generated).")
            displayItemList(active_collection_items)

        elif user_input.command == "process_pdfs": // Explicitly trigger full PDF text/embedding
            target_items = selectItemsFromList(active_collection_items, user_input.item_keys_or_all)
            print(f"Processing PDFs for {len(target_items)} items...")
            for item_data in target_items:
                 // Force re-processing might be needed if only metadata was loaded initially
                 processPaper(item_data['metadata'], cache_index, force_reprocess=True)
            print("PDF processing complete.")

        elif user_input.command == "summarize":
            target_items = selectItemsFromList(active_collection_items, user_input.item_keys_or_all)
            summaries = {}
            for item_data in target_items:
                item_key = item_data['metadata']['item_key']
                cached_summary = checkCache(item_key, "summary", cache_index)
                if cached_summary:
                    summaries[item_key] = cached_summary
                elif item_data.get("text"):
                    print(f"Generating summary for {item_key}...")
                    summary = summarizeText(item_data["text"], summarizer_model)
                    summaries[item_key] = summary
                    saveToCache(item_key, "summary", summary, cache_index)
                else:
                    summaries[item_key] = "Error: No text available to summarize."
            displaySummaries(summaries)

        elif user_input.command == "extract_methods" or user_input.command == "extract_findings":
             target_items = selectItemsFromList(active_collection_items, user_input.item_keys) # Usually specific items
             extracted_info = {}
             question = "What methods were used?" if user_input.command == "extract_methods" else "What were the key findings?"
             for item_data in target_items:
                 item_key = item_data['metadata']['item_key']
                 if item_data.get("text"):
                     print(f"Extracting '{question}' from {item_key}...")
                     # Option 1: Simple section extraction (less accurate)
                     # methods_text = findSection(item_data['text'], "Methods")
                     # Option 2: QA model (potentially better)
                     answer = answerQuestion(question, item_data["text"], qa_model)
                     extracted_info[item_key] = answer
                 else:
                     extracted_info[item_key] = "Error: No text available."
             displayExtractionResults(extracted_info)

        elif user_input.command == "discuss" or user_input.command == "ask":
            query = user_input.query_text
            if not vector_store or vector_store.isEmpty():
                 print("Error: No papers processed and indexed for discussion. Please process PDFs first.")
                 continue
            print(f"Searching and generating response for: '{query}'...")
            response, sources = performRAG(query, vector_store, embedding_model, local_llm)
            displayDiscussion(query, response, sources)

        elif user_input.command == "generate_report":
            report_type = user_input.report_type // e.g., "prisma_summary", "method_comparison"
            target_items = selectItemsFromList(active_collection_items, user_input.item_keys_or_all)
            report_data = prepareReportData(report_type, target_items, cache_index) // Gathers summaries, extractions etc.
            output_filename = f"Report_{report_type}_{user_input.collection_id}.pdf"
            generatePDFReport(report_data, f"template_{report_type}.html", output_filename)

        elif user_input.command == "exit":
            break

        else:
            print("Unknown command.")

    print("Exiting Zotero Agent.")

// --- Helper Functions (Placeholders) ---
Function initializeZoteroClient(key, id): pass
Function findLocalPdfPath(item_meta, base_path): pass // Needs logic for Zotero's structure
Function loadCacheIndex(path): pass
Function saveCacheIndex(index, path): pass
Function loadDataFromCacheFile(path): pass
Function saveDataToCacheFile(data, path): pass
Function usePyMuPDF(path): pass
Function splitTextIntoChunks(text, strategy, size): pass
Function initializeVectorStore(path, dim): pass // e.g., FAISS index init
Function loadEmbeddingModel(name, device): pass // HuggingFace sentence-transformers
Function loadSummarizationModel(name, device): pass // HuggingFace pipeline
Function loadQuestionAnsweringModel(name, device): pass // HuggingFace pipeline
Function loadNerModel(name): pass // spaCy load
Function loadLocalLLM(path, gpu_layers): pass // llama-cpp-python or similar
Function renderTemplate(template_name, data): pass // Jinja2
Function convertHtmlToPdf(html, path): pass // WeasyPrint or other
Function displayInterface(): pass // GUI/CLI updates
Function getUserInput(prompt): pass // GUI/CLI interaction
Function displayCollections(collections): pass
Function displayItemList(items): pass
Function displaySummaries(summaries): pass
Function displayExtractionResults(results): pass
Function displayDiscussion(query, response, sources): pass
Function selectItemsFromList(items, selection_rule): pass // Logic for selecting items based on input
Function prepareReportData(report_type, items, cache): pass // Aggregate data for the report

// --- Entry Point ---
if __name__ == "__main__":
    main()
