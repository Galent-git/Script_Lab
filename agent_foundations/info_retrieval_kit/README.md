# Information Retrieval Kit

This directory contains pluggable modules for information retrieval from various sources (local folders, Zotero, Google Drive, etc.) for use in agent-based systems.

## Structure

```
agent_foundations/
  info_retrieval_kit/
    local_pdf_retriever.py   # Module for local PDF search
    README.md                # This documentation
    zotero_retriever.py        # Module for Zotero library search
    # (future) gdrive_client.py
    # (future) retrieval_interface.py
```

---

## Module: `local_pdf_retriever.py`

### Purpose
- Recursively scan a folder for PDFs and extract text
- Search PDFs for a query, return relevant matches/snippets
- Aggregate and summarize PDF content using any LLM or summarizer function
- Extract context snippets for LLM prompt construction
- All functions are agent/LLM-pluggable and can be imported directly or used in your core module

### Requirements
- `PyPDF2` (install with `pip install PyPDF2`)

### Functions
- `retrieve_pdfs_from_folder(folder_path, query=None, max_results=10)`: Scan for PDFs, extract text, and search for a query
- `summarize_pdfs(pdf_results, summarizer=None, max_items=10)`: Aggregate and summarize PDF text using any LLM or function (optional)
- `get_context_from_pdfs(pdf_results, max_snippets=5, snippet_length=300)`: Aggregate relevant context snippets for LLM input

### Usage Examples
```python
from agent_foundations.info_retrieval_kit.local_pdf_retriever import (
    retrieve_pdfs_from_folder, summarize_pdfs, get_context_from_pdfs
)

# Retrieve PDFs and search for a query
results = retrieve_pdfs_from_folder('/path/to/folder', query='artificial intelligence')

# Summarize PDF results using your LLM (or just aggregate text)
def my_llm_summarizer(text):
    # Replace with your LLM call
    return text[:500] + '...'

summary = summarize_pdfs(results, summarizer=my_llm_summarizer)

# Get context snippets for LLM prompt
context = get_context_from_pdfs(results)
```
- All functions return lists of dicts or strings for use in LLM/agent pipelines.
- Use any summarizer function or LLM for aggregation (optional).

---

## Module: `zotero_retriever.py`

### Purpose
- Search and retrieve items from a Zotero library (user or group)
- Retrieve by keyword, author, tag (subject), or collection (sublibrary)
- Aggregate and summarize results (e.g., abstracts/notes) using any LLM or function
- All functions are agent/LLM-pluggable and can be imported directly or used in your core module

### Requirements
- `pyzotero` (install with `pip install pyzotero`)

### Functions
- `retrieve_from_zotero(api_key, library_id, query=None, library_type='user', max_results=10)`: General keyword search
- `retrieve_by_author(api_key, library_id, author, library_type='user', max_results=10)`: Retrieve all items by a specific author
- `retrieve_by_tag(api_key, library_id, tag, library_type='user', max_results=10)`: Retrieve all items with a specific tag/subject
- `retrieve_by_collection(api_key, library_id, collection_key, library_type='user', max_results=20)`: Retrieve all items in a specific collection (sublibrary)
- `summarize_items(items, summarizer=None, max_items=50)`: Aggregate and summarize abstracts/notes from a list of items, using any LLM or summarizer function (if none provided, returns concatenated abstracts)

### Usage Examples
```python
from agent_foundations.info_retrieval_kit.zotero_retriever import (
    retrieve_from_zotero, retrieve_by_author, retrieve_by_tag, retrieve_by_collection, summarize_items
)

# General keyword search
results = retrieve_from_zotero(api_key, library_id, query='machine learning')

# Retrieve by author
author_items = retrieve_by_author(api_key, library_id, author='Turing')

# Retrieve by tag/subject
tagged_items = retrieve_by_tag(api_key, library_id, tag='AI')

# Retrieve by collection (sublibrary)
collection_items = retrieve_by_collection(api_key, library_id, collection_key='YOUR_COLLECTION_KEY')

# Summarize a set of items using your LLM (or just aggregate abstracts)
def my_llm_summarizer(text):
    # Replace with your LLM call
    return text[:500] + '...'

summary = summarize_items(author_items, summarizer=my_llm_summarizer)
```
- All functions return lists of dicts with metadata, tags, and attachment links (e.g., PDFs).
- Use any summarizer function or LLM for aggregation (optional).

---

## Module: `local_pdf_retriever.py`

### Purpose
- Recursively scans a local folder for PDF files
- Extracts text from each PDF
- Optionally searches for a query string in the text
- Returns relevant matches (filename, path, snippet)
- Designed to be pluggable into any agent with a single import

### Requirements
- `PyPDF2` (install with `pip install PyPDF2`)

### Usage

```python
from agent_foundations.info_retrieval_kit.local_pdf_retriever import retrieve_pdfs_from_folder

results = retrieve_pdfs_from_folder('/path/to/folder', query='artificial intelligence')

for result in results:
    print('File:', result['filename'])
    print('Path:', result['path'])
    if 'matches' in result:
        print('Matches:', result['matches'])
    else:
        print('Text snippet:', result['text'])
```
- If `query` is provided, returns only PDFs containing the query, with matching lines.
- If no `query` is provided, returns a snippet of text from each PDF.
- `max_results` limits the number of results (default: 10).

---

## Extending
- Additional modules (Zotero, Google Drive, etc.) can be added to this directory following the same pluggable pattern.

---

For questions or contributions, open an issue or pull request in the repository.
