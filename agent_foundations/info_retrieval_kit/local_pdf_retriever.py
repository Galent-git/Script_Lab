import os
from typing import List, Dict, Optional, Callable
import PyPDF2

def retrieve_pdfs_from_folder(folder_path: str, query: Optional[str] = None, max_results: int = 10) -> List[Dict]:
    """
    Recursively scan a folder for PDFs, extract text, and optionally search for a query.
    Returns a list of dictionaries with filename, path, and matching snippets (if query is provided).
    """
    results = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_path = os.path.join(root, file)
                try:
                    with open(pdf_path, 'rb') as f:
                        reader = PyPDF2.PdfReader(f)
                        text = "\n".join(page.extract_text() or '' for page in reader.pages)
                except Exception:
                    continue  # Skip unreadable PDFs
                if query:
                    matches = []
                    for line in text.splitlines():
                        if query.lower() in line.lower():
                            matches.append(line.strip())
                    if matches:
                        results.append({
                            'filename': file,
                            'path': pdf_path,
                            'matches': matches[:max_results],
                            'full_text': text
                        })
                else:
                    results.append({
                        'filename': file,
                        'path': pdf_path,
                        'text': text[:1000],  # Return a snippet only
                        'full_text': text
                    })
                if len(results) >= max_results:
                    return results
    return results

def summarize_pdfs(pdf_results: List[Dict], summarizer: Optional[Callable[[str], str]] = None, max_items: int = 10) -> str:
    """
    Aggregate text from a list of PDF results and summarize using a provided summarizer (LLM or function).
    If no summarizer is provided, returns concatenated text snippets.
    """
    texts = [item.get('full_text', '') for item in pdf_results[:max_items] if item.get('full_text')]
    text = '\n'.join(texts)
    if summarizer:
        return summarizer(text)
    return text[:3000]  # Return up to 3000 chars if no summarizer

def get_context_from_pdfs(pdf_results: List[Dict], max_snippets: int = 5, snippet_length: int = 300) -> str:
    """
    Aggregate relevant context snippets from PDF results for LLM input or prompt construction.
    Returns a string of concatenated snippets with source filenames.
    """
    context = []
    for item in pdf_results[:max_snippets]:
        snippet = ''
        if 'matches' in item and item['matches']:
            snippet = '... '.join(item['matches'][:2])
        elif 'text' in item:
            snippet = item['text'][:snippet_length]
        context.append(f"From {item['filename']}:\n{snippet}\n")
    return '\n'.join(context)

# Example usage (to plug into any agent or LLM):
# from agent_foundations.info_retrieval_kit.local_pdf_retriever import (
#     retrieve_pdfs_from_folder, summarize_pdfs, get_context_from_pdfs)
# results = retrieve_pdfs_from_folder('/path/to/folder', query='artificial intelligence')
# summary = summarize_pdfs(results, summarizer=your_llm_function)
# context = get_context_from_pdfs(results)
