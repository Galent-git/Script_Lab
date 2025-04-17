from typing import List, Dict, Optional, Callable
try:
    from pyzotero import zotero
except ImportError:
    raise ImportError("pyzotero is required. Install with: pip install pyzotero")

def retrieve_from_zotero(api_key: str, library_id: str, library_type: str = 'user', query: Optional[str] = None, max_results: int = 10) -> List[Dict]:
    """
    Retrieve items from a Zotero library matching a query (title, author, tag, etc.).
    """
    zot = zotero.Zotero(library_id, library_type, api_key)
    if query:
        items = zot.items(q=query, limit=max_results)
    else:
        items = zot.top(limit=max_results)
    return _format_zotero_items(zot, items, max_results)

def retrieve_by_author(api_key: str, library_id: str, author: str, library_type: str = 'user', max_results: int = 10) -> List[Dict]:
    """
    Retrieve all items by a specific author from a Zotero library.
    """
    zot = zotero.Zotero(library_id, library_type, api_key)
    items = zot.items(q=author, limit=max_results)
    # Filter for author in creators
    filtered = [item for item in items if any(
        c.get('creatorType') in ['author', 'editor'] and author.lower() in (c.get('lastName','').lower() + c.get('firstName','').lower())
        for c in item.get('data', {}).get('creators', [])
    )]
    return _format_zotero_items(zot, filtered, max_results)

def retrieve_by_tag(api_key: str, library_id: str, tag: str, library_type: str = 'user', max_results: int = 10) -> List[Dict]:
    """
    Retrieve all items with a specific tag (subject) from a Zotero library.
    """
    zot = zotero.Zotero(library_id, library_type, api_key)
    items = zot.items(tag=tag, limit=max_results)
    return _format_zotero_items(zot, items, max_results)

def retrieve_by_collection(api_key: str, library_id: str, collection_key: str, library_type: str = 'user', max_results: int = 20) -> List[Dict]:
    """
    Retrieve all items in a specific collection (sublibrary) from a Zotero library.
    """
    zot = zotero.Zotero(library_id, library_type, api_key)
    items = zot.collection_items(collection_key, limit=max_results)
    return _format_zotero_items(zot, items, max_results)

def summarize_items(items: List[Dict], summarizer: Optional[Callable[[str], str]] = None, max_items: int = 50) -> str:
    """
    Aggregate abstracts/notes from a list of items and summarize using a provided summarizer (LLM or function).
    If no summarizer is provided, returns concatenated abstracts.
    """
    abstracts = [item.get('abstract') or '' for item in items[:max_items] if item.get('abstract')]
    text = '\n'.join(abstracts)
    if summarizer:
        return summarizer(text)
    return text

def _format_zotero_items(zot, items, max_results: int) -> List[Dict]:
    results = []
    for item in items:
        data = item.get('data', {})
        attachments = []
        # Get child attachments (e.g., PDFs)
        for att in zot.children(item['key']):
            if att.get('data', {}).get('itemType') == 'attachment':
                attachments.append({
                    'title': att['data'].get('title'),
                    'link': att['data'].get('url') or att['data'].get('filename')
                })
        results.append({
            'title': data.get('title'),
            'creators': data.get('creators'),
            'abstract': data.get('abstractNote'),
            'tags': [t['tag'] for t in data.get('tags', [])],
            'date': data.get('date'),
            'itemType': data.get('itemType'),
            'attachments': attachments,
            'key': item.get('key'),
        })
        if len(results) >= max_results:
            break
    return results

# Example usage (plug into any agent or LLM):
# from agent_foundations.info_retrieval_kit.zotero_retriever import (
#     retrieve_from_zotero, retrieve_by_author, retrieve_by_tag, retrieve_by_collection, summarize_items)
# results = retrieve_by_author(api_key, library_id, author='Turing')
# summary = summarize_items(results, summarizer=your_llm_function)
