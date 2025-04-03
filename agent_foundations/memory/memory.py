# --- Imports ---
import json
import os
import logging # Use logging instead of print
from datetime import datetime
from collections import deque
from typing import List, Dict, Optional, Any, Tuple # For type hinting

try:
    import chromadb
    from chromadb.api.models.Collection import Collection # Specific type hint
    from sentence_transformers import SentenceTransformer
    IS_LTM_AVAILABLE = True
except ImportError:
    # Allow the module to load even if LTM dependencies are missing,
    # but disable LTM functionality.
    IS_LTM_AVAILABLE = False
    # Define dummy types for hinting if imports fail
    class SentenceTransformer: pass
    class Collection: pass
    print("WARNING: LTM dependencies (chromadb, sentence-transformers) not found. LTM functionality disabled.")


# --- Configuration ---
DEFAULT_STM_MAX_TURNS = 10
DEFAULT_LTM_COLLECTION_NAME = "agent_long_term_memory"
DEFAULT_EMBEDDING_MODEL = 'all-MiniLM-L6-v2' # Ensure this model is installed or available
DEFAULT_CHROMA_PATH = "memory_vector_db"

# --- Setup Logging ---
# Configure logging basic settings (can be overridden by the application using this module)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s')
logger = logging.getLogger(__name__) # Get a logger specific to this module

class MemoryManager:
    """
    Manages different levels of memory for an AI agent:
    - Working Memory: Transient data for current task context.
    - Short-Term Memory (STM): Recent conversation history (optionally persistent).
    - Long-Term Memory (LTM): Persistent, semantically searchable knowledge using vector embeddings.
    """

    def __init__(self,
                 stm_max_turns: int = DEFAULT_STM_MAX_TURNS,
                 ltm_collection_name: str = DEFAULT_LTM_COLLECTION_NAME,
                 embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
                 vector_db_path: str = DEFAULT_CHROMA_PATH,
                 stm_persistence_path: Optional[str] = None,
                 log_level: int = logging.INFO): # Allow setting log level

        logger.setLevel(log_level)
        logger.info("Initializing MemoryManager...")

        # --- Working Memory ---
        self.working_memory: Dict[str, Any] = {}
        logger.debug("Working Memory initialized (transient).")

        # --- Short-Term Memory ---
        self.stm_max_turns: int = stm_max_turns
        self.short_term_memory: deque = deque(maxlen=self.stm_max_turns)
        self.stm_persistence_path: Optional[str] = stm_persistence_path
        self._load_stm() # Load STM if persistence path is provided
        logger.info(f"Short-Term Memory initialized (max_turns={self.stm_max_turns}, persistent={bool(self.stm_persistence_path)}).")

        # --- Long-Term Memory (Vector Store) ---
        self.embedding_model: Optional[SentenceTransformer] = None
        self._chroma_client: Optional[chromadb.Client] = None
        self._ltm_collection: Optional[Collection] = None # Use specific Collection type
        self.ltm_collection_name: str = ltm_collection_name
        self.vector_db_path: str = vector_db_path
        self.embedding_model_name: str = embedding_model_name

        if IS_LTM_AVAILABLE:
            try:
                logger.info(f"Loading embedding model: {self.embedding_model_name}...")
                # Consider adding device='cuda' or device='mps' if GPU available
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
                logger.info("Embedding model loaded.")

                logger.info(f"Initializing Vector DB Client (path: {self.vector_db_path})...")
                self._chroma_client = chromadb.PersistentClient(path=self.vector_db_path)

                logger.info(f"Getting/Creating LTM Collection: {self.ltm_collection_name}...")
                self._ltm_collection = self._chroma_client.get_or_create_collection(
                    name=self.ltm_collection_name,
                    # Optional: Specify embedding function if not using default MiniLM with Chroma
                    # embedding_function=chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(model_name=self.embedding_model_name)
                )
                logger.info(f"Long-Term Memory initialized (Collection: {self.ltm_collection_name}).")
                logger.info(f"Initial LTM Document Count: {self._ltm_collection.count()}")

            except Exception as e:
                logger.exception(f"Failed to initialize LTM. LTM functionality will be disabled. Error: {e}")
                self.embedding_model = None
                self._chroma_client = None
                self._ltm_collection = None
        else:
            logger.warning("LTM dependencies not found. Skipping LTM initialization.")

    # --- Working Memory Methods ---
    def set_working(self, key: str, value: Any) -> None:
        """Stores or updates a key-value pair in working memory."""
        self.working_memory[key] = value
        logger.debug(f"Set working memory key '{key}'.")

    def get_working(self, key: str, default: Optional[Any] = None) -> Optional[Any]:
        """Retrieves a value from working memory."""
        return self.working_memory.get(key, default)

    def get_all_working(self) -> Dict[str, Any]:
        """Returns a copy of the entire working memory dictionary."""
        return self.working_memory.copy()

    def clear_working(self, key: Optional[str] = None) -> None:
        """Clears a specific key or the entire working memory."""
        if key:
            if key in self.working_memory:
                del self.working_memory[key]
                logger.debug(f"Cleared working memory key '{key}'.")
            else:
                logger.warning(f"Attempted to clear non-existent working memory key '{key}'.")
        else:
            self.working_memory.clear()
            logger.info("Cleared all working memory.")

    # --- Short-Term Memory Methods ---
    def add_interaction(self, role: str, content: str, timestamp: Optional[str] = None) -> None:
        """Adds a single interaction turn (e.g., user query, agent response) to STM."""
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        interaction = {
            "role": role, # e.g., 'user', 'assistant', 'system'
            "content": content,
            "timestamp": timestamp
        }
        self.short_term_memory.append(interaction)
        logger.debug(f"Added interaction ({role}) to STM. Current size: {len(self.short_term_memory)}")
        self._save_stm() # Save STM if persistence is enabled

    def get_short_term_history(self, n: Optional[int] = None) -> List[Dict[str, str]]:
        """Returns the last n interactions from STM. If n is None or >= current size, returns all."""
        if n is None or n >= len(self.short_term_memory):
            return list(self.short_term_memory)
        else:
            # Deque doesn't support direct slicing like list[-n:], so convert
            return list(self.short_term_memory)[-n:]

    def clear_short_term_history(self) -> None:
        """Clears all interactions from STM."""
        self.short_term_memory.clear()
        logger.info("Cleared Short-Term Memory.")
        self._save_stm() # Save cleared STM if persistence is enabled

    def _save_stm(self) -> None:
        """Saves the STM deque to a JSON file if path is configured."""
        if self.stm_persistence_path:
            try:
                os.makedirs(os.path.dirname(self.stm_persistence_path), exist_ok=True)
                with open(self.stm_persistence_path, "w", encoding="utf-8") as f:
                    json.dump(list(self.short_term_memory), f, indent=2)
                logger.debug(f"Saved STM to {self.stm_persistence_path}")
            except IOError as e:
                logger.error(f"Failed to save STM to {self.stm_persistence_path}: {e}")
            except Exception as e:
                logger.exception(f"An unexpected error occurred while saving STM: {e}")


    def _load_stm(self) -> None:
        """Loads the STM deque from a JSON file if path is configured."""
        if self.stm_persistence_path and os.path.exists(self.stm_persistence_path):
            try:
                with open(self.stm_persistence_path, "r", encoding="utf-8") as f:
                    history_list = json.load(f)
                    # Initialize deque with loaded data, respecting maxlen
                    self.short_term_memory = deque(history_list, maxlen=self.stm_max_turns)
                    logger.info(f"Loaded {len(self.short_term_memory)} items into STM from {self.stm_persistence_path}")
            except json.JSONDecodeError as e:
                 logger.error(f"Failed to decode JSON from STM file {self.stm_persistence_path}: {e}")
                 self.short_term_memory = deque(maxlen=self.stm_max_turns) # Initialize empty
            except IOError as e:
                 logger.error(f"Failed to read STM file {self.stm_persistence_path}: {e}")
                 self.short_term_memory = deque(maxlen=self.stm_max_turns) # Initialize empty
            except Exception as e:
                 logger.exception(f"An unexpected error occurred while loading STM: {e}")
                 self.short_term_memory = deque(maxlen=self.stm_max_turns) # Initialize empty
        else:
             # Initialize empty if no persistence path or file doesn't exist
             self.short_term_memory = deque(maxlen=self.stm_max_turns)
             if self.stm_persistence_path:
                 logger.debug(f"STM persistence file not found at {self.stm_persistence_path}. Initializing empty STM.")


    # --- Long-Term Memory Methods ---

    def _is_ltm_ready(self) -> bool:
        """Checks if LTM components are initialized."""
        if not IS_LTM_AVAILABLE:
             logger.warning("Attempted LTM operation, but dependencies are missing.")
             return False
        if not self._ltm_collection or not self.embedding_model:
            logger.warning("Attempted LTM operation, but LTM is not properly initialized.")
            return False
        return True

    def _generate_embeddings(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Generates embeddings for a list of texts."""
        if not self._is_ltm_ready():
            return None
        try:
            # Batch encoding is efficient
            embeddings = self.embedding_model.encode(texts, convert_to_numpy=False)
             # Convert directly to list of lists
            return [emb.tolist() for emb in embeddings]
        except Exception as e:
            logger.exception(f"Failed to generate embeddings: {e}")
            return None

    def add_to_long_term(self, text_content: str, metadata: Optional[Dict[str, Any]] = None, doc_id: Optional[str] = None) -> bool:
        """Adds a single document to Long-Term Memory."""
        return self.add_many_to_long_term(
            texts=[text_content],
            metadatas=[metadata] if metadata else None,
            ids=[doc_id] if doc_id else None
        )

    def add_many_to_long_term(self, texts: List[str], metadatas: Optional[List[Optional[Dict[str, Any]]]] = None, ids: Optional[List[str]] = None) -> bool:
        """
        Adds multiple documents to the Long-Term Memory using batch processing.

        Args:
            texts (List[str]): The list of texts to store.
            metadatas (Optional[List[Optional[Dict[str, Any]]]]): List of metadata dicts, one per text.
                                                                  Must be ChromaDB compatible (str, int, float, bool).
                                                                  If None, empty metadata is used for all.
            ids (Optional[List[str]]): List of unique IDs, one per text. If None, IDs are generated.

        Returns:
            bool: True if adding was successful (or partially successful), False otherwise.
        """
        if not self._is_ltm_ready(): return False
        if not texts:
            logger.warning("Attempted to add empty list of texts to LTM.")
            return False

        batch_size = len(texts)
        logger.debug(f"Attempting to add {batch_size} document(s) to LTM.")

        # Validate metadata list length if provided
        if metadatas and len(metadatas) != batch_size:
            logger.error(f"Metadata list length ({len(metadatas)}) does not match text list length ({batch_size}).")
            return False
        # Validate ids list length if provided
        if ids and len(ids) != batch_size:
            logger.error(f"ID list length ({len(ids)}) does not match text list length ({batch_size}).")
            return False

        # Generate embeddings in batch
        embeddings = self._generate_embeddings(texts)
        if embeddings is None:
            return False # Error already logged in _generate_embeddings

        # Prepare final lists for ChromaDB
        final_metadatas = []
        final_ids = []
        timestamp_now = datetime.now().isoformat()

        for i in range(batch_size):
            # Prepare metadata
            current_meta = metadatas[i].copy() if metadatas and metadatas[i] is not None else {}
            current_meta['timestamp'] = timestamp_now # Add timestamp automatically
            # Consider adding original text to metadata if needed, though Chroma stores it in 'documents'
            # current_meta['original_text'] = texts[i]
            final_metadatas.append(current_meta)

            # Prepare ID
            current_id = ids[i] if ids else f"doc_{hash(texts[i])}_{timestamp_now}_{i}"
            final_ids.append(current_id)

        try:
            self._ltm_collection.add(
                ids=final_ids,
                embeddings=embeddings,
                metadatas=final_metadatas,
                documents=texts # Store the original text as the document
            )
            logger.info(f"Successfully added {batch_size} document(s) to LTM. New count: {self._ltm_collection.count()}")
            return True

        except Exception as e: # Catch Chroma specific errors if known, else generic
            logger.exception(f"Failed to add batch to LTM: {e}")
            return False


    def search_long_term(self, query_text: str, top_k: int = 5, filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Searches the LTM for text semantically similar to the query_text.

        Args:
            query_text (str): The text to search for.
            top_k (int): The maximum number of results to return.
            filter_metadata (Optional[Dict[str, Any]]): ChromaDB 'where' filter dictionary.

        Returns:
            List[Dict[str, Any]]: A list of search results, including id, document, distance, metadata.
                                   Returns empty list if LTM unavailable, error occurs, or no results found.
        """
        if not self._is_ltm_ready(): return []
        if not query_text:
             logger.warning("LTM search attempted with empty query text.")
             return []

        logger.debug(f"Searching LTM (top_k={top_k}) for query: '{query_text[:50]}...'")

        # Generate embedding for the query
        query_embedding_list = self._generate_embeddings([query_text])
        if query_embedding_list is None or not query_embedding_list:
            return [] # Error logged previously

        query_embedding = query_embedding_list[0]

        try:
            results = self._ltm_collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, self._ltm_collection.count()), # Avoid asking for more results than exist
                where=filter_metadata,
                include=['metadatas', 'documents', 'distances']
            )
            logger.debug(f"LTM raw search results: {results}")

            # Process results into a more usable format
            processed_results = []
            # Chroma returns lists for each included field, nested in another list for batch queries
            if results and results.get('ids') and results['ids'][0]:
                ids = results['ids'][0]
                distances = results['distances'][0]
                metadatas = results['metadatas'][0]
                documents = results['documents'][0]

                for i in range(len(ids)):
                     processed_results.append({
                        "id": ids[i],
                        "document": documents[i],
                        "distance": distances[i], # Lower distance = more similar
                        "metadata": metadatas[i]
                     })
                logger.debug(f"Processed {len(processed_results)} LTM search results.")
                return processed_results
            else:
                logger.debug("LTM search returned no results.")
                return []

        except Exception as e:
            logger.exception(f"Failed to search LTM: {e}")
            return []

    def delete_from_long_term(self, ids: Optional[List[str]] = None, where_filter: Optional[Dict[str, Any]] = None) -> bool:
        """
        Deletes items from LTM based on IDs or a metadata filter.

        Args:
            ids (Optional[List[str]]): A list of document IDs to delete.
            where_filter (Optional[Dict[str, Any]]): A ChromaDB 'where' filter.
                                                     Deletes all documents matching the filter.

        Returns:
            bool: True if deletion was attempted (ChromaDB doesn't easily confirm success count), False if LTM unavailable or invalid args.
        Note: You must provide *either* ids *or* where_filter, not both or neither.
        """
        if not self._is_ltm_ready(): return False

        if not ids and not where_filter:
            logger.error("Deletion from LTM requires either 'ids' or 'where_filter' to be provided.")
            return False
        if ids and where_filter:
             logger.error("Provide either 'ids' or 'where_filter' for LTM deletion, not both.")
             return False

        try:
            if ids:
                logger.info(f"Attempting to delete {len(ids)} documents from LTM by ID.")
                self._ltm_collection.delete(ids=ids)
            elif where_filter:
                 logger.info(f"Attempting to delete documents from LTM matching filter: {where_filter}")
                 self._ltm_collection.delete(where=where_filter)

            logger.info(f"Deletion request sent to LTM. New count: {self._ltm_collection.count()}") # Count might take a moment to update
            return True
        except Exception as e:
            logger.exception(f"Failed to delete from LTM: {e}")
            return False

    def clear_long_term_memory(self, are_you_sure: bool = False) -> bool:
        """
        Deletes the entire LTM collection. This is irreversible!
        Requires confirmation via the are_you_sure flag.
        """
        if not self._is_ltm_ready(): return False
        if not self._chroma_client: # Need client to delete collection
             logger.error("Chroma client not available for LTM clear operation.")
             return False

        if are_you_sure:
            try:
                collection_name = self._ltm_collection.name
                logger.warning(f"DELETING ENTIRE LTM Collection: {collection_name}...")
                self._chroma_client.delete_collection(name=collection_name)
                logger.info("LTM Collection Deleted.")
                # Recreate an empty one to allow continued use
                self._ltm_collection = self._chroma_client.get_or_create_collection(
                     name=collection_name,
                )
                logger.info(f"Empty LTM Collection '{collection_name}' recreated.")
                return True
            except Exception as e:
                logger.exception(f"Failed to delete LTM collection: {e}")
                # State might be inconsistent here, maybe try recreating anyway?
                try:
                   self._ltm_collection = self._chroma_client.get_or_create_collection(
                       name=self.ltm_collection_name, # Use original name
                   )
                   logger.warning("Recreated LTM collection after deletion error, but state may be unexpected.")
                except Exception as inner_e:
                    logger.error(f"Failed to recreate LTM collection after deletion error: {inner_e}")
                    self._ltm_collection = None # Mark as unavailable
                return False
        else:
            logger.warning("LTM clear aborted. Set 'are_you_sure=True' to proceed.")
            return False

    # --- Context Aggregation Method ---
    def get_context_for_prompt(self, current_query: Optional[str] = None, stm_turns: int = 5, ltm_results: int = 3) -> Dict[str, Any]:
        """
        Aggregates context from different memory types.

        Args:
            current_query (Optional[str]): The current user query or topic for LTM search.
                                          If None, LTM search is skipped.
            stm_turns (int): Number of recent STM turns to include.
            ltm_results (int): Max number of relevant LTM results to include if query provided.

        Returns:
            Dict[str, Any]: Contains 'working_memory', 'short_term_history', 'long_term_relevant'.
        """
        context: Dict[str, Any] = {}

        # 1. Working Memory
        context['working_memory'] = self.get_all_working()

        # 2. Short-Term Memory
        context['short_term_history'] = self.get_short_term_history(n=stm_turns)

        # 3. Long-Term Memory
        if self._is_ltm_ready() and current_query and ltm_results > 0:
            context['long_term_relevant'] = self.search_long_term(current_query, top_k=ltm_results)
        else:
            context['long_term_relevant'] = []
            if not current_query and ltm_results > 0:
                 logger.debug("No current_query provided, skipping LTM search for context.")

        return context

    # --- Helper to format context for LLM ---
    def format_context_for_llm(self, context_dict: Dict[str, Any],
                               include_working: bool = True,
                               include_stm: bool = True,
                               include_ltm: bool = True) -> str:
        """Formats the context dictionary into a string suitable for an LLM prompt."""
        prompt_parts = []

        # Working Memory
        if include_working and context_dict.get('working_memory'):
            wm_items = [f"- {k}: {v}" for k, v in context_dict['working_memory'].items()]
            if wm_items:
                prompt_parts.append("--- Current Context/Working Memory ---")
                prompt_parts.extend(wm_items)

        # Long-Term Memory
        if include_ltm and context_dict.get('long_term_relevant'):
             # Sort by distance (closer first) before formatting
             sorted_ltm = sorted(context_dict['long_term_relevant'], key=lambda x: x['distance'])
             ltm_items = [f"- {res['document']} (Similarity Score: {1 - res['distance']:.2f})" for res in sorted_ltm]
             # Note: Using 1-distance for cosine similarity representation (closer to 1 is better)
             if ltm_items:
                prompt_parts.append("\n--- Relevant Information Retrieved From Long-Term Memory ---")
                prompt_parts.extend(ltm_items)

        # Short-Term Memory (usually comes last as most recent)
        if include_stm and context_dict.get('short_term_history'):
             stm_items = [f"{turn['role'].capitalize()}: {turn['content']}" for turn in context_dict['short_term_history']]
             if stm_items:
                 prompt_parts.append("\n--- Recent Conversation History ---")
                 prompt_parts.extend(stm_items)

        return "\n".join(prompt_parts)


# --- Example Usage ---
if __name__ == "__main__":
    # Configure logging for the example
    # logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s') # Show DEBUG messages
    logger.info("--- Initializing MemoryManager (Example) ---")

    # Example: Use a different DB path and collection name
    example_db_path = "example_agent_ltm_db"
    example_stm_path = "example_agent_stm.json"
    example_collection = "my_cool_agent_memory"

    memory = MemoryManager(
        vector_db_path=example_db_path,
        stm_persistence_path=example_stm_path,
        ltm_collection_name=example_collection,
        log_level=logging.DEBUG # Set log level for this instance
    )
    logger.info("\n--- MemoryManager Initialized ---")

    # Check if LTM is actually working
    if not memory._is_ltm_ready():
         logger.error("LTM failed to initialize. Exiting example.")
         exit()

    # Clear previous run data (optional, careful!)
    memory.clear_working()
    memory.clear_short_term_history()
    # memory.clear_long_term_memory(are_you_sure=True) # Uncomment to clear LTM too

    # === Working Memory ===
    logger.info("\n--- Working Memory ---")
    memory.set_working("current_task", "Demonstrate memory module.")
    memory.set_working("user_preference", {"language": "python", "topic": "AI"})
    logger.info(f"Working Memory: {memory.get_all_working()}")

    # === Short-Term Memory ===
    logger.info("\n--- Short-Term Memory ---")
    memory.add_interaction("user", "Hello there!")
    memory.add_interaction("assistant", "Hi! How can I help you today?")
    memory.add_interaction("user", "Let's test the memory system.")
    logger.info(f"Last 2 STM entries: {memory.get_short_term_history(n=2)}")

    # === Long-Term Memory (Batch Add) ===
    logger.info("\n--- Long-Term Memory (Batch Add) ---")
    facts_to_add = [
        "The MemoryManager uses ChromaDB for vector storage.",
        "Sentence Transformers generate text embeddings.",
        "Logging is preferred over printing for libraries.",
        "Type hints improve code readability and safety.",
        "Batch operations can improve LTM insertion speed.",
    ]
    metadata_list = [
        {"source": "module_feature", "type": "database"},
        {"source": "module_feature", "type": "embedding"},
        {"source": "best_practice", "area": "output"},
        {"source": "best_practice", "area": "typing"},
        {"source": "optimization", "operation": "add"},
    ]
    ids_list = [f"fact_{i+1}" for i in range(len(facts_to_add))]

    memory.add_many_to_long_term(facts_to_add, metadata_list, ids_list)

    # === Long-Term Memory (Search) ===
    logger.info("\n--- Long-Term Memory (Search) ---")
    query = "How is vector data stored?"
    logger.info(f"Searching LTM for: '{query}'")
    search_results = memory.search_long_term(query, top_k=2)
    logger.info(f"Search Results:\n{json.dumps(search_results, indent=2)}")

    # === Long-Term Memory (Deletion) ===
    logger.info("\n--- Long-Term Memory (Deletion) ---")
    # Delete by ID
    memory.delete_from_long_term(ids=["fact_3"]) # Delete the logging fact
    # Delete by filter
    memory.delete_from_long_term(where_filter={"type": "embedding"}) # Delete the sentence transformer fact
    logger.info(f"LTM count after deletions: {memory._ltm_collection.count()}")

    # === Context Aggregation ===
    logger.info("\n--- Context Aggregation ---")
    current_user_query = "Tell me about code quality practices shown here."
    context_data = memory.get_context_for_prompt(
        current_query=current_user_query,
        stm_turns=2, # Get last 2 interactions
        ltm_results=3  # Get top 3 relevant facts (even after deletion)
    )

    logger.info("\nFormatted Context for LLM:")
    formatted_context = memory.format_context_for_llm(context_data)
    print(formatted_context) # Use print here to see the final formatted output clearly

    # Optional: Clean up persistent files after example
    # import shutil
    # if os.path.exists(example_db_path):
    #      logger.info(f"Removing example LTM DB at {example_db_path}")
    #      shutil.rmtree(example_db_path)
    # if os.path.exists(example_stm_path):
    #      logger.info(f"Removing example STM file at {example_stm_path}")
    #      os.remove(example_stm_path)
