"""
chroma_store.py — ChromaDB persistent storage for face embeddings.

Manages a ChromaDB collection configured with cosine similarity (HNSW index).
Provides methods to add embeddings, search for nearest matches, and reset
the collection for fresh runs.

Similarity scoring convention (used everywhere in this project):
    similarity = 1 - cosine_distance
    - 1.0 = identical
    - 0.0 = orthogonal
    - Values typically range 0.0–1.0 for face embeddings

Usage:
    store = ChromaStore(persist_dir="storage/chroma_db")
    store.add_embedding("emb_001", vector_list, {"person_id": "Person_01"})
    match_id, similarity = store.search(query_vector_list)
"""

import logging
import chromadb

logger = logging.getLogger(__name__)

COLLECTION_NAME = "face_embeddings"


class ChromaStore:
    """
    Persistent ChromaDB storage for face embeddings with cosine similarity.

    Uses PersistentClient to save data to disk. The collection is configured
    with HNSW spatial index using cosine distance metric.
    """

    def __init__(self, persist_dir):
        """
        Initialize ChromaDB with persistent storage.

        Args:
            persist_dir: Directory path for ChromaDB persistent storage.
        """
        self.persist_dir = persist_dir
        logger.info("Initializing ChromaDB at: %s", persist_dir)

        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

        logger.info(
            "ChromaDB collection '%s' ready (%d embeddings stored).",
            COLLECTION_NAME, self._collection.count()
        )

    def add_embedding(self, embedding_id, embedding_vector, metadata=None):
        """
        Store a face embedding in ChromaDB.

        Args:
            embedding_id: Unique string ID for this embedding (e.g., "emb_001").
            embedding_vector: List of floats (512-d normalized vector).
            metadata: Optional dict with additional info (person_id, frame, etc.).
        """
        if metadata is None:
            metadata = {}

        self._collection.add(
            ids=[embedding_id],
            embeddings=[embedding_vector],
            metadatas=[metadata]
        )

        logger.debug("Stored embedding '%s' in ChromaDB.", embedding_id)

    def search(self, query_embedding, n_results=1):
        """
        Search for the nearest face embedding in ChromaDB.

        Args:
            query_embedding: List of floats (512-d normalized vector).
            n_results: Number of nearest neighbors to return.

        Returns:
            List of dicts, each containing:
              - 'id': embedding ID string
              - 'similarity': float (1 - cosine_distance), higher = more similar
              - 'metadata': dict with stored metadata
            Returns empty list if collection is empty or no results found.
        """
        count = self._collection.count()
        if count == 0:
            logger.debug("ChromaDB is empty, no search results.")
            return []

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, count),
            include=["metadatas", "distances"]
        )

        matches = []
        if results and results["ids"] and results["ids"][0]:
            for i, emb_id in enumerate(results["ids"][0]):
                # ChromaDB returns cosine distance; convert to similarity
                cosine_distance = results["distances"][0][i]
                similarity = 1.0 - cosine_distance

                matches.append({
                    "id": emb_id,
                    "similarity": similarity,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {}
                })

        return matches

    def search_best(self, query_embedding):
        """
        Search for the single best match.

        Args:
            query_embedding: List of floats (512-d normalized vector).

        Returns:
            Tuple of (match_id, similarity, metadata) or (None, 0.0, {})
            if no match found or collection is empty.
        """
        matches = self.search(query_embedding, n_results=1)
        if matches:
            best = matches[0]
            return best["id"], best["similarity"], best["metadata"]
        return None, 0.0, {}

    def get_count(self):
        """Return the number of embeddings stored in the collection."""
        return self._collection.count()

    def reset(self):
        """
        Delete and recreate the collection (fresh start).

        Use this at the beginning of a new enrollment run to clear
        any previously stored embeddings.
        """
        logger.info("Resetting ChromaDB collection '%s'...", COLLECTION_NAME)
        self._client.delete_collection(COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("ChromaDB collection reset. Count: %d", self._collection.count())
