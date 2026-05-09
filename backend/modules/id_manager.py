"""
id_manager.py — Temporary ID assignment and deduplication during enrollment.

During enrollment (video_1), every detected quality face needs a temporary ID
(e.g., Person_01, Person_02, ...). The same person may appear across many frames,
so before assigning a new ID we check ChromaDB for similarity to existing enrollees.

If the new face is similar enough to an already-enrolled face (above DEDUP_THRESHOLD),
we reuse the existing ID instead of creating a duplicate entry.

Usage:
    manager = IDManager(chroma_store, dedup_threshold=0.55)
    person_id, embedding_id, is_new = manager.assign_id(embedding_vector)
"""

import logging

logger = logging.getLogger(__name__)


class IDManager:
    """
    Manages temporary person ID assignment with deduplication.

    On each call to assign_id():
      1. Search ChromaDB for the most similar existing embedding.
      2. If similarity >= dedup_threshold → reuse that person's ID (not new).
      3. If similarity < dedup_threshold → assign a new Person_XX ID (new).
    """

    def __init__(self, chroma_store, dedup_threshold=0.55):
        """
        Initialize the ID manager.

        Args:
            chroma_store: ChromaStore instance for dedup search.
            dedup_threshold: Cosine similarity threshold for deduplication.
                             Faces above this threshold are considered the same person.
        """
        self._store = chroma_store
        self._dedup_threshold = dedup_threshold
        self._person_counter = 0
        self._embedding_counter = 0

    def assign_id(self, embedding_list):
        """
        Assign a temporary person ID, with deduplication against ChromaDB.

        Args:
            embedding_list: List of floats (512-d L2-normalized embedding).

        Returns:
            Tuple of (person_temp_id, embedding_id, is_new_person):
              - person_temp_id: str like "Person_01"
              - embedding_id: str like "emb_001" (unique per embedding)
              - is_new_person: True if this is a newly assigned ID
        """
        # Always generate a unique embedding ID
        self._embedding_counter += 1
        embedding_id = f"emb_{self._embedding_counter:04d}"

        # Search ChromaDB for existing similar face
        match_id, similarity, metadata = self._store.search_best(embedding_list)

        if match_id is not None and similarity >= self._dedup_threshold:
            # Existing person — reuse their ID
            person_temp_id = metadata.get("person_temp_id", "Unknown")
            logger.debug(
                "Dedup match: %s (similarity %.3f >= %.3f)",
                person_temp_id, similarity, self._dedup_threshold
            )
            return person_temp_id, embedding_id, False

        # New person — assign fresh ID
        self._person_counter += 1
        person_temp_id = f"Person_{self._person_counter:02d}"
        logger.info(
            "New person assigned: %s (best match similarity: %.3f)",
            person_temp_id, similarity
        )
        return person_temp_id, embedding_id, True

    @property
    def total_persons(self):
        """Return the total number of unique persons assigned so far."""
        return self._person_counter
