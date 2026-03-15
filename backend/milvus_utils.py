"""
Milvus utility functions for face recognition system.
Handles collection creation, face embedding storage, and similarity search.

CONFIGURED FOR CPU-ONLY MODE:
- Uses CPU-friendly index types (IVF_FLAT, FLAT, or HNSW)
- No GPU acceleration features
- Optimized for CPU performance
"""
from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility,
)
import numpy as np
from typing import List, Dict, Optional, Tuple


class MilvusFaceDB:
    """Manages face embeddings in Milvus vector database."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: str = "19530",
        collection_name: str = "face_embeddings",
        embedding_dim: int = 512,  # ArcFace produces 512-dim embeddings
        index_type: str = "IVF_FLAT",  # CPU-friendly index: IVF_FLAT, FLAT, or HNSW
    ):
        """
        Initialize Milvus connection and collection (CPU-only mode).
        
        Args:
            host: Milvus server host
            port: Milvus server port
            collection_name: Name of the collection to create/use
            embedding_dim: Dimension of face embeddings (512 for ArcFace, 128 for FaceNet)
            index_type: Index type for CPU-only mode. Options: "IVF_FLAT" (default), "FLAT", "HNSW"
                       Note: GPU index types (GPU_IVF_FLAT, GPU_IVF_PQ, etc.) are NOT supported
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        
        # Validate index type is CPU-only (no GPU indexes)
        gpu_indexes = ["GPU_IVF_FLAT", "GPU_IVF_PQ", "GPU_CAGRA", "GPU_BRUTE_FORCE"]
        if index_type in gpu_indexes:
            raise ValueError(
                f"GPU index type '{index_type}' not supported. "
                f"Use CPU-only indexes: IVF_FLAT, FLAT, or HNSW"
            )
        
        # Connect to Milvus
        connections.connect("default", host=host, port=port)
        
        # Create or load collection
        self._setup_collection()
    
    def _setup_collection(self):
        """Create collection if it doesn't exist, or load existing one (CPU-only mode)."""
        if utility.has_collection(self.collection_name):
            print(f"Loading existing collection: {self.collection_name}")
            self.collection = Collection(self.collection_name)
            # Note: Existing collections may have different index types
            # If you need to change index type, drop the collection first
        else:
            print(f"Creating new collection: {self.collection_name} (CPU-only mode)")
            self._create_collection()
        
        # Load collection into memory for faster CPU-based search
        self.collection.load()
    
    def _create_collection(self):
        """Create a new collection with proper schema."""
        # Define fields
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
            FieldSchema(name="stable_id", dtype=DataType.INT64),  # Person tracking ID
            FieldSchema(name="frame_number", dtype=DataType.INT64),
            FieldSchema(name="person_name", dtype=DataType.VARCHAR, max_length=200),  # Optional: name if known
            FieldSchema(name="bbox_x1", dtype=DataType.INT64),
            FieldSchema(name="bbox_y1", dtype=DataType.INT64),
            FieldSchema(name="bbox_x2", dtype=DataType.INT64),
            FieldSchema(name="bbox_y2", dtype=DataType.INT64),
        ]
        
        # Create schema
        schema = CollectionSchema(
            fields=fields,
            description="Face embeddings for person recognition"
        )
        
        # Create collection
        self.collection = Collection(
            name=self.collection_name,
            schema=schema,
        )
        
        # Create CPU-optimized index for fast similarity search
        # IVF_FLAT: Best balance of speed and accuracy for CPU (default)
        # FLAT: Brute force, most accurate but slower (good for small datasets)
        # HNSW: Fast approximate search, good for large datasets on CPU
        
        if self.index_type == "IVF_FLAT":
            # IVF_FLAT: Optimized for CPU with balanced performance
            # nlist: Number of clusters (1024 is good for medium datasets, adjust based on data size)
            index_params = {
                "metric_type": "COSINE",  # Use cosine similarity for face recognition
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}  # CPU-optimized: 1024-4096 for medium datasets
            }
        elif self.index_type == "FLAT":
            # FLAT: Brute force search, most accurate, CPU-friendly
            index_params = {
                "metric_type": "COSINE",
                "index_type": "FLAT",
            }
        elif self.index_type == "HNSW":
            # HNSW: Hierarchical Navigable Small World, fast approximate search on CPU
            # M: Number of bi-directional links (16-64, higher = more accurate but slower)
            # efConstruction: Size of dynamic candidate list (200-500)
            index_params = {
                "metric_type": "COSINE",
                "index_type": "HNSW",
                "params": {
                    "M": 32,  # CPU-optimized: 16-64
                    "efConstruction": 200  # CPU-optimized: 200-500
                }
            }
        else:
            raise ValueError(
                f"Unsupported index type: {self.index_type}. "
                f"Use CPU-only indexes: IVF_FLAT, FLAT, or HNSW"
            )
        
        self.collection.create_index(
            field_name="embedding",
            index_params=index_params
        )
        print(f"Collection '{self.collection_name}' created with CPU-only index: {self.index_type}")
    
    def insert_face(
        self,
        embedding: np.ndarray,
        stable_id: int,
        frame_number: int,
        bbox: Tuple[int, int, int, int],
        person_name: str = "Unknown",
    ):
        """
        Insert a single face embedding into Milvus.
        
        Args:
            embedding: Face embedding vector (numpy array)
            stable_id: Person tracking ID from ByteTrack
            frame_number: Frame number in video
            bbox: Bounding box (x1, y1, x2, y2)
            person_name: Optional name for the person
        """
        x1, y1, x2, y2 = bbox
        
        # Ensure embedding is the right shape and type
        if isinstance(embedding, list):
            embedding = np.array(embedding)
        embedding = embedding.flatten().astype(np.float32)
        
        if len(embedding) != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.embedding_dim}, got {len(embedding)}"
            )
        
        data = [{
            "embedding": embedding.tolist(),
            "stable_id": stable_id,
            "frame_number": frame_number,
            "person_name": person_name,
            "bbox_x1": int(x1),
            "bbox_y1": int(y1),
            "bbox_x2": int(x2),
            "bbox_y2": int(y2),
        }]
        
        self.collection.insert(data)
    
    def search_face(
        self,
        embedding: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.5,
    ) -> List[Dict]:
        """
        Search for similar faces in the database.
        
        Args:
            embedding: Face embedding vector to search for
            top_k: Number of top results to return
            threshold: Minimum similarity threshold (0-1, higher = more strict)
        
        Returns:
            List of matching results with metadata
        """
        # Ensure embedding is the right shape and type
        if isinstance(embedding, list):
            embedding = np.array(embedding)
        embedding = embedding.flatten().astype(np.float32)
        
        if len(embedding) != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.embedding_dim}, got {len(embedding)}"
            )
        
        # Prepare CPU-optimized search parameters
        # Note: For existing collections, Milvus will use the existing index type
        # These params are optimized for CPU-only operation
        
        if self.index_type == "IVF_FLAT":
            # nprobe: Number of clusters to search (for IVF_FLAT)
            # CPU-optimized: 10-64 is a good balance (16 is default)
            # Lower = faster but less accurate, Higher = slower but more accurate
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 16}  # CPU-optimized: search 16 clusters
            }
        elif self.index_type == "HNSW":
            # ef: Size of dynamic candidate list for HNSW
            # CPU-optimized: 32-128 (64 is a good balance)
            # Higher = more accurate but slower
            search_params = {
                "metric_type": "COSINE",
                "params": {"ef": 64}  # CPU-optimized
            }
        else:  # FLAT or other CPU-friendly indexes
            # FLAT uses brute force search, no special params needed
            # Works well on CPU for small to medium datasets
            search_params = {
                "metric_type": "COSINE",
            }
        
        # Perform search
        results = self.collection.search(
            data=[embedding.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["stable_id", "frame_number", "person_name", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"],
        )
        
        # Process results
        matches = []
        for hits in results:
            for hit in hits:
                # Cosine similarity: higher score = more similar
                # Convert distance to similarity (1 - distance)
                similarity = 1 - hit.distance if hasattr(hit, 'distance') else hit.score
                
                if similarity >= threshold:
                    matches.append({
                        "similarity": float(similarity),
                        "stable_id": hit.entity.get("stable_id"),
                        "frame_number": hit.entity.get("frame_number"),
                        "person_name": hit.entity.get("person_name", "Unknown"),
                        "bbox": (
                            hit.entity.get("bbox_x1"),
                            hit.entity.get("bbox_y1"),
                            hit.entity.get("bbox_x2"),
                            hit.entity.get("bbox_y2"),
                        ),
                    })
        
        return matches
    
    def get_person_embeddings(self, stable_id: int) -> List[Dict]:
        """
        Get all embeddings for a specific person (stable_id).
        
        Args:
            stable_id: Person tracking ID
        
        Returns:
            List of embeddings with metadata
        """
        # Query by stable_id
        results = self.collection.query(
            expr=f'stable_id == {stable_id}',
            output_fields=["embedding", "frame_number", "person_name", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"],
        )
        return results
    
    def flush(self):
        """Flush data to disk."""
        self.collection.flush()
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection."""
        stats = {
            "num_entities": self.collection.num_entities,
            "collection_name": self.collection_name,
        }
        return stats

