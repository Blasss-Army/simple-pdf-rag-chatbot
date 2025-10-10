
from dataclasses import dataclass
from pathlib import Path
from qdrant_client.models import Distance

# conf.py is located at .../create_retriever/conf.py
# parents[0] = .../create_retriever
# parents[1] = .../  (project root)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class RetrieverConfig:
    #-----------PATH CONFIG---------------
    data_path: Path = PROJECT_ROOT/"data"                  # path to the data (documents)
    index_path: Path = PROJECT_ROOT/"index"                # path to Qdrant's local storage
    #-----------COLLECTION CONFIG---------------
    collection_name: str = "my_collection"                 # name of the collection in Qdrant
    distance: Distance = Distance.COSINE                   # similarity metric in Qdrant
    prefer_grpc: bool = True                               # use gRPC (faster/more efficient)
    recreate: bool = False                                 # if True, delete and recreate the collection
    temperature: float = 0.2                               # temperature for text generation (LLM)
    reset_collection: bool = True                         # if True, delete the collection on startup
    #-----------EMBEDDING MODEL CONFIG---------------
    embed_model: str = "models/text-embedding-004"         # embeddings model (Google)
    #-----------VECTOR STORE CONFIG----------------
    vectore_store_search_type: str = "mmr"                 # Search type for vectore store (similarity, mmr)
    vectore_store_k: int = 5                               # k= how many chunks will be retrieved
    vectore_store_fetch_k: int = 30                        # When using mmr, how many chunk should be compared before retrival
    vectore_store_lambda_mult: int = 0.5                   # Lambda value
