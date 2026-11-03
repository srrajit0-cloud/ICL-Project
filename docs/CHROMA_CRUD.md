# ChromaDB CRUD operations

This guide demonstrates Create, Read, Update, and Delete operations for text records in a ChromaDB collection, along with semantic search. It follows the RAG guide's model: each stored record has a unique ID, text, and optional metadata. Examples are written to be repeatable and avoid relying on a hosted service.

## Create a persistent collection

```python
import chromadb

client = chromadb.PersistentClient(path="./chroma_data")
collection = client.get_or_create_collection(
    name="crud_demo",
    metadata={"hnsw:space": "cosine"},
)
```

When no embedding function is provided, Chroma's default embedding function is used. If a custom embedding function is configured, use the same function for both ingestion and query operations.

## Create: add records

`add` inserts records and expects IDs that are not already present. IDs must be unique within the collection.

```python
collection.add(
    ids=["doc-1", "doc-2", "doc-3"],
    documents=[
        "Python is a programming language used in AI.",
        "The weather is sunny and warm today.",
        "Neural networks learn patterns from data.",
    ],
    metadatas=[
        {"category": "technology", "source": "example"},
        {"category": "weather", "source": "example"},
        {"category": "technology", "source": "example"},
    ],
)
print("Count:", collection.count())
```

Use `upsert` instead of `add` when the operation should insert new IDs and replace records with existing IDs; see Update below.

## Read: inspect and fetch records

`count` returns the number of records. `peek` shows a small sample. `get` reads by ID and can also filter by metadata.

```python
print("Count:", collection.count())
print("Preview:", collection.peek(limit=3))

records = collection.get(
    ids=["doc-1", "doc-3"],
    include=["documents", "metadatas"],
)
for record_id, text, metadata in zip(
    records["ids"], records["documents"], records["metadatas"]
):
    print(record_id, metadata, text)

technology = collection.get(
    where={"category": "technology"},
    include=["documents", "metadatas"],
)
```

`get` is for retrieving records by ID or filter. `query` is for nearest-neighbor search using text or vectors.

## Read: semantic and filtered search

```python
results = collection.query(
    query_texts=["Tell me about AI and neural networks"],
    n_results=2,
    include=["documents", "metadatas", "distances"],
)

for record_id, text, metadata, distance in zip(
    results["ids"][0],
    results["documents"][0],
    results["metadatas"][0],
    results["distances"][0],
):
    print(f"{record_id} | distance={distance:.4f} | {metadata} | {text}")

sports_results = collection.query(
    query_texts=["something fun to do outdoors"],
    n_results=2,
    where={"category": "sports"},
    include=["documents", "metadatas", "distances"],
)
```

Only query up to the number of available records (for example, `n_results=min(2, collection.count())`). A filtered query may return fewer results or none. Distances are metric-dependent and are not confidence probabilities.

## Update: replace existing record fields

`update` modifies records that already exist. Keep the original ID so references to the record remain stable.

```python
collection.update(
    ids=["doc-1"],
    documents=["Python is widely used for machine learning and AI applications."],
    metadatas=[{"category": "technology", "source": "example", "updated": True}],
)
updated = collection.get(ids=["doc-1"], include=["documents", "metadatas"])
print(updated)
```

Provide the fields you intend to change. Re-embedding occurs as needed for changed documents when the collection has an embedding function configured.

## Upsert: insert or replace by ID

`upsert` is useful for repeatable ingestion or synchronizing records from source documents. Existing IDs are replaced; new IDs are created.

```python
collection.upsert(
    ids=["doc-3", "doc-4"],
    documents=[
        "Neural networks learn patterns from data using layers of computation.",
        "The Eiffel Tower is in Paris, France.",
    ],
    metadatas=[
        {"category": "technology", "source": "example"},
        {"category": "travel", "source": "example"},
    ],
)
```

## Delete: remove by ID or metadata filter

Deletes are permanent for the collection. Confirm IDs and filters before deleting. Prefer targeted deletion by IDs when cleaning up chunks from a changed source.

```python
collection.delete(ids=["doc-2"])

# Delete every record matching a metadata filter:
collection.delete(where={"category": "travel"})

print("Remaining IDs:", collection.get()["ids"])
```

When a source document is replaced, a robust ingestion workflow should track its old chunk IDs, delete obsolete chunks, then upsert the current chunks. Otherwise old passages may continue to be retrieved.

## End-to-end CRUD checklist

| Operation | Chroma method | Typical use |
|---|---|---|
| Create | `add` | Insert new records with unique IDs |
| Read | `get`, `peek`, `count` | Inspect records by ID/filter or collection size |
| Search | `query` | Find semantically similar records, optionally filtered |
| Update | `update` | Change fields on existing records |
| Insert-or-update | `upsert` | Idempotent ingestion keyed by stable IDs |
| Delete | `delete` | Remove records by IDs or metadata predicate |

## Practical cautions

- Use persistent storage when data must survive process restarts; an in-memory client is temporary.
- Use stable, deterministic IDs for source documents and chunks.
- Keep source metadata on every chunk so retrieved answers can cite their origin.
- Do not combine vectors generated by incompatible embedding models in the same collection.
- Do not store API keys in the vector database or commit secrets to the repository.
- Check ChromaDB's current API documentation when upgrading versions, and test changes against a disposable collection first.
