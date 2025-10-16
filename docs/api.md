# API – Endpoint Documentation
<p align="center">
    <img src="images/logo.png" alt="Tapyre Logo" width="200" />
</p>

**Base URL:**

```
http://localhost:8000
```

---

## **GET /** – Healthcheck

Checks the availability of the API and its dependencies (MySQL and Qdrant).

**Response Example:**

```json
{
  "status": "ok",
  "mysql": "up",
  "qdrant": "up"
}
```

**Example cURL:**

```bash
curl -X GET http://localhost:8000/
```

---

## **GET /statistics** – System Statistics

Returns general statistics collected from both MySQL and Qdrant.

**Response Example:**

```json
{
  "mysql": { "papers_count": 1240 },
  "qdrant": { "collections": { "chunks": 105000, "papers": 48000 } }
}
```

**Example cURL:**

```bash
curl -X GET http://localhost:8000/statistics
```

---

## **POST /chunks/semantic** – Semantic Chunk Search

Embeds the input text and retrieves semantically similar chunks from Qdrant.

**Request Body:**

```json
{
  "text": "transformer sparse attention",
  "limit": 10
}
```

**Response Example:**

```json
{
  "results": [
    {
      "id": "chunk_001",
      "score": 0.87,
      "text": "In this work, we explore sparse attention mechanisms...",
      "arxiv_id": "2301.12345"
    }
  ]
}
```

**Example cURL:**

```bash
curl -X POST http://localhost:8000/chunks/semantic \
  -H "Content-Type: application/json" \
  -d '{"text": "transformer sparse attention", "limit": 10}'
```

---

## **POST /chunks/rerank-bm25** – BM25 Re-ranking for Chunks

Performs semantic pre-filtering via Qdrant and re-ranks the results using BM25.

**Request Body:**

```json
{
  "text": "contrastive learning for sentence embeddings",
  "limit": 20
}
```

**Response Example:**

```json
{
  "results": [
    {
      "id": "chunk_045",
      "original_score": 0.79,
      "bm25_score": 13.2,
      "text": "We propose a contrastive loss for improved embeddings...",
      "arxiv_id": "2205.01234"
    }
  ]
}
```

**Example cURL:**

```bash
curl -X POST http://localhost:8000/chunks/rerank-bm25 \
  -H "Content-Type: application/json" \
  -d '{"text": "contrastive learning for sentence embeddings", "limit": 20}'
```

---

## **POST /papers/semantic** – Semantic Paper Search

Finds scientific papers semantically related to the input text.
Combines vector similarity from Qdrant with metadata from MySQL.

**Request Body:**

```json
{
  "text": "multimodal retrieval for scientific articles",
  "limit": 5
}
```

**Response Example:**

```json
{
  "results": [
    {
      "arxiv_id": "2305.01234",
      "title": "A Survey on Multimodal Retrieval",
      "author": "Jane Doe; John Smith",
      "subject": "cs.IR",
      "score": 0.91
    }
  ]
}
```

**Example cURL:**

```bash
curl -X POST http://localhost:8000/papers/semantic \
  -H "Content-Type: application/json" \
  -d '{"text": "multimodal retrieval for scientific articles", "limit": 5}'
```

---

## **POST /papers/rerank-bm25** – BM25 Re-ranking for Papers

Combines semantic vector search with BM25 re-ranking using full-text data from MySQL.

**Request Body:**

```json
{
  "text": "efficient transformer architectures for long documents",
  "limit": 10
}
```

**Response Example:**

```json
{
  "results": [
    {
      "arxiv_id": "2110.00001",
      "bm25_score": 27.8,
      "title": "Long-Document Transformers Revisited",
      "meta": {
        "author": "Rao, A.; Lee, B.",
        "subject": "cs.CL",
        "keywords": "long context, transformer"
      }
    }
  ]
}
```

**Example cURL:**

```bash
curl -X POST http://localhost:8000/papers/rerank-bm25 \
  -H "Content-Type: application/json" \
  -d '{"text": "efficient transformer architectures for long documents", "limit": 10}'
```

---

## ✅ Endpoint Summary

| Method | Endpoint              | Description                                      |
| ------ | --------------------- | ------------------------------------------------ |
| `GET`  | `/`                   | API and database healthcheck                     |
| `GET`  | `/statistics`         | Retrieve MySQL & Qdrant system statistics        |
| `POST` | `/chunks/semantic`    | Semantic vector search for text chunks           |
| `POST` | `/chunks/rerank-bm25` | BM25 re-ranking of semantically retrieved chunks |
| `POST` | `/papers/semantic`    | Semantic vector search for scientific papers     |
| `POST` | `/papers/rerank-bm25` | BM25 re-ranking of semantically retrieved papers |
