<p align="center">
    <img src="docs/images/logo.png" alt="Tapyre Logo" width="200" />
</p>

<h1 align="center">Tapyre — Paper Search & Embedding Pipeline 🚀</h1>

<p align="center">
    <em>
        Lightweight research paper ingestion, embedding, and semantic search.<br>
        Extracts PDFs, stores metadata in MySQL, vectors in Qdrant, and exposes a Flask API.<br>
        Perfect for building semantic search over research papers. 🧠📚
    </em>
</p>

---

## 🧠 What is Tapyre?

**Tapyre** is a research infrastructure project designed to make large collections of scientific papers searchable, analyzable, and usable for AI systems.

Modern research datasets contain millions of PDFs that are difficult to search using traditional keyword methods. Tapyre solves this by transforming papers into structured, machine-readable knowledge:

- PDFs are parsed and converted into text  
- Text is split into semantic chunks  
- Each chunk is embedded into vector representations  
- Metadata is stored in a relational database  
- Vectors are stored in a high-performance vector database  

This allows developers, researchers, and AI systems to:

- perform semantic search across papers  
- find meaning, not just keywords  
- build RAG systems (Retrieval-Augmented Generation)  
- analyze scientific trends at scale  
- create intelligent research assistants  

**In short:** Tapyre turns raw scientific PDFs into a searchable AI knowledge base.


## 🌟 Highlights

- **PDF conversion & chunking**  
    Efficiently process and split research papers.
- **Specter-2 style embedding**  
    Generate high-quality vector representations.
- **MySQL metadata storage**  
    Reliable and structured metadata management.
- **Qdrant vector database**  
    Fast and scalable vector search.
- **Docker Compose deployment**  
    One-command setup for API, pipeline, MySQL, and Qdrant.

---

## 🚀 Quick Start

See the [Install Guide](./docs/install-guide.md) for setup instructions.

---

## 📖 Documentation

- [Architecture Overview](./docs/architecture.md)
- [API Overview](./docs/api.md)

---

## 🤝 Contributing

We welcome contributions!  
- Open an issue for major changes  
- Submit pull requests with focused commits  
- Tests and documentation updates are appreciated

---

## 📜 License

See [`LICENSE`](./LICENSE) or contact the maintainer for details.

---

## ✉️ Contact

Questions or issues?  
- [Open an issue](https://github.com/tapyre/tapyre-paper-search/issues)
- Reach out to the repository owner

---
## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=tapyre/tapyre-paper-search&type=date&legend=top-left)](https://www.star-history.com/#tapyre/tapyre-paper-search&type=date&legend=top-left)
