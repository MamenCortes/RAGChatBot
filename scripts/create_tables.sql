CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS rag_documents (
  doc_id TEXT PRIMARY KEY,
  source_path TEXT,
  fingerprint TEXT NOT NULL,
  num_pages INTEGER,
  updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS rag_chunks (
  id BIGSERIAL PRIMARY KEY,
  doc_id TEXT NOT NULL,
  chunk_id TEXT NOT NULL,
  page_num INTEGER,
  topic TEXT,
  source TEXT,
  lang TEXT,
  content TEXT NOT NULL,
  embedding vector(384) NOT NULL,
  created_at TIMESTAMPTZ DEFAULT now(),
  UNIQUE (doc_id, chunk_id),

  CONSTRAINT fk_rag_chunks_doc
    FOREIGN KEY (doc_id)
    REFERENCES rag_documents (doc_id)
    ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS rag_chunks_embedding_idx
  ON rag_chunks
  USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS rag_chunks_doc_id_idx ON rag_chunks (doc_id);
CREATE INDEX IF NOT EXISTS rag_chunks_topic_idx ON rag_chunks (topic);