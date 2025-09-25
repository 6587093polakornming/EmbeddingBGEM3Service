from services.embedding_service import EmbeddingService

service = EmbeddingService(device="cuda")  # or "cuda" / "cpu"

print("Query dim:", len(service.embed_query("อธิบาย RAG สั้นๆ")))

vecs = service.embed_texts(["hello", "สวัสดี"])
print("Batch dims:", [len(v) for v in vecs])

chunks = service.embed_file("docs/sample.pdf")
print("PDF chunks:", len(chunks), "first vec dim:", len(chunks[0].vector) if chunks else 0)

rows = service.embed_file("data/articles.csv", csv_text_cols=["title", "summary"])
print("CSV chunks:", len(rows))
