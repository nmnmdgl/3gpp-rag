from src.retrieval.retriever import Retriever


retriever = Retriever(
    dense_k=50,
    bm25_k=50,
    final_k=50,
    rerank_k=20,
)

query = "What is the AMF in the 5G System?"

documents = retriever.retrieve(query)

print("\n" + "=" * 80)
print("RETRIEVAL DEBUG")
print("=" * 80)

print(f"Query: {query}")
print(f"Documents returned: {len(documents)}")

for i, document in enumerate(documents, start=1):

    print("\n" + "-" * 80)

    print(f"RANK        : {i}")
    print(f"ID          : {document.get('id')}")
    print(f"DOCUMENT ID : {document.get('document_id')}")
    print(f"SPEC NUMBER : {document.get('spec_number')}")
    print(f"TITLE       : {document.get('title')}")
    print(f"CLAUSE      : {document.get('clause')}")
    print(f"CLAUSE TITLE: {document.get('clause_title')}")
    print(f"BM25 SCORE  : {document.get('bm25_score')}")
    print(f"SIMILARITY  : {document.get('similarity')}")
    print(f"RERANK SCORE: {document.get('rerank_score')}")

    print("\nTEXT:")
    print(document.get("text", "")[:2500])

print("\n" + "=" * 80)
print("END RETRIEVAL DEBUG")
print("=" * 80)