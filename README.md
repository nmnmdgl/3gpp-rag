# 3GPP RAG Chatbot

This project is built around exactly four 3GPP DOCX knowledge sources:

1. TR 21.905 V19.2.0 — Vocabulary for 3GPP Specifications (Release 19)
2. TS 23.501 V20.2.0 — System architecture for the 5G System; Stage 2 (Release 20)
3. TS 23.502 V20.2.0 — Procedures for the 5G System (5GS); Stage 2 (Release 20)
4. TS 38.300 V15.23.0 — NR and NG-RAN Overall Description; Stage 2 (Release 15)

Pipeline:
DOCX -> ordered block parser -> clause-aware chunks -> local embeddings/Qdrant + BM25 -> hybrid retrieval.

Generation/API/frontend are deliberately separate from ingestion and retrieval.
