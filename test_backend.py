from app.rag import MedicalRAG

#test phase
rag = MedicalRAG(
    index_path="data/medical_rag.index",
    documents_path="data/documents.pkl",
    model_path= NONE,
)


question = "What does ACE2 convert angiotensin II into?"

result = rag.ask(question)


print("=" * 80)
print("ANSWER")
print("=" * 80)

print(result["answer"])


print("\n" + "=" * 80)
print("SOURCES")
print("=" * 80)

for source in result["sources"]:
    print(f"- Abstract ID: {source}")