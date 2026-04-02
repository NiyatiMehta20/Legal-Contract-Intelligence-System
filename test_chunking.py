from chunk_contract import chunk_contract

with open("contracts/BAMA-NDA_ASTRO.pdf.txt", "r", encoding="utf-8") as f:
    text = f.read()

chunks = chunk_contract(text)
print(f"Total chunks: {len(chunks)}\n")

for i, c in enumerate(chunks):
    print("=" * 60)
    print(f"CHUNK {i+1} | TYPE: {c['type']}")
    print(c["text"][:500])
