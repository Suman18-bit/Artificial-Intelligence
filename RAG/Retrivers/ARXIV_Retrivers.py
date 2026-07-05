from langchain_community.retrievers import ArxivRetriever

arxiv = ArxivRetriever(
    load_max_docs=2,
    top_k_results=2,              # this is what actually limits summary results
    load_all_available_meta=True
)

docs = arxiv.invoke("large language models")

for i, d in enumerate(docs, start=1):
    print(f"\nDocument {i}")
    print(f"Title: {d.metadata.get('Title', 'N/A')}")
    print(f"Authors: {d.metadata.get('Authors', 'N/A')}")
    print(f"Published: {d.metadata.get('Published', 'N/A')}")
    print("Summary:")
    print(d.page_content[:500])
    print("-" * 80)