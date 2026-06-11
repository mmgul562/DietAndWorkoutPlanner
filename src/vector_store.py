import chromadb
from chromadb.utils import embedding_functions
from config import CHROMA_DB_DIR, EMBEDDING_MODEL_NAME


class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL_NAME
        )
        self.exercise_collection = self.client.get_or_create_collection(
            name="exercises",
            embedding_function=self.embedding_fn
        )
        self.injury_collection = self.client.get_or_create_collection(
            name="injury_knowledge",
            embedding_function=self.embedding_fn
        )
        self.diet_collection = self.client.get_or_create_collection(
            name="diet_knowledge",
            embedding_function=self.embedding_fn
        )

    def index_exercises(self, enriched_list: list[dict], batch_size=5000):
        docs = []
        metas = []
        ids = []
        for i, ex in enumerate(enriched_list):
            text_parts = [
                f"Name: {ex['name']}",
                f"Body parts: {', '.join(ex['body_parts'])}",
                f"Target muscles: {', '.join(ex['target_muscles'])}",
                f"Secondary muscles: {', '.join(ex['secondary_muscles'])}",
                f"Equipments: {', '.join(ex['equipments'])}"
            ]
            if ex.get("instructions"):
                text_parts.append(f"Instructions: {ex['instructions']}")
            if ex.get("safety_info"):
                text_parts.append(f"Safety info: {ex['safety_info']}")

            doc_text = ". ".join(text_parts)
            docs.append(doc_text)
            metas.append({
                "exercise_id": ex.get("exercise_id", ""),
                "name": ex["name"],
                "body_parts": ", ".join(ex["body_parts"]),
                "target_muscles": ", ".join(ex["target_muscles"]),
                "secondary_muscles": ", ".join(ex["secondary_muscles"]),
                "equipments": ", ".join(ex["equipments"]),
                "has_instructions": bool(ex.get("instructions")),
                "safety_info": ex.get("safety_info", "")
            })
            ids.append(str(i))

        # Dodawanie w paczkach
        total = len(ids)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            print(f"Adding exercises {start}-{end - 1} out of {total}...")
            self.exercise_collection.add(
                documents=docs[start:end],
                metadatas=metas[start:end],
                ids=ids[start:end]
            )

    def index_injury_knowledge(self, knowledge_items: list[dict]):
        docs = [item["text"] for item in knowledge_items]
        metas = [{"injury": item["injury"]} for item in knowledge_items]
        ids = [f"inj_{i}" for i in range(len(knowledge_items))]
        self.injury_collection.add(documents=docs, metadatas=metas, ids=ids)

    def index_diet_knowledge(self, knowledge_items: list[dict]):
        docs = [item["text"] for item in knowledge_items]
        metas = [{"topic": item["topic"]} for item in knowledge_items]
        ids = [f"diet_{i}" for i in range(len(knowledge_items))]
        self.diet_collection.add(documents=docs, metadatas=metas, ids=ids)

    def search_exercises(self, query: str, n_results=5):
        return self.exercise_collection.query(query_texts=[query], n_results=n_results)

    def search_injury_guidelines(self, injury: str, n_results=3):
        return self.injury_collection.query(query_texts=[injury], n_results=n_results)

    def search_diet_knowledge(self, query: str, n_results=3):
        return self.diet_collection.query(query_texts=[query], n_results=n_results)
