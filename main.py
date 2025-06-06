from ollama import embeddings,chat,Message
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
import numpy as np

class KB:
    def __init__(self,filepath) -> None:
        with open(filepath, 'r', encoding="utf-8") as file:
            self.data = file.read().splitlines()
        self.embed = self.encode()
    def encode(self):
        embed = []
        for text in self.data:
            response = embeddings(model="bge-m3:latest", prompt=text)
            embed.append(response["embedding"])
        return torch.tensor(embed, device=device)

    @staticmethod
    def similarity(embedding1, embedding2):
        # 计算余弦相似度
        cos_sim = torch.nn.functional.cosine_similarity(embedding1, embedding2, dim=0)
        return cos_sim.item()

    def search(self, query, top_k=1):
        """
        在知识库中搜索与查询前k个相似的条目
        """
        query_embedding = embeddings(model="bge-m3:latest", prompt=query)["embedding"]
        query_tensor = torch.tensor(query_embedding, device=device)
        similarities = [self.similarity(query_tensor, emb) for emb in self.embed]
        top_k_indices = np.argsort(similarities)[-top_k:][::-1]
        return [(self.data[i], similarities[i]) for i in top_k_indices]

class Rag:
    def __init__(self, model, kb: KB):
        self.model = model
        self.kb = kb
        self.prompt = "Based on {knowledge_base}, answer the question: {question}"

    def chatwithkb(self, question, top_k=1):
        """
        使用知识库回答问题
        """
        results = self.kb.search(question, top_k)
        print(f"查询结果: {results}")
        print("=" * 20)
        if results:
            knowledge_base = "\n".join([f"{i+1}. {item[0]} (相似度: {item[1]:.4f})" for i, item in enumerate(results)])
            prompt = self.prompt.format(knowledge_base=knowledge_base, question=question)
            response = chat(model=self.model, messages=[{"role": "user", "content": prompt}])
            return response["message"]
        else:
            return "No relevant information found."
    def chatwithoutkb(self, question):
        """
        不使用知识库直接回答问题
        """
        response = chat(model=self.model, messages=[{"role": "user", "content": question}])
        return response["message"]

testkb = KB("testdata.txt")
testRag = Rag("qwen2.5:latest", testkb)
# 问答聊天测试
# while True:
#     question = input("请输入问题: ")
#     if question.lower() == "exit":
#         break
#     answer = testRag.chatwithkb(question, top_k=1)
#     print(f"回答: {answer}")

# 无法做出正确回答的问题测试
with open("unsupported_questions.txt", "r", encoding="utf-8") as f:
    unsupported = f.read().splitlines()
    for question in unsupported:
        answerwithoutkb = testRag.chatwithoutkb(question)
        print(f"问题: {question}")
        print(f"无知识库回答: {answerwithoutkb}")
        print("=" * 20)
        answerwithkb = testRag.chatwithkb(question, top_k=1)
        print(f"有知识库回答: {answerwithkb}")
        print("=" * 20)
