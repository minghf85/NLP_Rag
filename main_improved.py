from ollama import embeddings,chat,Message
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
import numpy as np
from scholarly import scholarly
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

    def search(self, query, top_k=1, similarity_threshold=0.5):
        """
        在知识库中搜索与查询前k个相似的条目
        """
        query_embedding = embeddings(model="bge-m3:latest", prompt=query)["embedding"]
        query_tensor = torch.tensor(query_embedding, device=device)
        similarities = [self.similarity(query_tensor, emb) for emb in self.embed]

        # 应用相似度阈值
        filtered_indices = [i for i, sim in enumerate(similarities) if sim > similarity_threshold]
        if not filtered_indices:
            print("没有找到满足相似度阈值的条目")
            return []

        # 获取前k个相似条目
        top_k_indices = np.argsort(similarities)[-top_k:][::-1]
        return [(self.data[i], similarities[i]) for i in top_k_indices]

class Rag:
    def __init__(self, model, kb: KB):
        self.model = model
        self.kb = kb
        self.prompt = "Based on {knowledge_base}, answer the question: {question}"

    def chatwithkb(self, question):
        """
        使用知识库回答问题
        """
        results = self.kb.search(question)
        while not results:
            print("没有找到相关信息，尝试网络搜索...")
            """没有则网络搜索，下面以scholarly搜索论文为例"""
            search_prompt = f"帮我输出这段话{question}论文搜索的一个英文关键词，只需要一个关键词，不要其他内容，不要自己添加多余的词或者符号，尽量简短，格式为：keyword"
            
            keywordresponse = chat(model=self.model, messages=[{"role": "user", "content": search_prompt}])
            print(f"关键词: {keywordresponse['message']['content']}")
            pub= scholarly.search_single_pub(keywordresponse["message"]["content"])
            # 提取核心信息并格式化为一行文本
            pub_info = f"Title: {pub['bib'].get('title', 'N/A')} | Authors: {', '.join(pub['bib'].get('author', ['N/A']))[:50]}... | Year: {pub['bib'].get('pub_year', 'N/A')} | Citations: {pub.get('num_citations', 0)}"
            print(f"找到论文: {pub_info}")
            # 添加到KB中
            pubembedding = embeddings(model="bge-m3:latest", prompt=str(pub_info))["embedding"]
            self.kb.data.append(str(pub_info))
            self.kb.embed = torch.cat((self.kb.embed, torch.tensor([pubembedding], device=device)), dim=0)
            print("重新查询知识库...")
            results = self.kb.search(question)
        print(f"查询结果: {results}")
        print("=" * 20)
        knowledge_base = "\n".join([f"{i+1}. {item[0]} (相似度: {item[1]:.4f})" for i, item in enumerate(results)])
        prompt = self.prompt.format(knowledge_base=knowledge_base, question=question)
        response = chat(model=self.model, messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"]
    def chatwithoutkb(self, question):
        """
        不使用知识库直接回答问题
        """
        response = chat(model=self.model, messages=[{"role": "user", "content": question}])
        return response["message"]["content"]

testkb = KB("testdata.txt")
testRag = Rag("qwen2.5:latest", testkb)
# 问答聊天测试
# while True:
#     question = input("请输入问题: ")
#     if question.lower() == "exit":
#         break
#     answer = testRag.chatwithkb(question)
#     print(f"回答: {answer}")

# 无法做出正确回答的问题测试
with open("unsupported_questions_improved.txt", "r", encoding="utf-8") as f:
    unsupported = f.read().splitlines()
    with open("unsupported_questions_answers_improved.txt", "w", encoding="utf-8") as f_out:
        for question in unsupported:
            answerwithoutkb = testRag.chatwithoutkb(question)
            print(f"问题: {question}")
            f_out.write(f"问题: {question}\n")
            print(f"无知识库回答: {answerwithoutkb}")
            f_out.write(f"无知识库回答: {answerwithoutkb}\n")
            f_out.write("=" * 20 + "\n")
            print("=" * 20)
            answerwithkb = testRag.chatwithkb(question)
            print(f"有知识库回答: {answerwithkb}")
            print("=" * 20)
            f_out.write(f"有知识库回答: {answerwithkb}\n")
            f_out.write("=" * 20 + "\n")
