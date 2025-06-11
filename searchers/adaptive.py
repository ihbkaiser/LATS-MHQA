import os
import re
from typing import List, Optional, Literal
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from searchers.base_searcher import BaseSearcher
from main import Node

from langchain.schema import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END



class CorpusBuilder:
    #build corpus and chroma vectorstores
    def __init__(self,
                 chunk_size: int = 800,
                 chunk_overlap: int = 50,
                 persist_dir: str = "chroma_db",
                 embedding_model: Optional[OpenAIEmbeddings] = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.persist_dir = persist_dir
        self.embedding_model = embedding_model or OpenAIEmbeddings()

    def build_index(self, file_path: str) -> Chroma:
        docs = self._build_corpus(file_path)
        if not docs:
            raise ValueError("No documents extracted . Check input file format.")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        chunks = splitter.split_documents(docs)
        print(f"[Info] Split into {len(chunks)} chunks (chunk_size={self.chunk_size}, overlap={self.chunk_overlap})")

        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=self.embedding_model,
            persist_directory=self.persist_dir,
        )
        print("[Info] Chroma index built and persisted ✅")
        return vectordb
    
    @staticmethod
    def _build_corpus(file_path: str) -> List[Document]:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        docs: List[Document] = []
        use_title = "Title:" in text
        if use_title:
            parts = re.split(r"(Title:)", text)
            for i in range(1, len(parts), 2):
                block = parts[i + 1].strip()
                m = re.match(r"([^\n]+)\n(.*)", block, re.DOTALL)
                title, body = (m.group(1).strip(), m.group(2).strip()) if m else (block.strip(), "")
                if body.lower().startswith("passage:"):
                    body = body[len("passage:"):].strip()
                docs.append(Document(page_content=body or "No content", metadata={"source": title}))
        else:
            for idx, seg in enumerate(text.split("\n\n")):
                seg = seg.strip()
                if not seg:
                    continue
                if seg.lower().startswith("passage:"):
                    seg = seg[len("passage:"):].strip()
                src = f"Part {idx + 1} from {os.path.basename(file_path)}"
                docs.append(Document(page_content=seg, metadata={"source": src}))
        return docs


class GraphState(TypedDict):
    question: str
    documents: List[Document]
    generation: str

class RouteQuery(BaseModel):
    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )

class AdaptiveRAG:
    """Encapsulates the adaptive RAG graph logic."""

    def __init__(self, retriever, rag_chain, question_router,
                 retrieval_grader, question_rewriter,
                 hallucination_grader, answer_grader,
                 web_search_tool):
        self.retriever = retriever
        self.rag_chain = rag_chain
        self.question_router = question_router
        self.retrieval_grader = retrieval_grader
        self.question_rewriter = question_rewriter
        self.hallucination_grader = hallucination_grader
        self.answer_grader = answer_grader
        self.web_search_tool = web_search_tool
        self.app = self._build_graph()



    def answer(self, question: str) -> GraphState:
        state: GraphState = {"question": question, "documents": [], "generation": ""}
        return self.app.invoke(state)


    def _build_graph(self):

        workflow = StateGraph(GraphState)

        workflow.add_node("web_search", self._web_search_node)
        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("grade_documents", self._grade_documents)
        workflow.add_node("generate", self._generate)
        workflow.add_node("transform_query", self._transform_query)

        workflow.add_conditional_edges(
            START,
            self._route_question,
            {"web_search": "web_search", "vectorstore": "retrieve"},
        )
        workflow.add_edge("web_search", "generate")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            self._decide_to_generate,
            {"transform_query": "transform_query", "generate": "generate"},
        )
        workflow.add_edge("transform_query", "retrieve")
        workflow.add_conditional_edges(
            "generate",
            self._grade_generation,
            {"not supported": "generate", "useful": END, "not useful": "transform_query"},
        )

        return workflow.compile()

    def _route_question(self, state: GraphState):
        source = self.question_router.invoke({"question": state["question"]})
        return "web_search" if source.datasource == "web_search" else "vectorstore"

    def _retrieve(self, state: GraphState):
        docs = self.retriever.invoke(state["question"])
        return {"question": state["question"], "documents": docs}

    def _generate(self, state: GraphState):
        generation = self.rag_chain.invoke({"context": state["documents"], "question": state["question"]})
        return {**state, "generation": generation}

    def _grade_documents(self, state: GraphState):
        filtered_docs: List[Document] = []
        for doc in state["documents"]:
            if self.retrieval_grader.invoke({"question": state["question"], "document": doc.page_content}).binary_score == "yes":
                filtered_docs.append(doc)
        return {"question": state["question"], "documents": filtered_docs}

    def _decide_to_generate(self, state: GraphState):
        return "generate" if state["documents"] else "transform_query"

    def _transform_query(self, state: GraphState):
        better_q = self.question_rewriter.invoke({"question": state["question"]})
        return {"question": better_q, "documents": state["documents"]}

    def _web_search_node(self, state: GraphState):
        results = self.web_search_tool.invoke({"query": state["question"]})
        combined = "\n".join([r["content"] for r in results])
        return {"question": state["question"], "documents": [Document(page_content=combined)]}

    def _grade_generation(self, state: GraphState):
        grounded = self.hallucination_grader.invoke({"documents": state["documents"], "generation": state["generation"]}).binary_score == "yes"
        relevant = self.answer_grader.invoke({"question": state["question"], "generation": state["generation"]}).binary_score == "yes"
        if grounded and relevant:
            return "useful"
        elif not grounded:
            return "not supported"
        else:
            return "not useful"


class AdaptiveSearcher(BaseSearcher):
    """Encapsulates MCTS logic for question decomposition and evidence gathering."""

    MAX_HOP = 5

    def __init__(self, sub_question_chain, sub_answer_chain, action_chain,
                 reflection_llm_chain, web_search_tool):
        self.sub_question_chain = sub_question_chain
        self.sub_answer_chain = sub_answer_chain
        self.action_chain = action_chain
        self.reflection_llm_chain = reflection_llm_chain
        self.web_search_tool = web_search_tool

    # ---------------- MCTS Phases ----------------

    def select(self, node: Node, exploration_weight: float, verbose: bool) -> Node:
        current = node
        while current.children:
            unvisited = [c for c in current.children if c.visits == 0]
            if unvisited:
                return unvisited[0]
            current = max(current.children, key=lambda c: c.upper_confidence_bound(exploration_weight))
        return current

    def expand(self, node: Node, config: RunnableConfig, verbose: bool):
        question_msg = node.get_trajectory()[0]
        rag_app = self._build_adaptive_rag()  # new instance per expand for isolation
        state = {"question": question_msg.content, "generation": "", "documents": []}
        result = rag_app.answer(question_msg.content)
        docs = result.get("documents", [])
        answer = result.get("generation", "")
        docs_msgs = [HumanMessage(content=doc.page_content) for doc in docs]
        sub_q_resp = self.sub_question_chain.invoke(
            {"Question": [question_msg], "Question_history": docs_msgs},
            config,
        )
        subq_list = [
            line.replace("Sub-question:", "").strip()
            for line in sub_q_resp.content.strip().splitlines()
            if line.strip().startswith("Sub-question:")
        ]
        for sq in subq_list:
            child = Node(messages=[HumanMessage(content=sq)], reflection=node.reflection, parent=node)
            node.children.append(child)

    def simulate_and_backprop(self, node: Node, config: RunnableConfig, verbose: bool):
        path = node.get_trajectory()
        question_msg = path[0]
        accumulated_paragraphs: List[str] = []
        evidence_msgs: List[HumanMessage] = []
        current = node
        hops = 0
        while current.depth > 0 and hops < self.MAX_HOP:
            sub_q = current.messages[0].content
            retrievals = self.web_search_tool.invoke({"query": sub_q})
            paras = [r["content"] for r in retrievals]
            accumulated_paragraphs.extend(paras)
            paras_msgs = [HumanMessage(content=p) for p in accumulated_paragraphs]
            sub_ans_obj = self.sub_answer_chain.invoke(
                {
                    "Paragraphs": paras_msgs,
                    "Evidence": [HumanMessage(content=sub_q)],
                    "Question": [question_msg],
                    "Sub_question": [HumanMessage(content=sub_q)],
                },
                config,
            )
            evidence_msgs.append(HumanMessage(content=f"Sub-question: {sub_q}\nSub-answer: {sub_ans_obj.content}"))
            if hops >= self.MAX_HOP - 1:
                break
            next_action = self.action_chain.invoke(
                {"Question": [question_msg], "Question_history": evidence_msgs},
                config,
            ).content.strip()
            next_node = Node(messages=[HumanMessage(content=next_action)], reflection=node.reflection, parent=current)
            current.children.append(next_node)
            current = next_node
            hops += 1
        reflection_obj = self.reflection_llm_chain.invoke(
            {"Question": [question_msg], "Question_history": evidence_msgs},
            config,
        )[0]
        node.backpropagate(reflection_obj.normalized_score, verbose)


    def search(self, root: Node, config: RunnableConfig, budget: int = 30, exploration_weight: float = 1.0, verbose: bool = False):
        for _ in range(budget):
            leaf = self.select(root, exploration_weight, verbose)
            if leaf.depth <= self.MAX_HOP:
                self.expand(leaf, config, verbose)
            for child in leaf.children:
                if child.visits == 0:
                    self.simulate_and_backprop(child, config, verbose)
        return root 


    def _build_adaptive_rag(self) -> AdaptiveRAG:
        return AdaptiveRAG(
        retriever=self.vectorstore.as_retriever(search_kwargs={"k": self.k}),
        rag_chain=self.rag_generation_chain,
        question_router=self.router_chain,
        retrieval_grader=self.doc_relevance_chain,
        question_rewriter=self.question_rewrite_chain,
        hallucination_grader=self.hallu_chain,
        answer_grader=self.answer_grade_chain,
        web_search_tool=self.web_search_tool,
    )
