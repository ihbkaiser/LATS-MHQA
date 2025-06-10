
import os
import re
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, TypedDict

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig
from chat_models.openai_chat_model import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from searchers.base_searcher import BaseSearcher

# --- Constants ---
MAX_RETRIEVAL_ATTEMPTS = 3
MAX_GENERATION_ATTEMPTS = 3
VECTORSTORE_BATCH_SIZE = 100

# --- Corpus Parsing Function ---
# --- Self-RAG Corpus Parsing Function ---
def parse_corpus(file_path: str) -> List[Document]:
    if not os.path.exists(file_path):
        print(f"Error: Corpus file not found at {file_path}. Please create it or provide the correct path.")
        print("Self-RAG needs a corpus to retrieve information.")
        print("Example: Create a 'data' directory with 'multihoprag_corpus.txt' inside.")
        return []

    documents = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading corpus file '{file_path}': {e}")
        return []

    if not content.strip():
        print(f"Warning: Corpus file '{file_path}' is empty or contains only whitespace.")
        return []

    articles_raw = re.split(r'(Title:)', content)

    if len(articles_raw) <= 1 and not (articles_raw[0].strip().startswith("Title:") if articles_raw else False):
        print(f"ℹCorpus file '{file_path}' does not seem to use 'Title:' markers. Treating as plain text and splitting by double newlines.")
        passages = content.split('\n\n')
        for i, passage_text in enumerate(passages):
            cleaned_passage = passage_text.strip()
            if cleaned_passage:
                documents.append(Document(
                    page_content=cleaned_passage,
                    metadata={"source": f"Part {i+1} from {os.path.basename(file_path)}"}
                ))
        if documents:
             print(f"Parsed {len(documents)} documents using plain text fallback.")
    else:
        parsed_with_titles = 0
        for i in range(1, len(articles_raw), 2):
            text_block = articles_raw[i+1].strip()

            title_match = re.match(r'([^\n]+)\n(.*)', text_block, re.DOTALL)
            current_title = ""
            passage_content = ""

            if title_match:
                current_title = title_match.group(1).strip()
                passage_content = title_match.group(2).strip()
            else:
                current_title = text_block.split('\n')[0].strip()
                passage_content = text_block[len(current_title):].strip()

            if passage_content.lower().startswith("passage:"):
                passage_content = passage_content[len("passage:"):].strip()

            if current_title:
                documents.append(Document(page_content=passage_content if passage_content else "No passage content provided.",
                                          metadata={"source": current_title}))
                parsed_with_titles +=1
            else:
                print(f"Warning: Could not parse a title for block: '{text_block[:100]}...'")
        if parsed_with_titles > 0:
            print(f"Parsed {parsed_with_titles} documents using 'Title:' structure.")

    if not documents:
        print(f"Warning: No documents were successfully parsed from '{file_path}'. Self-RAG might not function correctly.")
    else:
        if 0 < len(documents) < 5 :
            for doc_idx, doc in enumerate(documents[:3]):
                print(f"  Sample Doc {doc_idx+1} - Source: {doc.metadata.get('source', 'N/A')}, Passage (start): {doc.page_content[:80].replace(os.linesep, ' ')}...")
    return documents

# --- Self-RAG Graph State ---
class GraphState(TypedDict):
    question: str
    documents: List[Document]
    answer: str
    relevant_documents: bool
    answer_grounded: bool
    answer_relevant_to_question: bool
    retrieval_attempts: int
    generation_attempts: int
    final_decision: str

# --- Self-RAG LangGraph Nodes and Builder ---
def retrieve_documents_node(state: GraphState) -> GraphState:
    retriever = retrieve_documents_node.vectorstore.as_retriever(search_kwargs={"k": retrieve_documents_node.k})
    docs = retriever.invoke(state["question"])
    return {**state, "documents": docs, "retrieval_attempts": state["retrieval_attempts"] + 1}

def grade_documents_relevance_node(state: GraphState) -> GraphState:
    docs_text = "\n\n".join([doc.page_content for doc in state["documents"]])
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Return JSON {\"relevant\": boolean} based on whether any document is relevant to the question."),
        ("human", "Question: {question}\nDocuments:\n{docs_text}")
    ])
    result = (prompt | grade_documents_relevance_node.llm | JsonOutputParser()).invoke({
        "question": state["question"], "docs_text": docs_text
    })
    return {**state, "relevant_documents": result.get("relevant", False)}

def generate_answer_node(state: GraphState) -> GraphState:
    if not state["relevant_documents"]:
        return {**state, "answer": "No relevant documents found.", "generation_attempts": state["generation_attempts"] + 1}
    context = "\n\n".join([doc.page_content for doc in state["documents"]])
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer using only the provided context. If not answerable, say 'unknown'."),
        ("human", "Context:\n{context}\nQuestion: {question}")
    ])
    answer = (prompt | generate_answer_node.llm | StrOutputParser()).invoke({
        "context": context, "question": state["question"]
    })
    return {**state, "answer": answer, "generation_attempts": state["generation_attempts"] + 1}

def grade_answer_node(state: GraphState) -> GraphState:
    docs_text = "\n\n".join([doc.page_content for doc in state["documents"]])
    prompt_faith = ChatPromptTemplate.from_messages([
        ("system", "Return JSON {\"is_supported\": boolean} if answer is grounded in the context."),
        ("human", "Context:\n{docs}\nAnswer: {ans}")
    ])
    prompt_rel = ChatPromptTemplate.from_messages([
        ("system", "Return JSON {\"is_relevant\": boolean} if answer is relevant to the question."),
        ("human", "Question: {q}\nAnswer: {a}")
    ])
    is_supported = (prompt_faith | grade_answer_node.llm | JsonOutputParser()).invoke({
        "docs": docs_text, "ans": state["answer"]
    }).get("is_supported", False)
    is_relevant = (prompt_rel | grade_answer_node.llm | JsonOutputParser()).invoke({
        "q": state["question"], "a": state["answer"]
    }).get("is_relevant", False)
    return {**state, "answer_grounded": is_supported, "answer_relevant_to_question": is_relevant}

def finalize_successful_answer_node(state: GraphState) -> GraphState:
    return {**state, "final_decision": state["answer"]}

def handle_no_relevant_docs_failure_node(state: GraphState) -> GraphState:
    return {**state, "final_decision": "No relevant documents found after retries."}

def handle_generation_failure_node(state: GraphState) -> GraphState:
    return {**state, "final_decision": f"Failed to generate grounded/relevant answer: {state.get('answer','')}"}

def decide_to_finish_or_retry_retrieval(state: GraphState) -> str:
    if state["relevant_documents"]:
        return "generate_answer"
    return "retrieve_documents" if state["retrieval_attempts"] < MAX_RETRIEVAL_ATTEMPTS else "handle_no_relevant_docs_failure"

def decide_to_finish_or_retry_generation(state: GraphState) -> str:
    if state["answer_grounded"] and state["answer_relevant_to_question"]:
        return "finalize_successful_answer"
    return "generate_answer" if state["generation_attempts"] < MAX_GENERATION_ATTEMPTS else "handle_generation_failure"

def build_self_rag_graph(llm: ChatOpenAI, vectorstore: FAISS, k: int) -> StateGraph:
    retrieve_documents_node.vectorstore = vectorstore
    retrieve_documents_node.k = k
    grade_documents_relevance_node.llm = llm
    generate_answer_node.llm = llm
    grade_answer_node.llm = llm

    builder = StateGraph(GraphState)
    builder.add_node("retrieve_documents", retrieve_documents_node)
    builder.add_node("grade_documents_relevance", grade_documents_relevance_node)
    builder.add_node("generate_answer", generate_answer_node)
    builder.add_node("grade_answer", grade_answer_node)
    builder.add_node("finalize_successful_answer", finalize_successful_answer_node)
    builder.add_node("handle_no_relevant_docs_failure", handle_no_relevant_docs_failure_node)
    builder.add_node("handle_generation_failure", handle_generation_failure_node)

    builder.set_entry_point("retrieve_documents")
    builder.add_edge("retrieve_documents", "grade_documents_relevance")
    builder.add_conditional_edges("grade_documents_relevance", decide_to_finish_or_retry_retrieval, {
        "generate_answer": "generate_answer",
        "retrieve_documents": "retrieve_documents",
        "handle_no_relevant_docs_failure": "handle_no_relevant_docs_failure"
    })
    builder.add_edge("generate_answer", "grade_answer")
    builder.add_conditional_edges("grade_answer", decide_to_finish_or_retry_generation, {
        "generate_answer": "generate_answer",
        "finalize_successful_answer": "finalize_successful_answer",
        "handle_generation_failure": "handle_generation_failure"
    })
    builder.add_edge("finalize_successful_answer", END)
    builder.add_edge("handle_no_relevant_docs_failure", END)
    builder.add_edge("handle_generation_failure", END)
    return builder

# --- Vectorstore + Graph Initializer ---
def initialize_vectorstore_and_self_rag_graph(
    FAISS_INDEX_STORE_PATH: str,
    CORPUS_FILE_PATH: str,
    embeddings: OpenAIEmbeddings,
    llm: ChatOpenAI,
    k: int = 3
) :
    vectorstore = None
    self_rag_app = None
    if os.path.exists(FAISS_INDEX_STORE_PATH):
        try:
            vectorstore = FAISS.load_local(FAISS_INDEX_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
            self_rag_app = build_self_rag_graph(llm, vectorstore, k)
            return vectorstore, self_rag_app
        except Exception:
            pass
    corpus_docs = parse_corpus(CORPUS_FILE_PATH)
    vectorstore = FAISS.from_documents(documents=corpus_docs[:VECTORSTORE_BATCH_SIZE], embedding=embeddings)
    if len(corpus_docs) > VECTORSTORE_BATCH_SIZE:
        for i in range(VECTORSTORE_BATCH_SIZE, len(corpus_docs), VECTORSTORE_BATCH_SIZE):
            vectorstore.add_documents(corpus_docs[i:i + VECTORSTORE_BATCH_SIZE])
    try:
        os.makedirs(FAISS_INDEX_STORE_PATH, exist_ok=True)
        vectorstore.save_local(FAISS_INDEX_STORE_PATH, embeddings)
    except Exception:
        pass
    self_rag_app = build_self_rag_graph(llm, vectorstore, k)
    return vectorstore, self_rag_app

# --- SelfRAGSearcher ---
class SelfRAGSearcher(BaseSearcher):
    def __init__(
        self,
        faiss_index_path: str,
        corpus_path: str,
        llm: ChatOpenAI,
        embeddings: OpenAIEmbeddings,
        k: int = 3,
        config: Optional[RunnableConfig] = None
    ):
        self.vectorstore, self.self_rag_app = initialize_vectorstore_and_self_rag_graph(
            FAISS_INDEX_STORE_PATH=faiss_index_path,
            CORPUS_FILE_PATH=corpus_path,
            embeddings=embeddings,
            llm=llm,
            k=k
        )
        self.k = k
        self.llm = llm
        self.config = config or RunnableConfig()

    def retrieve(self, query: str) -> List[str]:
        sub_q_answer_from_self_rag = f"Self-RAG: Default - No answer processed for '{query[:50]}...'."
        retrieved_docs_for_this_sub_q: List[Document] = []

        state: GraphState = {
            'question': query,
            'documents': [],
            'answer': '',
            'relevant_documents': False,
            'answer_grounded': False,
            'answer_relevant_to_question': False,
            'retrieval_attempts': 0,
            'generation_attempts': 0,
            'final_decision': ''
        }

        try:
            self_rag_final_state_output = self.self_rag_app.invoke(state, config=self.config)

            if isinstance(self_rag_final_state_output, dict):
                sub_q_answer_from_self_rag = self_rag_final_state_output.get(
                    "final_decision",
                    f"Self-RAG: No 'final_decision' key in output for '{query[:50]}...'."
                )
                retrieved_docs_for_this_sub_q = self_rag_final_state_output.get("documents", [])
            else:
                sub_q_answer_from_self_rag = f"Self-RAG: Unexpected output type. Got {type(self_rag_final_state_output)}."

        except Exception as e:
            sub_q_answer_from_self_rag = f"Self-RAG Error: {str(e)[:100]}"

        if not retrieved_docs_for_this_sub_q:
            retriever = self.vectorstore.as_retriever(search_kwargs={'k': self.k})
            retrieved_docs_for_this_sub_q = retriever.invoke(query)

        return [doc.page_content for doc in retrieved_docs_for_this_sub_q]
