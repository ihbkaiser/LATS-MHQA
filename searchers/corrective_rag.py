import os
import re
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, TypedDict

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langgraph.graph import StateGraph, END, START
from langchain_core.runnables import RunnableConfig
from chat_models.openai_chat_model import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
from pydantic import BaseModel, Field

from .base_searcher import BaseSearcher

# --- Constants ---
MAX_RETRIEVAL_ATTEMPTS = 3
VECTORSTORE_BATCH_SIZE = 100

# --- Corpus Parsing Function ---
def parse_corpus(file_path: str) -> List[Document]:
    """Parse corpus file and return list of documents."""
    if not os.path.exists(file_path):
        print(f"Error: Corpus file not found at {file_path}. Please create it or provide the correct path.")
        print("Corrective-RAG needs a corpus to retrieve information.")
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
        print(f"Warning: No documents were successfully parsed from '{file_path}'. Corrective-RAG might not function correctly.")
    else:
        if 0 < len(documents) < 5 :
            for doc_idx, doc in enumerate(documents[:3]):
                print(f"  Sample Doc {doc_idx+1} - Source: {doc.metadata.get('source', 'N/A')}, Passage (start): {doc.page_content[:80].replace(os.linesep, ' ')}...")
    return documents

# --- Data Model for Grading Documents ---
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

# --- CRAG Graph State ---
class GraphState(TypedDict):
    """Represents the state of our graph."""
    question: str
    generation: str
    web_search: str
    documents: List[Document]

# --- Helper Functions ---
def format_docs(docs):
    """Format documents for context."""
    return "\n\n".join(doc.page_content for doc in docs)

# --- CRAG LangGraph Nodes ---
def retrieve_node(state: GraphState) -> GraphState:
    """Retrieve documents"""
    print("---RETRIEVE---")
    question = state["question"]
    
    # Retrieval
    documents = retrieve_node.retriever.invoke(question)
    return {"documents": documents, "question": question}

def generate_node(state: GraphState) -> GraphState:
    """Generate answer"""
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    
    # RAG generation
    generation = generate_node.rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

def grade_documents_node(state: GraphState) -> GraphState:
    """Determines whether the retrieved documents are relevant to the question."""
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    
    # Score each doc
    filtered_docs = []
    for d in documents:
        score = grade_documents_node.retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
    
    # Determine web_search based on filtered results
    web_search = "Yes" if len(filtered_docs) == 0 else "No"
    return {"documents": filtered_docs, "question": question, "web_search": web_search}

def transform_query_node(state: GraphState) -> GraphState:
    """Transform the query to produce a better question."""
    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]
    
    # Re-write question
    better_question = transform_query_node.question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}

def web_search_node(state: GraphState) -> GraphState:
    """Web search based on the re-phrased question."""
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]
    
    # Web search
    docs = web_search_node.web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    documents.append(web_results)
    
    return {"documents": documents, "question": question}

# --- Edge Decision Functions ---
def decide_to_generate(state: GraphState) -> str:
    """Determines whether to generate an answer, or re-generate a question."""
    print("---ASSESS GRADED DOCUMENTS---")
    web_search = state["web_search"]
    
    if web_search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---")
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"

# --- Graph Builder ---
def build_corrective_rag_graph(
    llm: ChatOpenAI, 
    vectorstore: FAISS, 
    k: int, 
    web_search_tool: TavilySearchResults
) -> StateGraph:
    """Build the CRAG workflow graph."""
    
    # Set up components
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    
    # LLM with function call for grading
    structured_llm_grader = llm.with_structured_output(GradeDocuments)
    
    # Grading prompt
    system = """You are a grader assessing relevance of a retrieved document to a user question. 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. 
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ])
    retrieval_grader = grade_prompt | structured_llm_grader
    
    # RAG prompt
    rag_prompt = hub.pull("rlm/rag-prompt")
    rag_chain = rag_prompt | llm | StrOutputParser()
    
    # Question rewriter
    rewrite_system = """You a question re-writer that converts an input question to a better version that is optimized 
     for web search. Look at the input and try to reason about the underlying semantic intent / meaning."""
    re_write_prompt = ChatPromptTemplate.from_messages([
        ("system", rewrite_system),
        ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question."),
    ])
    question_rewriter = re_write_prompt | llm | StrOutputParser()
    
    # Attach components to nodes
    retrieve_node.retriever = retriever
    generate_node.rag_chain = rag_chain
    grade_documents_node.retrieval_grader = retrieval_grader
    transform_query_node.question_rewriter = question_rewriter
    web_search_node.web_search_tool = web_search_tool
    
    # Build workflow
    workflow = StateGraph(GraphState)
    
    # Define the nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("transform_query", transform_query_node)
    workflow.add_node("web_search_node", web_search_node)
    
    # Build graph
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )
    workflow.add_edge("transform_query", "web_search_node")
    workflow.add_edge("web_search_node", "generate")
    workflow.add_edge("generate", END)
    
    return workflow

# --- Vectorstore + Graph Initializer ---
def initialize_vectorstore_and_corrective_rag_graph(
    FAISS_INDEX_STORE_PATH: str,
    CORPUS_FILE_PATH: str,
    embeddings: OpenAIEmbeddings,
    llm: ChatOpenAI,
    k: int = 3,
    web_search_tool: Optional[TavilySearchResults] = None
):
    """Initialize vectorstore and CRAG graph."""
    vectorstore = None
    corrective_rag_app = None
    
    # Try to load existing vectorstore
    if os.path.exists(FAISS_INDEX_STORE_PATH):
        try:
            vectorstore = FAISS.load_local(FAISS_INDEX_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
            corrective_rag_app = build_corrective_rag_graph(llm, vectorstore, k, web_search_tool)
            return vectorstore, corrective_rag_app.compile()
        except Exception:
            pass
    
    # Create new vectorstore from corpus
    corpus_docs = parse_corpus(CORPUS_FILE_PATH)
    if not corpus_docs:
        print("No documents found in corpus. Using empty vectorstore.")
        corpus_docs = [Document(page_content="Empty corpus", metadata={"source": "empty"})]
    
    vectorstore = FAISS.from_documents(documents=corpus_docs[:VECTORSTORE_BATCH_SIZE], embedding=embeddings)
    if len(corpus_docs) > VECTORSTORE_BATCH_SIZE:
        for i in range(VECTORSTORE_BATCH_SIZE, len(corpus_docs), VECTORSTORE_BATCH_SIZE):
            vectorstore.add_documents(corpus_docs[i:i + VECTORSTORE_BATCH_SIZE])
    
    # Save vectorstore
    try:
        os.makedirs(os.path.dirname(FAISS_INDEX_STORE_PATH), exist_ok=True)
        vectorstore.save_local(FAISS_INDEX_STORE_PATH)
    except Exception as e:
        print(f"Warning: Could not save vectorstore: {e}")
    
    corrective_rag_app = build_corrective_rag_graph(llm, vectorstore, k, web_search_tool)
    return vectorstore, corrective_rag_app.compile()

# --- CorrectiveRAGSearcher ---
class CorrectiveRAGSearcher(BaseSearcher):
    """Corrective RAG searcher implementing CRAG workflow."""
    
    def __init__(
        self,
        faiss_index_path: str,
        corpus_path: str,
        llm: ChatOpenAI,
        embeddings: OpenAIEmbeddings,
        k: int = 3,
        web_search_tool: Optional[TavilySearchResults] = None,
        config: Optional[RunnableConfig] = None
    ):
        self.vectorstore, self.corrective_rag_app = initialize_vectorstore_and_corrective_rag_graph(
            FAISS_INDEX_STORE_PATH=faiss_index_path,
            CORPUS_FILE_PATH=corpus_path,
            embeddings=embeddings,
            llm=llm,
            k=k,
            web_search_tool=web_search_tool or TavilySearchResults(k=3)
        )
        self.config = config or {}
        self.last_result = None

    def retrieve(self, query: str) -> List[str]:
        """Retrieve and generate answer using CRAG workflow."""
        initial_state = {
            "question": query,
            "generation": "",
            "web_search": "No",
            "documents": []
        }
        
        result = None
        for output in self.corrective_rag_app.stream(initial_state, config=self.config):
            for key, value in output.items():
                result = value
        
        self.last_result = result
        return [result.get("generation", "No answer generated")]

    def name(self) -> str:
        return "Corrective RAG (CRAG)"

    def set_config(self, config: RunnableConfig):
        self.config = config

    def get_retrieved_documents(self) -> List[Document]:
        """Get documents from last result."""
        if self.last_result and "documents" in self.last_result:
            return self.last_result["documents"]
        return []

    def invoke(self, input: Dict[str, Any], config: Optional[RunnableConfig] = None, **kwargs) -> Dict[str, Any]:
        """Invoke CRAG workflow with structured input/output."""
        config = config or self.config
        initial_state = {
            "question": input.get("question", ""),
            "generation": "",
            "web_search": "No",
            "documents": []
        }
        
        result = None
        for output in self.corrective_rag_app.stream(initial_state, config=config):
            for key, value in output.items():
                result = value
        
        self.last_result = result
        return {
            "answer": result.get("generation", "No answer generated"),
            "documents": result.get("documents", []),
            "web_search_used": result.get("web_search", "No") == "Yes"
        }

