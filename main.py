import argparse
import json
import math
import os
import re
from typing import Dict, List, Optional, Tuple, Union

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig

from chat_models.gemini_chat_model import GeminiChatModel
from chat_models.openai_chat_model import OpenAIChatModel
from searchers.base_searcher import BaseSearcher
from langchain.embeddings.openai import OpenAIEmbeddings
from searchers.tavily_search import TavilySearcher
from searchers.self_rag import SelfRAGSearcher
from searchers.corrective_rag import CorrectiveRAGSearcher

from prompts import (
    action_prompt,
    reflection_prompt,
    reader_prompt,
    sub_answer_prompt,
    sub_question_prompt,
)

load_dotenv()

PREFIX_PATTERN = re.compile(
    r'^(?:\d+:\s*|Sub-question\s*:?\s*)+', re.IGNORECASE
)


def clean_and_parse(
    response: Union[str, Dict]
) -> Dict:
    """
    Parse a JSON-like response string or dict into a dict.
    """
    if isinstance(response, dict):
        return response

    if not isinstance(response, str):
        return {}

    text = response.strip()
    if not text:
        return {}

    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]

    text = "\n".join(lines).strip()

    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def strip_prefix(text: str) -> str:
    """Remove numeric and 'Sub-question' prefixes."""
    return PREFIX_PATTERN.sub('', text).strip()


def format_history_and_evidence(
    trajectory: List[HumanMessage]
) -> Tuple[List[HumanMessage], List[HumanMessage]]:
    """
    Build question history and evidence lists from the message trajectory.
    """
    history: List[HumanMessage] = []
    evidence: List[HumanMessage] = []
    steps = trajectory[1:]

    for i in range(0, len(steps), 2):
        raw_q = steps[i].content
        raw_a = steps[i + 1].content if i + 1 < len(steps) else ''
        subq = strip_prefix(raw_q)
        suba = strip_prefix(raw_a)

        history.append(HumanMessage(content=subq))
        history.append(HumanMessage(content=suba))
        evidence.append(HumanMessage(content=f"{subq} {suba}"))

    return history, evidence


class Reflection(BaseModel):
    reflections: str = Field(
        description="Reflection on the quality of the response"
    )
    score: int = Field(description="Score from 0 to 10", ge=0, le=10)
    found_solution: bool = Field(
        description="Does the response fully answer the question?"
    )

    def as_message(self) -> HumanMessage:
        return HumanMessage(
            content=f"Reflection: {self.reflections}\nScore: {self.score}"
        )

    @property
    def normalized_score(self) -> float:
        return self.score / 10.0


class SubAnswerSchema(BaseModel):
    sub_question: str = Field(..., alias="Sub-question")
    sub_question_result: str = Field(
        ..., alias="Sub-question Result"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LATS_MHQA main entrypoint"
    )
    parser.add_argument(
        "--model",
        choices=["openai", "gemini"],
        default="openai",
        help="Chat model to use",
    )
    parser.add_argument(
        "--question", required=True, help="Question to answer"
    )
    parser.add_argument(
        "--searcher",
        choices=["tavily", "self-rag", "corrective-rag"],
        default="tavily",
        help="Searcher to use",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging"
    )

    return parser.parse_args()


class Node:
    def __init__(
        self,
        messages: List[BaseMessage],
        reflection: Reflection,
        parent: Optional['Node'] = None,
    ):
        self.messages = messages
        self.parent = parent
        self.children: List['Node'] = []
        self.value = 0.0
        self.visits = 0
        self.reflection = reflection
        self.depth = parent.depth + 1 if parent else 1
        self.sub_answer: Optional[HumanMessage] = None
        self.simulation_retrieved_documents: List[Document] = []

    def upper_confidence_bound(
        self, exploration_weight: float = 1.0
    ) -> float:
        if self.visits == 0:
            return float('inf')

        parent_visits = (
            self.parent.visits if self.parent else self.visits
        )
        exploitation = self.value / self.visits
        exploration = (
            exploration_weight
            * math.sqrt(math.log(parent_visits) / self.visits)
        )
        return exploitation + exploration

    def backpropagate(self, reward: float, verbose: bool) -> None:
        self.visits += 1
        self.value += reward

        if verbose:
            print(
                f"[Backpropagate] depth={self.depth},"
                f" visits={self.visits}, value={self.value:.2f}"
            )

        if self.parent:
            self.parent.backpropagate(reward, verbose)

    def get_trajectory(self) -> List[BaseMessage]:
        if self.parent:
            traj = self.parent.get_trajectory() + self.messages
            if self.sub_answer:
                traj.append(self.sub_answer)
            return traj

        traj = self.messages.copy()
        if self.sub_answer:
            traj.append(self.sub_answer)
        return traj

    def get_best_solution(self) -> 'Node':
        if not self.children:
            return self

        best = max(
            self.children,
            key=lambda c: (c.value / c.visits)
            if c.visits else float('inf'),
        )
        return best.get_best_solution()


def select(
    node: Node,
    exploration_weight: float,
    verbose: bool,
) -> Node:
    current = node
    if verbose:
        print(f"[Select] start depth={current.depth}")

    while current.children:
        unvisited = [c for c in current.children if c.visits == 0]
        if unvisited:
            if verbose:
                print(f"[Select] unvisited at depth={unvisited[0].depth}")
            return unvisited[0]

        current = max(
            current.children,
            key=lambda c: c.upper_confidence_bound(exploration_weight),
        )
        if verbose:
            print(
                f"[Select] descend depth={current.depth},"
                f" UCT={current.upper_confidence_bound(exploration_weight):.2f}"
            )

    return current


def expand(
    node: Node,
    config: RunnableConfig,
    verbose: bool,
) -> None:
    traj = node.get_trajectory()
    question_msg = traj[0]
    history_msgs, _ = format_history_and_evidence(traj)

    if verbose:
        print(
            f"[Expand] depth={node.depth},"
            f" question='{strip_prefix(question_msg.content)}'"
        )

    resp = sub_question_chain.invoke(
        {"Question": [HumanMessage(content=QUESTION_TEXT)],
         "Question_history": history_msgs},
        config,
    )
    if verbose:
        print(f"[Expand] response:\n{resp.content}")

    for line in resp.content.splitlines():
        sq = strip_prefix(line)
        if sq:
            if verbose:
                print(f"[Expand] add sub-question '{sq}'")
            child = Node(
                [HumanMessage(content=sq)],
                Reflection(reflections="init", score=0, found_solution=False),
                parent=node,
            )
            node.children.append(child)


def simulate_and_backprop(
    node: Node,
    config: RunnableConfig,
    verbose: bool,
) -> None:
    current = node
    question_msg = current.get_trajectory()[0]

    for step in range(MAX_HOP):
        subq = strip_prefix(current.messages[0].content)
        if verbose:
            print(f"[Simulate] step={step+1}, subq='{subq}'")

        paras = searcher.retrieve(subq)
        if verbose:
            print(f"[Simulate] got {len(paras)} paras")

        docs = [Document(page_content=p) for p in paras]
        current.simulation_retrieved_documents = docs

        history_msgs, evidence_msgs = format_history_and_evidence(
            current.get_trajectory()
        )
        if verbose:
            print("[Simulate] evidence:")
            for ev in evidence_msgs:
                print(f"  - {ev.content}")

        resp = sub_answer_chain.invoke(
            {
                "Paragraphs": [HumanMessage(content=d.page_content)
                               for d in docs],
                "Evidence": evidence_msgs,
                "Question": [question_msg],
                "Sub_question": [HumanMessage(content=subq)],
            },
            config,
        )
        if verbose:
            print(f"[Simulate] answer:\n{resp.content}")

        parsed = clean_and_parse(resp.content)
        if not parsed or "Sub-question Result" not in parsed:
            if verbose:
                print("[Simulate] no valid answer, break")
            break

        sub_ans = strip_prefix(parsed["Sub-question Result"])  # type: ignore
        current.sub_answer = HumanMessage(content=sub_ans)
        if verbose:
            print(f"[Simulate] sub-answer='{sub_ans}'")

        next_resp = action_chain.invoke(
            {"Question": [HumanMessage(content=QUESTION_TEXT)],
             "Question_history": history_msgs},
            config,
        )
        next_q = strip_prefix(next_resp.content)
        if verbose:
            print(f"[Simulate] next subq='{next_q}'")
        if not next_q or next_q.lower() == "none":
            break

        child = Node(
            [HumanMessage(content=next_q)],
            Reflection(reflections="init", score=0, found_solution=False),
            parent=current,
        )
        current.children.append(child)
        current = child

    refl_resp = reflection_llm_chain.invoke(
        {"Question": [question_msg],
         "Question_history": history_msgs},
        config,
    )
    refl_obj = (
        refl_resp[0]
        if isinstance(refl_resp, list) and refl_resp
        else Reflection(reflections="no answers", score=0, found_solution=False)
    )
    if verbose:
        print(f"[Reflect] score={refl_obj.score}")
    current.backpropagate(refl_obj.normalized_score, verbose)


def search(
    root: Node,
    config: RunnableConfig,
    budget: int = 30,
    w: float = 1.0,
    verbose: bool = False,
) -> Node:
    for i in range(budget):
        if verbose:
            print(f"[Search] iter={i+1}/{budget}")
        leaf = select(root, w, verbose)
        if leaf.depth <= MAX_HOP:
            expand(leaf, config, verbose)
        for child in leaf.children:
            print(f"[Search] child score={child.reflection.score}")
            if child.reflection.found_solution and child.reflection.score >= 8:
                if verbose:
                    print(f"[Early Stop] Found good solution at iter {i+1}, score={child.reflection.score}")
                break
            if child.visits == 0:
                simulate_and_backprop(child, config, verbose)

    best = root.get_best_solution()
    if verbose:
        print(
            f"[Search] best depth={best.depth},"
            f" value={best.value:.2f}, visits={best.visits}"
        )
    return best

def initial_answer(searcher: BaseSearcher, question: str, config: RunnableConfig, verbose: bool = False) -> tuple[str, Reflection]:
    """Generate initial answer using searcher and evaluate its quality."""
    if verbose:
        print("🔍 [Initial Answer] Generating initial answer...")
    
    # Retrieve documents
    paras = searcher.retrieve(question)
    docs = [Document(page_content=p) for p in paras]
    
    if verbose:
        print(f"[Initial Answer] Retrieved {len(docs)} documents")
    
    # For initial answer, we have no history yet
    history_msgs = []
    evidence_msgs = [HumanMessage(content="No prior evidence available.")]
    
    # Generate answer using reader chain
    resp = reader_chain.invoke(
        {
            "Question": [HumanMessage(content=question)],
            "Paragraphs": [HumanMessage(content=d.page_content) for d in docs],
            "Evidence": evidence_msgs,
            "Question_history": history_msgs
        },
        config,
    )
    
    initial_answer_text = strip_prefix(resp.content)
    if verbose:
        print(f"[Initial Answer] Generated: {initial_answer_text[:100]}...")
    
    # Evaluate the answer quality using reflection
    # Create a simple question history with the answer
    question_history = [HumanMessage(content=f"Answer: {initial_answer_text}")]
    
    refl_resp = reflection_llm_chain.invoke(
        {
            "Question": [HumanMessage(content=question)],
            "Question_history": question_history
        },
        config,
    )
    
    refl_obj = (
        refl_resp[0]
        if isinstance(refl_resp, list) and refl_resp
        else Reflection(reflections="no evaluation", score=0, found_solution=False)
    )
    
    if verbose:
        print(f"[Initial Answer] Score: {refl_obj.score}/10, Found Solution: {refl_obj.found_solution}")
    
    return initial_answer_text, refl_obj

if __name__ == '__main__':
    args = parse_args()
    QUESTION_TEXT = args.question  # type: ignore
    VERBOSE = args.verbose  # type: ignore

    if args.model == 'openai':  # type: ignore
        chat_model = OpenAIChatModel(model_name='gpt-4o', temperature=0.0)  # type: ignore
    else:
        chat_model = GeminiChatModel(
            model_name='gemini-2.0-flash', temperature=0.0
        )  # type: ignore

    # Initialize embeddings
    embeddings = OpenAIEmbeddings()

    if args.searcher == 'tavily':  # type: ignore
        searcher: BaseSearcher = TavilySearcher(max_results=5)  # type: ignore
    elif args.searcher == 'self-rag':  # type: ignore
        searcher: BaseSearcher = SelfRAGSearcher(corpus_path="data/multihoprag_corpus.txt")
    elif args.searcher == 'corrective-rag':  # type: ignore
        searcher: BaseSearcher = CorrectiveRAGSearcher(
            faiss_index_path="data/faiss_index",
            corpus_path="data/multihoprag_corpus.txt",
            llm=chat_model,
            embeddings=embeddings,
        )

    reflection_llm_chain = (
        reflection_prompt
        | chat_model.bind_tools(tools=[Reflection], tool_choice='Reflection')
        | PydanticToolsParser(tools=[Reflection])
    )
    sub_question_chain = sub_question_prompt | chat_model
    sub_answer_chain = sub_answer_prompt | chat_model
    action_chain = action_prompt | chat_model
    reader_chain = reader_prompt | chat_model
    MAX_HOP = 5

    cfg = RunnableConfig()
    
    # Try initial answer first
    print("Trying initial answer...")
    initial_ans, initial_reflection = initial_answer(searcher, QUESTION_TEXT, cfg, VERBOSE)
    
    # Check if initial answer is good enough (score >= 7 or found_solution=True with score >= 5)
    threshold_met = (initial_reflection.score >= 7) or (initial_reflection.found_solution and initial_reflection.score >= 5)
    
    if threshold_met:
        print("Initial answer is good enough! Skipping MCTS...")
        print(f"Score: {initial_reflection.score}/10, Found Solution: {initial_reflection.found_solution}")
        print('===== Final Answer (Initial) =====')
        print(initial_ans)
    else:
        print(f"Initial answer not sufficient (Score: {initial_reflection.score}/10). Running MCTS...")
        
        root = Node(
            [HumanMessage(content=QUESTION_TEXT)],
            Reflection(reflections='init', score=0, found_solution=False),
        )
        best_node = search(root, cfg, budget=2, w=1.0, verbose=VERBOSE)  # type: ignore

        print('===== Best Trajectory =====')
        for msg in best_node.get_trajectory():
            print(f'- {strip_prefix(msg.content)}')

        print('===== Final Answer (MCTS) =====')
        final_msgs = best_node.simulation_retrieved_documents
        history_msgs, evidence_msgs = format_history_and_evidence(
            best_node.get_trajectory()
        )
        final = reader_chain.invoke(
            {
                'Question': [HumanMessage(content=QUESTION_TEXT)],
                'Paragraphs': [HumanMessage(content=d.page_content) for d in final_msgs],
                'Evidence': evidence_msgs,
                'Question_history': history_msgs,
            },
            RunnableConfig(),
        )
        print(strip_prefix(final.content))
