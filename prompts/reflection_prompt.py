from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage

reflection_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="Question"),
    MessagesPlaceholder(variable_name="Question_history"),
    HumanMessage(content=(
        "You are given a Question and a Question history. "
        "Assign a score from 0 to 10 to indicate how helpful 'Question history' is in solving the 'Question'.\n\n"
        "Instructions:\n"
        "- Only assign a score of 10 if the 'Question history' can be directly used to answer the 'Question'.\n"
        "- Assign a score of 0 if there is no relevant information in the 'Question history'.\n"
        "- Set 'found_solution' to True if the Question history contains sufficient information to answer the Question completely.\n\n"
        "Output format:\n"
        "Let's think step by step: [Your reasoning here]\n"
        "Score: [0–10]\n"
        "Found Solution: [True/False]\n"
    ))
])
