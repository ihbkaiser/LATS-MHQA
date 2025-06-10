from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage

action_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="Question"),
    MessagesPlaceholder(variable_name="Question_history"),
    HumanMessage(content=(
        "Based on the Question and its history, generate one 'Sub-question' to continue solving the Question.\n\n"
        "Instructions:\n"
        "- Create 'Sub-question' with keywords from 'Question' and 'Question_history'.\n\n"
        "Output format:\n"
        "Sub-question: [Your sub-question here]"
    ))
])
