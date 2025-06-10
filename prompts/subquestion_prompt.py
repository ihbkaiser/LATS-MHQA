from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage

sub_question_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="Question"),
    MessagesPlaceholder(variable_name="Question_history"),
    HumanMessage(content=(
        "Given the Question and its history, decompose it into the next 'Sub-questions' "
        "needed to answer the original Question.\n\n"
        "Instructions:\n"
        "- Use keywords from 'Question' and 'Question_history' to form each Sub-question.\n\n"
        "Output format:\n"
        "Sub-question: [Your sub-question here]"
    ))
])
