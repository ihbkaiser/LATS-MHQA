from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
sub_answer_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="Paragraphs"),
    MessagesPlaceholder(variable_name="Evidence"),
    MessagesPlaceholder(variable_name="Question"),
    MessagesPlaceholder(variable_name="Sub_question"),
    HumanMessage(content=(
        "Answer the Sub-question using the provided Paragraphs and Evidence.\n\n"
        "Instructions:\n"
        "- Provide the sentence from 'Paragraphs' that directly answers the Sub-question. If no sentence answers it, return 'No information'.\n"
        "- Output the result for the Sub-question.\n"
        "- The output must be a JSON with two keys: 'Sentence' and 'Sub-question Result', both values must be strings.\n\n"
        "- Do not include any explanation or additional text in the output.\n\n"
        "Output format (JSON):\n"
        "{\n"
        "  \"Sentence\": sentence or 'No information',\n"
        "  \"Sub-question Result\": answer here\n"
        "}"
    ))
])
