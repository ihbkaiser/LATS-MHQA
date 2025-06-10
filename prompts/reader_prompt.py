from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage

reader_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="Question"),
    MessagesPlaceholder(variable_name="Paragraphs"),
    MessagesPlaceholder(variable_name="Evidence"),
    HumanMessage(content=(
        "Answer 'Question' in words by referring the 'Paragraph' and 'Evidence' "
        "produce the final concise answer in 10 words or less. Do not include any explanation, only output the answer.\n\n"
        "Output format:\n"
        "Answer: Your concise answer here"
    ))
])

