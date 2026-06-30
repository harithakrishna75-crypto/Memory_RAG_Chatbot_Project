from backend import initialize_chatbot, ask_question

#Give the path of your PDF here

llm, memory, retriever = initialize_chatbot(
    "PDFs/NCHS Data Brief.pdf"             
)

while True:

    question = input("\nAsk a question: ")

    if question.lower() == "exit":
        break

    response = ask_question(
        question,
        retriever,
        memory,
        llm
    )

    print("\nAnswer:")
    print(response)
    