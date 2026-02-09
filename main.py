from src.chatbot import RAGChatBot

bot = RAGChatBot()

if __name__ == '__main__':
    print('-'*100)
    print('\n= MODO CHAT =')
    print('-'*100)
    bot.chat('Qual a capital da frança?', use_rag=False)

    print('-'*100)
    print('\n= MODO RAG =')
    print('-'*100)
    bot.load_documents("C:\\Users\\wsant\\Downloads\\O'Reilly - PT - SQL Guia Prático - Alice Zhao.pdf")
    bot.chat('Como usar o comando "select"?')