import logging
from src.chatbot import ChatBot

logger = logging.getLogger(__name__)

bot = ChatBot()

if __name__ == '__main__':
    logger.info('MODO CHAT')
    bot.chat('Qual a capital da frança?')

    logger.info('MODO RAG')
    bot.load_documents("C:\\Users\\wsant\\Downloads\\O'Reilly - PT - SQL Guia Prático - Alice Zhao.pdf")
    bot.chat('Como usar o comando "select"?')