from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from google import genai
import os
from create_retriever.retriver import Retriever
from create_retriever.conf import RetrieverConfig
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


class Chat:
    def __init__(self):
        '''
        Used for sending the retrieved chunks from the documents as context for the LLM,
        besides ,it provides the respond from the LLM.
        '''
        self.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key="question",   
        output_key="answer"    
        )

        load_dotenv()
        # Obtain the key from the .env
        api_key = os.getenv("GOOGLE_API_KEY")
        # Initialaze Gemini client
        client = genai.Client(api_key=api_key)
        # Configure the LLM
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)
        # Create retriever
        self.retriever = Retriever(RetrieverConfig(reset_collection=False))
        # Question
        self.chain = ConversationalRetrievalChain.from_llm(llm = llm, 
                                                        retriever = self.retriever.retriever,
                                                        return_source_documents=True,
                                                        memory = self.memory,)
        
    def run_llm_call(self, question):
        '''
        Get a respond from the LLM along all metadata, document_sources and chat history.
            Input: question
            Output: Answer
        '''
        return self.chain.invoke({"question": question})
        

    def clear_memory(self):
        '''Clears the chat memory'''
        self.memory.clear()

        return 'All memory storaged has been deleted'