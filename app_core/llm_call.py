from langchain_google_genai import ChatGoogleGenerativeAI
from .prompt import QA_PROMPT_DEF
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
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)
        # Create retriever
        self.retriever = Retriever(RetrieverConfig())
        # Question
        self._make_chain()
        
    def _make_chain(self):
        self.chain = ConversationalRetrievalChain.from_llm(llm = self.llm, 
                                                        retriever = self.retriever.retriever,
                                                        return_source_documents=True,
                                                        memory = self.memory,
                                                        verbose=False,   # <-- True : prints the prompt and internal steps
                                                        combine_docs_chain_kwargs = {
                                                            'prompt': QA_PROMPT_DEF
                                                        }
                                                        )
        
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
    
    def retriever_close(self):
        try:
            self.retriever.close()   
        except Exception:
            pass

    
    def modify_retriever_cfg(self, cfg):

        for i in cfg:
            if i in vars(self.retriever.cfg):
                vars(self.retriever.cfg)[i] = cfg[i]
                
    def update_retriever_settings(self, cfg):
       
        # cierra conexiones/clients si procede
        self.retriever_close()   
        # 
        self.modify_retriever_cfg(cfg)  

        #Update the retriever confg with the new confg
        self.retriever = Retriever(self.retriever.cfg)

        # opción A: solo reasignar
        if hasattr(self.chain, "retriever"):
            self.chain.retriever = self.retriever.retriever

        else:
            # opción B: reconstruir chain completa
            self.chain = self._make_chain()
            
    


        
