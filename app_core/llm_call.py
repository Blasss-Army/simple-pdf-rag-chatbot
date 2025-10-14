from typing import Any, Dict, Iterable, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from .prompt import QA_PROMPT_DEF
from dotenv import load_dotenv
from create_retriever.retriever import Retriever
from create_retriever.conf import RetrieverConfig
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


class Chat:
    """
    Orchestrates the RAG pipeline: holds the retriever, LLM and chat memory, and
    exposes a simple `run_llm_call(question)` entrypoint.
    """

    def __init__(self):
        """
        Initialize memory, LLM and retriever.
        Inputs:  None
        Outputs: None
        """

        self.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key="question",   
        output_key="answer"    
        )

        load_dotenv()
        # Configure the LLM
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)

        # Create the retriever with default config
        self.retriever = Retriever(RetrieverConfig())

        # Bild the chain
        self.chain = self._make_chain()
        
    def _make_chain(self) -> ConversationalRetrievalChain:
        """
        Build and return a ConversationalRetrievalChain wired with current LLM,
        retriever and memory.
        Inputs:  None
        Outputs: ConversationalRetrievalChain
        """
        chain = ConversationalRetrievalChain.from_llm(
            llm = self.llm, 
            retriever = self.retriever.retriever,
            return_source_documents=True,
            memory = self.memory,
            verbose=False,   # set True to debug the prompt and internals
            combine_docs_chain_kwargs = {'prompt': QA_PROMPT_DEF}
            )
        return chain
        
    def run_llm_call(self, question: str) -> Dict[str, Any]:
        """
        Run a question through the chain and return the full response (answer,
        source docs, and metadata).
        Inputs:
            question (str): User question.
        Outputs:
            dict: Chain invocation result with keys like 'answer' and 'source_documents'.
        """
        return self.chain.invoke({"question": question})
        
    def clear_memory(self) -> str:
        """
        Clear chat memory.
        Inputs:  None
        Outputs: str message confirming the operation.
        """
        self.memory.clear()
        return 'All memory storaged has been deleted'
    
    def retriever_close(self) -> None:
        """
        Close underlying retriever resources (if any).
        Inputs:  None
        Outputs: None
        """
        try:
            self.retriever.close()   
        except Exception:
            pass

    
    def modify_retriever_cfg(self, cfg: Dict[str,Any]) -> None:
        """
        Update current RetrieverConfig with provided keys (in-place).
        Inputs:
            cfg (dict): Partial config to overlay on existing RetrieverConfig.
        Outputs: None
        """
        for i in cfg:
            if i in vars(self.retriever.cfg):
                vars(self.retriever.cfg)[i] = cfg[i]
                
    def update_retriever_settings(self, cfg: Dict[str, Any]) -> None:
        """
        Re-create the retriever with updated settings and rewire the chain.
        Inputs:
            cfg (dict): Partial RetrieverConfig fields to update.
        Outputs: None
        """
        # close connections/clients if needed
        self.retriever_close()   
        # patch config
        self.modify_retriever_cfg(cfg)  

        # Rebuild retriever
        self.retriever = Retriever(self.retriever.cfg)

       # Option A: reassign on the existing chain if possible
        if hasattr(self.chain, "retriever"):
            self.chain.retriever = self.retriever.retriever

        else:
            # Option B: rebuild the chain
            self.chain = self._make_chain()
            
    


        
