import gradio as gr
import os

def run_gradio(Chat):

    def get_pdf_url(File):
        space_id = os.getenv("SPACE_ID") or os.getenv("HF_SPACE_ID")
        if space_id:
             return f"https://huggingface.co/spaces/{space_id}/blob/main/data/{File}"
        # En local → crea URL relativa
        return f"./data/{File}"

    def get_pdf_link(answer):

        lines = []
        for i, d in enumerate(answer['source_documents']):
            file_name  = d.metadata.get("source")
            file_name = file_name.replace("/", "\\").split("\\")[-1]
            page = d.metadata.get("page", "¿?")
            url = get_pdf_url(file_name)    
            lines.append(f"[{i}] [{file_name} — p.{page}] {url}\n")
        sources = "\n".join(lines)

        return sources
  

    def send_question(question, chat_state):
        if chat_state is None:
            chat_state = Chat()               

        answer = chat_state.run_llm_call(question)
        sources = get_pdf_link(answer=answer)

        return answer['answer'], sources , chat_state             

    def clear_memory(chat_state):
        if chat_state is None:
            return 'There isnt memory yet'
        if (len(chat_state.memory.chat_memory.messages) == 0):
            return 'Memory is already empty'
        return chat_state.clear_memory()
    

    with gr.Blocks() as demo:
        gr.Markdown(
            """
            # 📘 RAG PDF Chatbot
            **Ask questions about your DOCS (PDFS) , and obtain certain answers with the source**
            """
        )
        question = gr.Textbox(label="Input")
        with gr.Row():
            boton = gr.Button("Send", scale=1)
            boton_limpiar = gr.Button("Limpiar", scale=1)

        salida = gr.Textbox(label="Answer",
                            lines=20,                    
                            max_lines=25,                     
                            show_copy_button=True)

        # State container -> The sesion state will be storaged here
        state = gr.State(value=None)
        fuentes = gr.Markdown() 

        boton.click(
            fn=send_question,
            inputs=[question,state],  
            outputs=[salida,fuentes, state]              
        )

        boton_limpiar.click(
            fn=clear_memory,
            inputs=[state],
            outputs=[salida]
        )

    demo.launch()
