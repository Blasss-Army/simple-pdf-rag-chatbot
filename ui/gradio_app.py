import gradio as gr
import os
from .style import status_box, metric_card

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


        count =  chat_state.retriever.vector_count()
        vector_card_html = metric_card("Vector store ", str(count), "points in collection")
        return answer['answer'], sources , chat_state , vector_card_html            




    def clear_memory(chat_state):
        if chat_state is None:
            return 'There isnt memory yet'
        if (len(chat_state.memory.chat_memory.messages) == 0):
            return 'Memory is already empty'
        return chat_state.clear_memory()
    
    # ----------------------- APPLY THE CONF CHANGES TO THE RETRIEVER ------------------------

        
    def pack_and_apply(chat_state, temperature, reset_collection):

        cfg = {
            'temperature': temperature,
            'reset_collection':reset_collection
        }

        try:
            if hasattr(chat_state, "update_retriever_settings"):
                chat_state.update_retriever_settings(cfg)
            
            else:
                chat_state = Chat()
                chat_state.update_retriever_settings(cfg)
        
        except Exception as e:
           
             return status_box(f"Error applying settings: {e}", "error"), chat_state, ""
        
      

        count = chat_state.retriever.vector_count()
        ok_html = status_box("Changes have been applied", "success")
        metric_html = metric_card("Vector store", count, "points in collection")
        return chat_state , ok_html, metric_html 


    CSS = """
        .gradio-container { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, Helvetica Neue, Arial; }
        """       

    with gr.Blocks(CSS) as demo:
        gr.Markdown(
            """
            # 📘 RAG PDF Chatbot
            **Ask questions about your DOCS (PDFS) , and obtain certain answers with the source**
            """
        )
        # State container -> The sesion state will be storaged here, as well as shared between Tabs
        state = gr.State(value=None)

        with gr.Tabs():
            # ======== Tab 1: Chatbot ========
            with gr.TabItem("Chat"):
                question = gr.Textbox(label="Input")
                with gr.Row():
                    boton = gr.Button("Send", scale=1)
                    boton_limpiar = gr.Button("Limpiar", scale=1)

                salida = gr.Textbox(label="Answer",
                                    lines=20,                    
                                    max_lines=25,                     
                                    show_copy_button=True)


                fuentes = gr.Markdown() 

               

            # ======== Tab 2: retriever conf ========
            with gr.TabItem("Retriever"):
                gr.Markdown(
                """
                # 📘 Retriever paramethers configuration
                """
                )

                with gr.Row():
                    temperature = gr.Slider(0.0, 1.0, value=0.2, step=0.05, label="temperature")
                    reset_collection = gr.Checkbox(value=False, label="reset collection")

                apply_btn = gr.Button("Apply settings")
                with gr.Row():
                    apply_msg = gr.HTML()
                    vectore_storage_size = gr.HTML()

            apply_btn.click(
                fn=pack_and_apply,
                inputs=[state, temperature, reset_collection],
                outputs=[state, apply_msg , vectore_storage_size],
            )

            boton.click(
                fn=send_question,
                inputs=[question, state],
                outputs=[salida, fuentes, state, vectore_storage_size]
            )

            boton_limpiar.click(
                fn=clear_memory,
                inputs=[state],
                outputs=[salida]
            )
                

    demo.launch()
