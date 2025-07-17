import gradio as gr  
from src.agents.agent import RAG  
from src.agents.tools import ToolDispatcher  
from src.models.model import TravelMind  
from src.configs.config import RAG_DATA_PATH, SFT_MODEL_PATH  

import pandas as pd  
import matplotlib.pyplot as plt  
from typing import Dict 

from src.finetune.sft_trainer import SFTTrainer



class TrainingMonitor:  
    """Real-time training monitor"""  
    def __init__(self):  
        self.loss_history = []  
        self.metric_history = []  
        self.current_step = 0  

    def update(self, logs: Dict):  
        if "loss" in logs:  
            self.loss_history.append((self.current_step, logs["loss"]))  
        if "metrics" in logs:  
            self.metric_history.append((self.current_step, logs["metrics"]))  
        self.current_step += 1  

    def get_loss_plot(self):  
        df = pd.DataFrame(self.loss_history, columns=["step", "loss"])  
        return gr.LinePlot(  
            df,  
            x="step",  
            y="loss",  
            title="Training Loss Curve",  
            width=400,  
            height=300  
        )  

    def get_latest_metrics(self):  
        if not self.metric_history:  
            return "Waiting for first evaluation..."  
        return pd.DataFrame(  
            self.metric_history[-1][1],  
            index=["LatestMetric"]  
        )  



# Initialize RAG system  
def initialize_rag():  
    agent = TravelMind(model_name=SFT_MODEL_PATH)  
    rag = RAG(  
        agent=agent,  
        dataset_name_or_path=RAG_DATA_PATH,  
        use_db=True,  
        use_prompt_template=True  
    )  
    return rag  


# CreateGradioInterfaceComponent  
def create_demo(rag:RAG):  
    
    
    monitor = TrainingMonitor() 
    
    with gr.Blocks(title="Travel RAG Assistant", theme=gr.themes.Soft()) as demo:  
        gr.Markdown("# 🌍 Smart Travel Planning Assistant")
        
        
        # Chat interface
        with gr.Row():  
            with gr.Column(scale=2):  
                chatbot = gr.Chatbot(height=450, label="Conversation History")
                query_box = gr.Textbox(  
                    placeholder="Enter your travel questions...",
                    label="UserInput",  
                    lines=3  
                )   
                
                # ExampleQuestion/Problem  
                examples = gr.Examples(  
                    examples=[  
                        ["Help me plan a 3-day trip to Shanghai with a budget of 5000 yuan"],
                        ["Find the cheapest flight tickets from Beijing to Paris for next week"],
                        ["Recommend 4-star hotels near West Lake in Hangzhou"],
                        ["Query the weather forecast for Tokyo next week"]
                    ],  
                    inputs=query_box,  
                    label="ExampleQuestion/Problem"  
                )  
                
            # Result display area
            with gr.Column(scale=1):  
                with gr.Tab("Tool Call Results"):
                    tool_output = gr.JSON(label="ToolExecuteDetails")  
                with gr.Tab("Database Match Results"):  
                    db_output = gr.DataFrame(  
                        headers=["Related Results"],
                        datatype=["str"],  
                        # max_rows=5,  
                        # overflow_row_behaviour="show_ends"  
                    )  
                with gr.Tab("Raw Response"):
                    raw_output = gr.Textbox(  
                        lines=8,  
                        max_lines=12,  
                        label="Complete/IntactResponse"  
                    )  
            
            # Fine-tuning control panel
            with gr.Column(scale=1):  
                with gr.Tab("Model Fine-tuning"):  
                    with gr.Accordion("Training Parameter Configuration", open=True):  
                        learning_rate = gr.Slider(  
                            minimum=1e-6,  
                            maximum=1e-3,  
                            value=2e-4,  
                            step=1e-6,  
                            label="Learning rate"  
                        )  
                        num_epochs = gr.Slider(  
                            minimum=1,  
                            maximum=10,  
                            value=3,  
                            step=1,  
                            label="Training Epochs"  
                        )  
                        batch_size = gr.Slider(  
                            minimum=1,  
                            maximum=32,  
                            value=4,  
                            step=1,  
                            label="Batch Size"
                        )  

            
            
                    
        # ControlButton  
        with gr.Row():  
            submit_btn = gr.Button("Submit", variant="primary")
            clear_btn = gr.Button("Clear Conversation")
        
        with gr.Row():
            with gr.Accordion("LanguageSet/Configure", open=False):  
                lang = gr.Dropdown(  
                    choices=["Chinese", "English", "Japanese"],
                    value="Chinese",
                    label="InterfaceLanguage"  
                )  
            
        # Process logic
        def respond(query, chat_history):  
            # GenerateResponse  
            prompt = rag.prompt_template.generate_prompt(  
                query,   
                "\n".join([f"User:{u}\nSystem:{s}" for u, s in chat_history])  
            )  
            
            # Get model response  
            tool_call_str = rag.agent.generate_response(prompt)  
            
            # Execute tool calls
            tool_result = rag.dispatcher.execute(tool_call_str)  
            
            # Database query  
            db_result = rag.query_db(query)  
            
            # GenerateNatureLanguageResponse  
            response = f"{tool_call_str}\n\nTool Results: {tool_result}\nDatabase Match: {db_result[:2]}"  
            
            # Get final travel plan
            travel_plan = rag.get_travel_plan(response)
            
            # Update conversation history
            chat_history.append((query, travel_plan))  
            
            return {  
                chatbot: gr.update(value=chat_history),  
                tool_output: tool_result,  
                db_output: [[res] for res in db_result],  
                raw_output: response,  
                query_box: ""  
            }  
            
        # BindingEvent  
        submit_btn.click(  
            fn=respond,  
            inputs=[query_box, chatbot],  
            outputs=[chatbot, tool_output, db_output, raw_output, query_box]  
        )  
        
        clear_btn.click(  
            fn=lambda: ([], [], [], ""),  
            outputs=[chatbot, tool_output, db_output, query_box]  
        )  

    return demo 




if __name__ == "__main__":  
    rag_system = initialize_rag()  
    demo = create_demo(rag_system)  
    demo.launch(  
        server_name="0.0.0.0",  
        server_port=7860,  
        share=False,  
        favicon_path="./travel_icon.png"  
    )  