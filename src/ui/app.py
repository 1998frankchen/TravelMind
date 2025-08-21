import gradio as gr
import torch
from typing import Dict, Tuple, List
from ..models.model import TravelMind
from .mindmap import generate_mindmap
import re

'''
Example: How do I travel from Shanghai to Paris?
'''

class TravelMindUI:
    def __init__(self, agent:TravelMind):
        self.agent = agent
        self.chat_history = []
        
        # Pre-set example questions  
        self.example_prompts = [  
            "Recommend three cities suitable for travel in December",
            "Help me plan a 3-day Beijing travel itinerary",
            "I want to go to the beach for vacation with a budget of $1200, any suggestions?",
            "Recommend some destinations suitable for traveling with parents",
            "Help me list items to prepare for traveling to Japan"  
        ]  
    def set_example_text(self, example: str) -> str:
        """Set example text to input box"""  
        return example  
        
    def _format_chat_history(self) -> str:
        """Format chat history"""
        formatted = ""  
        for msg in self.chat_history:  
            if msg["role"] == "user":  
                formatted += f"User: {msg['content']}\n"  
            elif msg["role"] == "assistant":  
                formatted += f"Assistant: {msg['content']}\n\n"  
        
        if formatted == "":  
            formatted = "System: You are TravelMind that can help user plan a route from one start location to a end location. This plan you give should be in detail.\n\n"  
        
        return formatted + "User: "  
    
    def merge_history_into_mindmap(self) -> str:
        """Merge chat history into mind map"""
        content = self._format_chat_history()
        return re.sub(r"User:\s*$", "", content)
    
    def generate_mindmap_using_chatbot(self) -> str:
        """Generate mind map"""
        content = self.merge_history_into_mindmap()
        return generate_mindmap(content)
        
    def respond(
        self,
        message: str,
        history: List[Tuple[str, str]],
        temperature: float,
        top_p: float
    ) -> Tuple[str, List[Tuple[str, str]]]:
        """Handle user input and generate response"""
        # Build prompt
        formatted_chat_history = self._format_chat_history()
        prompt = f"{formatted_chat_history}{message}\nAssistant:"
        
        # Generate response
        response = self.agent.generate_response(
            prompt=prompt,
            max_length=1024,
            temperature=temperature,
            top_p=top_p
        )
        
        self.chat_history.append({"role": "user", "content": message})  
        self.chat_history.append({"role": "assistant", "content": response})  
        
        # return response, self.chat_history
        return self.chat_history
    
    def create_interface(self):
        """Create Gradio Interface"""
        with gr.Blocks(css="footer {display: none !important}") as interface:
            gr.Markdown("# 🌍 TravelMind - Transforming Travel Through Intelligent Reasoning")
            
            with gr.Row():
                with gr.Column(scale=4):
                    chatbot = gr.Chatbot(
                        [],
                        type="messages",
                        elem_id="chatbot",
                        height=500
                    )
                    
                with gr.Column(scale=1):
                    with gr.Accordion("Settings", open=False):
                        temperature = gr.Slider(
                            minimum=0.1,
                            maximum=2.0,
                            value=0.7,
                            step=0.1,
                            label="Temperature"
                        )
                        top_p = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=0.9,
                            step=0.1,
                            label="Top P"
                        )
            
            with gr.Row():
                message = gr.Textbox(
                    show_label=False,
                    placeholder="Enter your travel-related questions...",
                    container=True,
                    min_width=300,  # Fixed width
                    lines=5,  # Set number of lines
                )
                
                submit = gr.Button("Send")
                
                mindmap_button = gr.Button("Generate Mind Map")
                
                
            # Add example tooltip buttons
            with gr.Row():  
                example_buttons = []  
                for example in self.example_prompts:  
                    btn = gr.Button(example, size="sm")  
                    example_buttons.append(btn)  
                    # Bind click event to input box
                    btn.click(  
                        fn=self.set_example_text,  
                        inputs=[btn],  
                        outputs=[message]  
                    )  
            
            # Add instruction/description text
            gr.Markdown("""  
            ### 💡 Usage Tips:
            - Click the buttons above to quickly select FAQ
            - You can also directly input custom questions
            - You can adjust reply diversity (Temperature) and quality (Top P) in settings
            """)  
            
            with gr.Row():
                mindmap_output = gr.Image(
                    label="Generated Mind Map",
                    show_label=False,
                    height=500,
                )
            
            # Bind events
            tmp=None
            
            submit_click = submit.click(
                self.respond,
                inputs=[message, chatbot, temperature, top_p],
                outputs=[chatbot]
            )
            
            # message.submit(
            #     self.respond,
            #     inputs=[message, chatbot, temperature, top_p],
            #     outputs=[message, chatbot]
            # )
            
            mindmap_button.click(
                self.generate_mindmap_using_chatbot,
                inputs=[],
                outputs=[mindmap_output]
            )
            
        return interface

# Create and launch interface
def launch_ui(agent):
    ui = TravelMindUI(agent)
    interface = ui.create_interface()
    interface.launch(share=True)