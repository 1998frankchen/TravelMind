import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Dict, List
import re
import graphviz
from pathlib import Path
import tempfile
import os
import uuid

import sys
sys.path.append("../../")  # Add parent directory's parent directory to sys.path
from configs.config import MODEL_CONFIG

model_path = MODEL_CONFIG['model']['name']  


import subprocess
def check_graphviz_installed():  
    """Check if graphviz is installed on the system"""
    try:  
        # Run command 'dot -V', where dot is Graphviz's command line tool, -V is used to print version information
        # capture_output=True means capture command output, check=True means throw exception if command returns non-zero exit status
        subprocess.run(['dot', '-V'], capture_output=True, check=True)  
        return True  
    except (subprocess.SubprocessError, FileNotFoundError):  
        return False



def clean_text(text: str, truncate_length: int = 100) -> str:  
    """  
    Clean text, remove or replace characters that might cause graphviz syntax errors
    """  
    # Remove or replace special characters
    text = re.sub(r'[^\w\s-]', '_', text)  
    # Ensure text is not empty
    text = text.strip() or "node"  
    # Restriction/LimitationTextLength  
    return text[:truncate_length]  

class MindMapGenerator:
    def __init__(
        self, 
        model_name: str = model_path,
        level_num:int=3, 
        item_num:int=15, 
        max_new_tokens:int=1024
        ):
        """
        InitializeMind mapGenerator
        Args:
            model_name: Hugging FaceModelName
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        self.level_num = level_num
        self.item_num = item_num
        self.max_new_tokens = max_new_tokens
        
    def generate_mindmap_content(self, topic: str,) -> str:
        """
        Use large model to generate mind map content
        Args:
            topic: User input topic
        Returns:
            Generated mind map content in hierarchical list format
        """
        prompt = f"""Please create a detailed mind map using the content: "{topic}". 
        The output should be in a hierarchical format with main topics and subtopics.
        Format the output as a list with proper indentation using - for each level.
        Keep it concise but informative. Generate no more than {self.level_num} levels and {self.item_num} total items. 
        
        Example format:
        \n\n
        - Main Topic
          - Subtopic 1
            - Detail 1
            - Detail 2
          - Subtopic 2
            - Detail 3
            - Detail 4
            
        Here is your mindmap:
        """
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):]
        
        print("response:", response)
        # Extract the generated list portion
        # content = response.split("\n\n")[-1]
        # Use regex to extract the last portion containing hierarchical list
        pattern = r'(?:^|\n)(-\s+[^\n]+(?:\n\s+-\s+[^\n]+)*)'  
        matches = re.finditer(pattern, response, re.MULTILINE)  
        content = list(matches)[-1].group(0) if matches else f"- {topic}\n  - Generation failed"  
        return content

    def parse_hierarchy(self, content: str) -> List[tuple]:
        """
        Parse hierarchical list content into node relationships
        Args:
            content: Generated hierarchical list content
        Returns:
            Node relationship list [(parent, child, level)]
        """
        lines = content.strip().split('\n')
        nodes = []
        previous_nodes = [''] * self.item_num  # Store previous node at each level
        
        for line in lines:
            # Calculate indentation level
            indent_level = len(re.match(r'^\s*', line).group()) // 2
            # Extract text content
            text = line.strip().strip('- ')
            
            if indent_level == 0:
                nodes.append(('ROOT', text, indent_level))
            else:
                parent = previous_nodes[indent_level - 1]
                nodes.append((parent, text, indent_level))
            
            if indent_level >= len(previous_nodes):
                tmp_nodes = ['']*2*indent_level
                for idx, node in enumerate(previous_nodes):
                    tmp_nodes[idx] = node
                previous_nodes = tmp_nodes
                
            previous_nodes[indent_level] = text
            
        return nodes

    def create_mindmap(self, topic: str, nodes: List[tuple]) -> str:
        """
        UsagegraphvizCreateMind map
        Args:
            topic: Subject
            nodes: Node relationship list
        Returns:
            Generated image path
        """
        if not check_graphviz_installed():  
            raise RuntimeError(  
                "Graphviz not found. Please install it first:\n"  
                "Ubuntu/Debian: sudo apt-get install graphviz\n"  
                "CentOS: sudo yum install graphviz\n"  
                "MacOS: brew install graphviz\n\n"
                "pip install graphviz"  
            )  
            
        # Create directed graph
        # By creating this directed graph object dot, we can later add nodes, edges and other operations to build specific directed graph content
        dot = graphviz.Digraph(
                comment='MindMap',
                format='jpg',
                engine='dot' # dot is one of the engines in graphviz used for layout and rendering graphs
            )
        dot.attr(rankdir='LR')  # Left to right layout
        # Set/ConfigureGraphAttribute/Property
        dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue')
        
        # Add root node (subject)
        root_id = 'root'
        clean_topic = clean_text(topic)
        dot.node(root_id, clean_topic, fillcolor='lightblue')
        
        # Used to store created node IDs
        created_nodes = {root_id} 
        
        # Add all other nodes and edges
        for parent, child, level in nodes:
            # Generate unique ID for each node
            node_id = f"node_{uuid.uuid4().hex[:8]}"  
            
            # TeardownNodeText  
            clean_child = clean_text(child) 
            
            # Set different colors based on hierarchy level
            colors = ['lightblue', 'lightgreen', 'lightyellow']  
            color = colors[min(level, len(colors)-1)]  
            # AddNode  
            dot.node(node_id, clean_child, fillcolor=color)  
            
            # if parent == 'ROOT':
            #     parent = topic
            
            # # Set different colors based on hierarchy level
            # colors = ['lightblue', 'lightgreen', 'lightyellow']
            # color = colors[min(level, len(colors)-1)]
            
            # # Add nodes and edges
            # dot.node(child, child, fillcolor=color)
            # dot.edge(parent, child)
            
            
            if level==0:
                dot.edge(root_id, node_id)
            else:
                # Find parent node ID
                parent_nodes = [n for n in created_nodes if clean_text(parent) in dot.body]  
                if parent_nodes:  
                    dot.edge(parent_nodes[-1], node_id)  
            created_nodes.add(node_id)  
            
            
        # Create temporary folder to save image
        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, 'mindmap.png')
        
        # Render and save image
        dot.render(os.path.join(temp_dir, 'mindmap'), format='png', cleanup=True)
        
        return output_path

def generate_mindmap(topic: str) -> str:
    """
    GradioInterfaceFunction
    Args:
        topic: User input subject
    Returns:
        Generated mind map image path
    """
    generator = MindMapGenerator(max_new_tokens=1024)
    content = generator.generate_mindmap_content(topic)
    nodes = generator.parse_hierarchy(content)
    image_path = generator.create_mindmap(topic, nodes)
    
    return image_path

# CreateGradioInterface
with gr.Blocks() as demo:
    gr.Markdown("""
    # AIMind mapGenerator
    Input a subject, and AI will generate a corresponding mind map for you.
    """)
    
    with gr.Row():
        topic_input = gr.Textbox(label="Input Subject", placeholder="For example: Artificial Intelligence, Machine Learning, Python Programming...")
        generate_btn = gr.Button("GenerateMind map")
    
    with gr.Row():
        # Use Image component to display generated mind map
        mindmap_output = gr.Image(label="Generated Mind Map")
    
    generate_btn.click(
        fn=generate_mindmap,
        inputs=[topic_input],
        outputs=[mindmap_output]
    )

# if __name__ == "__main__":
#     demo.launch()