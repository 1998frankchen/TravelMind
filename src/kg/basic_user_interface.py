import os
import openai
import re
import json
import numpy as np
import pandas as pd
from collections import Counter
from typing import Any, Optional, Tuple, Dict, List, NamedTuple, Set
import scipy
import time
from pprint import pprint as pprint


import tkinter as tk
from tkinter import ttk

'''
Tkinter Basic Widgets

Tkinter provides many basic GUI widgets, such as:
Tk: Create main window (main application).
Label: Display text or images.
Button: Create buttons.
Entry: Single-line text input box.
Text: Multi-line text input box.
Canvas: Drawing area.
Frame: Container for organizing other widgets.
These widgets can be used to build various GUI applications.
TTK Enhanced Widgets

TTK provides enhanced widgets, such as:
ttk.Button: Themed buttons.
ttk.Label: Themed labels.
ttk.Entry: Themed single-line text boxes.
ttk.Progressbar: Progress bars.
ttk.Treeview: Tree view widgets.
ttk.Combobox: Dropdown list boxes.
These widgets have a more modern system design style.
'''





from basic_utils import *
from knowledge_graph import *
from knowledge_graph_querying import *
from initial_card_processing import *

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


model_chat_engine = "gpt-4o" 

SYSTEM_MESSAGE = ("You are a helpful professor and polymath scientist. You want to help a fellow researcher learn more about the world. "
                  + "You are clear, concise, and precise in your answers, and you follow instructions carefully.")



def _gen_chat_response(prompt='hi'):
    response = client.chat.completions.create(
        model=model_chat_engine,
        messages=[
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": prompt},
            ],
        temperature=0.7,
        
        )
    message = response.choices[0].message

    return message['content']

def gen_chat_response(prompt='hi'):
    prompt_succeeded = False
    wait_time = 0.1
    while not prompt_succeeded:
        try:
            response = _gen_chat_response(prompt)
            prompt_succeeded = True
        except:
            print('  LM response failed. Server probably overloaded. Retrying after ', wait_time, ' seconds...')
            time.sleep(wait_time)
            wait_time += wait_time*2  # exponential backoff 
    return response





def convert_abstraction_group_to_concept_list(abs_grp):
    '''
    Convert abstraction group to concept list
    '''
    concept_list = set()
    
    [concept_list.update(concepts) for concepts in abs_grp.values()]
    
    return list(concept_list)




def sample_question_list(question_list):
    """ Take up to 3 questions randomly from a question list and concatenate. used for getting subject lists"""
    res = np.random.choice(question_list, size = min(3, len(question_list)), replace = False)
    return " ".join(res)





def user_triage_list(objects_to_triage):
    '''
    Classify a group of objects to decide what content to follow next
    Show objects to user and ask whether to keep them. Return refined list
    '''
    pass




def extract_abstraction_groups():
    pass








def get_card_representation():
    pass








def get_question_subject_list_in_graph(knowledgeGraph, 
                                       question, 
                                       related_cardIDs = [],
                                       num_random_cards_to_show=5, verbose=False):

    pass

