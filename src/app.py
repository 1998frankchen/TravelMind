import os
import uuid

import streamlit as st
from langchain_core.messages import HumanMessage

from agents.agent import Agent



def populate_envs(sender_email, receiver_email, subject):
    os.environ['FROM_EMAIL'] = sender_email
    os.environ['TO_EMAIL'] = receiver_email
    os.environ['EMAIL_SUBJECT'] = subject
    
    
    



def send_email(sender_email, receiver_email, subject, thread_id):
    try:
        populate_envs(sender_email, receiver_email, subject)
        config = {"configurable": {"thread_id": thread_id}}
        st.session_state.agent.graph.invoke(None, config = config)
        st.success('Email sent successfully!')
        # Clear session state
    
    except Exception as e:
        st.error(f"Error sending email: {e}")
    
    
def initialize_agent():
    '''
    Function: If agent does not exist, create a new Agent instance and save it to st.session_state.
    Purpose: Ensure agent is persistent throughout the entire application.
    '''
    if "agent" not in st.session_state:
        st.session_state.agent = Agent()



def render_custom_css():
    st.markdown(
        """
        <style>
        /* Custom CSS styles can be added here */
        </style>
        """
    )
    
    
    




def render_ui():
    pass





def process_query():
    pass





def main():
    pass



