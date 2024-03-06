import streamlit as st
import os
os.environ['OPENAI_API_KEY'] = 'KEY'
api_key = os.environ["OPENAI_API_KEY"]

from openai import OpenAI
import tiktoken
import json
from datetime import datetime

### ConversationManager Class 
#Step1
class ConversationManager:
    def __init__(self, api_key, base_url="https://api.openai.com/v1", history_file=None, default_model="gpt-3.5-turbo", default_temperature=0.7, default_max_tokens=150, token_budget=4096):
        self.client = OpenAI(api_key=api_key)
        self.base_url = base_url
        if history_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.history_file = f"conversation_history_{timestamp}.json"
        else:
            self.history_file = history_file
        self.default_model = default_model
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.token_budget = token_budget

        self.system_messages = {
            "sassy_assistant": "You are a sassy assistant that is fed up with answering questions.",
            "angry_assistant": "You are an angry assistant that likes yelling in all caps.",
            "thoughtful_assistant": "You are a thoughtful assistant, always ready to dig deeper. You ask clarifying questions to ensure understanding and approach problems with a step-by-step methodology.",
            "custom": "Enter your custom system message here."
        }
        self.system_message = self.system_messages["sassy_assistant"]  # Default persona

        self.load_conversation_history()

    def count_tokens(self, text):
        try:
            encoding = tiktoken.encoding_for_model(self.default_model)
        except KeyError:
            print(f"Warning: Model '{self.default_model}' not found. Using 'gpt-3.5-turbo' encoding as default.")
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

        tokens = encoding.encode(text)
        return len(tokens)

    def total_tokens_used(self):
        try:
            return sum(self.count_tokens(message['content']) for message in self.conversation_history)
        except Exception as e:
            print(f"An unexpected error occurred while calculating the total tokens used: {e}")
            return None
    
    def enforce_token_budget(self):
        try:
            while self.total_tokens_used() > self.token_budget:
                if len(self.conversation_history) <= 1:
                    break
                self.conversation_history.pop(1)
        except Exception as e:
            print(f"An unexpected error occurred while enforcing the token budget: {e}")

    def set_persona(self, persona):
        if persona in self.system_messages:
            self.system_message = self.system_messages[persona]
            self.update_system_message_in_history()
        else:
            raise ValueError(f"Unknown persona: {persona}. Available personas are: {list(self.system_messages.keys())}")

    def set_custom_system_message(self, custom_message):
        if not custom_message:
            raise ValueError("Custom message cannot be empty.")
        self.system_messages['custom'] = custom_message
        self.set_persona('custom')

    def update_system_message_in_history(self):
        try:
            if self.conversation_history and self.conversation_history[0]["role"] == "system":
                self.conversation_history[0]["content"] = self.system_message
            else:
                self.conversation_history.insert(0, {"role": "system", "content": self.system_message})
        except Exception as e:
            print(f"An unexpected error occurred while updating the system message in the conversation history: {e}")

    def chat_completion(self, prompt, temperature=None, max_tokens=None, model=None):
        temperature = temperature if temperature is not None else self.default_temperature
        max_tokens = max_tokens if max_tokens is not None else self.default_max_tokens
        model = model if model is not None else self.default_model

        self.conversation_history.append({"role": "user", "content": prompt})

        self.enforce_token_budget()

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=self.conversation_history,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            print(f"An error occurred while generating a response: {e}")
            return None

        ai_response = response.choices[0].message.content
        self.conversation_history.append({"role": "assistant", "content": ai_response})
        self.save_conversation_history()

        return ai_response
    
    def load_conversation_history(self):
        try:
            with open(self.history_file, "r") as file:
                self.conversation_history = json.load(file)
        except FileNotFoundError:
            self.conversation_history = [{"role": "system", "content": self.system_message}]
        except json.JSONDecodeError:
            print("Error reading the conversation history file. Starting with an empty history.")
            self.conversation_history = [{"role": "system", "content": self.system_message}]

    def save_conversation_history(self):
        try:
            with open(self.history_file, "w") as file:
                json.dump(self.conversation_history, file, indent=4)
        except IOError as e:
            print(f"An I/O error occurred while saving the conversation history: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while saving the conversation history: {e}")

    def reset_conversation_history(self):
        self.conversation_history = [{"role": "system", "content": self.system_message}]
        try:
            self.save_conversation_history()  # Attempt to save the reset history to the file
        except Exception as e:
            print(f"An unexpected error occurred while resetting the conversation history: {e}")

### Streamlit Codes below

#Step2
st.title('AI Chatbot')
    
#Step3
if 'chat_session' not in st.session_state:
    st.session_state['chat_session'] = ConversationManager(api_key=api_key)
conv_manager = st.session_state['chat_session']

#Step4
st.sidebar.title('AI Chatbot Options')
st.sidebar.text('Adjust your AI Chatbot Characteristics')
token_budget = st.sidebar.slider('Token Budget',0,1500,100)
temperature = st.sidebar.slider('AI Temperature',0.0,1.0,0.1)
persona = st.sidebar.selectbox('Select AI Persona', ['Sassy','Angry','Thoughtful','Custom'])

if persona == 'Sassy':
    conv_manager.set_persona('sassy_assistant')
elif persona == 'Angry':
    conv_manager.set_persona('angry_assistant')
elif persona == 'Thoughtful':
    conv_manager.set_persona('thoughtful_assistant')
elif persona == 'Custom':
    persona = st.sidebar.text_area('Enter custom persona')
    if st.sidebar.button('Submit',key='submit'):
        conv_manager.set_custom_system_message(persona)
else:
    conv_manager.set_persona('sassy_assistant')

#Step5
with st.chat_message('assistant'):
    st.write('Welcome! Start a conversation with the AI chatbot!')
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = conv_manager.conversation_history
    
user_input = st.chat_input('Type your message here.')
if user_input:
    conv_manager.chat_completion(user_input, temperature, token_budget)
    st.session_state['chat_history'] = conv_manager.conversation_history
    
#Step6
for chat in st.session_state['chat_history']:
    if chat['role'] != 'system':
        with st.chat_message(chat['role']):
            st.write(chat['content'])

#Step7
def clear_convo():
    conv_manager.reset_conversation_history()
    st.session_state['chat_history'] = conv_manager.conversation_history
                    
st.sidebar.button('Reset Conversation', on_click=clear_convo)