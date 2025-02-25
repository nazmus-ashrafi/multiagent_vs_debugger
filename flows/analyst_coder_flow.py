# from roles import Analyst, Coder, Tester
from utils import find_method_name
import time
from utils import code_truncate
from typing_extensions import TypedDict

from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

import re
from typing import Annotated, Iterator, Literal, TypedDict

from langchain import hub
from langchain_community.document_loaders import web_base
from langchain_community.vectorstores import Chroma
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage, AIMessage, convert_to_messages
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.retrievers import BaseRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langgraph.graph import END, StateGraph, add_messages

from langchain_community.document_loaders import PyPDFLoader
import os
from typing import Iterator, List
# from PyPDF2 import PdfReader 
# import pdfplumber
from langchain_community.document_loaders import PyPDFLoader
# from langchain_groq import ChatGroq

from langchain.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage

from langgraph.checkpoint.memory import MemorySaver

import json


# Shared object between nodes and edges of our graph
class GraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    graph_state: str

    requirement: str
    plan : str
    result: str
    logged: bool

class GraphConfig(TypedDict):
    # max_retries: int
    pass

class Analyst_Coder_Flow(object):
    def __init__(self, PYTHON_DEVELOPER, ANALYST, 
                 requirement, method_name, test, OUTPUT_PATH, task):

        # Our memory log
        self.session_history = {}

        self.method_name = method_name
        self.test = test
        self.OUTPUT_PATH = OUTPUT_PATH
        self.task = task

        # Task
        self.requirement = requirement

        # Roles
        self.analyst = ANALYST
        self.coder = PYTHON_DEVELOPER

        # self.tester = Tester(TEAM, TESTER, requirement, model, majority, max_tokens, temperature, top_p)

    
    def run_flow(self):
        
        
        # Function to write a "high level plan to guide coder"
        def write_plan(state):

            requirement = self.requirement
            plan = self.analyst.implement(requirement)

            return {"plan": plan, "requirement": requirement}
                
        # Function to generate the code completion
        def simple_text2code(state: GraphState):

            plan = state["plan"]
            requirement = state["requirement"]

            naivecode = self.coder.implement(function_description=requirement, plan=plan)
            
            # return func_body
            return {"result": naivecode}
        
        def clean_code_function(original_code: str) -> str:
            # Use regex to remove the triple quotes and ```python\n from the string
            clean_code = re.sub(r'```python\n|```|"""', '', original_code)
            return clean_code.strip()  # Strip any leading/trailing whitespace
                
        def create_log(state):

            result = state.get("result", "No result available")

            result = clean_code_function(result)

            entry_point = self.method_name

            solution = {
                'task_id': self.task['task_id'],
                'prompt': self.requirement+"\n",
                'test': self.test,
                'entry_point': entry_point,
                'completion': result,
                # 'session_history': session_history,
            }

            with open(self.OUTPUT_PATH, 'a') as f:
                f.write(json.dumps(solution) + '\n')
                f.flush()
            
            return {"logged": True}
        

        workflow = StateGraph(GraphState, config_schema=GraphConfig)


        # Define Nodes
        workflow.add_node("write_plan", write_plan)
        workflow.add_node("simple_text2code", simple_text2code)
        workflow.add_node("create_log", create_log)

        # Define Edges
        workflow.add_edge("write_plan", "simple_text2code")
        workflow.add_edge("simple_text2code", "create_log")


        workflow.set_entry_point("write_plan")


        graph = workflow.compile()

        graph.invoke({ "graph_state": "none" })

    