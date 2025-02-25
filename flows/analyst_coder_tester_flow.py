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

    retries: int

    requirement: str

    plan : str

    result: str

    tester_report: str
    tester_verdict: str
    logger: bool


MAX_RETRIES = 2 # means 3 times max retries, passes tester node 4 times 

# Create a thread
config = {"configurable": {"thread_id": "1"}}

class GraphConfig(TypedDict):
    max_retries: int


class Analyst_Coder_Tester_Flow(object):
    def __init__(self, CODER_MAIN, CODER_IMPROVER, ANALYST, TESTER,
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
        self.coder_main = CODER_MAIN
        self.coder_improver = CODER_IMPROVER
        self.tester = TESTER

        # self.tester = Tester(TEAM, TESTER, requirement, model, majority, max_tokens, temperature, top_p)

   

    # graph functions
    def run_flow(self):
        
        
        # Function to write a "high level plan to guide coder"
        def write_plan(state: GraphState):

            requirement = self.requirement
            plan = self.analyst.implement(requirement)

            

            return {"plan": plan, "requirement": requirement}
        
        # Function to generate the code completion
        def simple_text2code(state: GraphState):

            plan = state["plan"]
            requirement = state["requirement"]


            naivecode = self.coder_main.implement(function_description=requirement, plan=plan)
            
            # return func_body
            return {"result": naivecode}
        
        def parse_verdict(report: str) -> str:
            # Use a regular expression to search for the verdict (either "PASS" or "FAIL")
            match = re.search(r'PASS|FAIL', report)
            
            # If a match is found, return the matched verdict
            if match:
                return match.group(0)  # Use group(0) to return the entire matched string
            
            return None  # Return None if no verdict is found

        # Function to test the code
        def test_code(state: GraphState):

            code = state["result"]
            requirement = state["requirement"]

            tester_report = self.tester.implement(function_description=requirement, generated_code=code)

            return {"tester_report": tester_report}
        
        def parse_verdict_node(state: GraphState):


            tester_report = state["tester_report"]

            print("tester_report_____________")
            print(tester_report)
            tester_verdict = parse_verdict(report = tester_report)
    

            return {"tester_verdict": tester_verdict}


        def feedback2code(state: GraphState):

            tester_report = state["tester_report"]
            requirement = state["requirement"]

            # Initializing "retries" key in state & storing
            retries = state["retries"] if state.get("retries") is not None else -1
            print("__________________________Retries__________________________ " + str(retries))

            naivecode = self.coder_improver.implement(function_description=requirement, report=tester_report)
            
            # return func_body
            return {"result": naivecode, "retries": retries + 1}



        # Conditional edje logic
        def decide_to_give_feedback(state: GraphState, config):
            """
            Determines whether the tester should give feedback to the coder.

            Args:
                state (dict): The current graph state

            Returns:
                str: Binary decision for next node to call
            """

            # Initializing "retries" key (if it does not exist) in state (but not storing)
            retries = state["retries"] if state.get("retries") is not None else -1

    
            tester_verdict = state["tester_verdict"]

            max_retries = config.get("configurable", {}).get("max_retries", MAX_RETRIES)

            print("_______________________Tester Verdict_______________________")
            print(tester_verdict)

            if tester_verdict == "PASS":
                return "log"
            else:
                
                if retries < max_retries:
                    return "feedback2code"
                else:
                    return "log"
                
        def clean_code_function(original_code: str) -> str:
            # Use regex to remove the triple quotes and ```python\n from the string
            clean_code = re.sub(r'```python\n|```|"""', '', original_code)
            return clean_code.strip()  # Strip any leading/trailing whitespace
                
        def create_log(state):

            result = state.get("result", "No result available")

            result = clean_code_function(result)

            entry_point = self.method_name

            tester_verdict = state["tester_verdict"]

            # Checking if state["retries"] exists. IE if retries were needed at all in the flow.
            retries = state["retries"] if state.get("retries") is not None else -1

            solution = {
                'task_id': self.task['task_id'],
                'prompt': self.requirement+"\n",
                'test': self.test,
                'entry_point': entry_point,
                'completion': result,
                'tester_verdict': tester_verdict,
                'retries': retries,

                # 'session_history': session_history,
            }

            with open(self.OUTPUT_PATH, 'a') as f:
                f.write(json.dumps(solution) + '\n')
                f.flush()
            
            return {"logger": True}


        ## Compile graph

        workflow = StateGraph(GraphState, config_schema=GraphConfig)

        # Define Nodes
        workflow.add_node("write_plan", write_plan)
        workflow.add_node("simple_text2code", simple_text2code)
        workflow.add_node("test_code", test_code)
        workflow.add_node("parse_verdict_node", parse_verdict_node)
        workflow.add_node("feedback2code", feedback2code)
        workflow.add_node("create_log", create_log)

        # Conditional Edges
        workflow.add_conditional_edges(
            "parse_verdict_node",
            decide_to_give_feedback,
            {
                # path: node name
                "feedback2code": "feedback2code",
                "log": "create_log"
            },
        )

        # Define Edges
        workflow.add_edge("write_plan", "simple_text2code")
        workflow.add_edge("simple_text2code", "test_code")
        workflow.add_edge("test_code", "parse_verdict_node")
        workflow.add_edge("feedback2code", "test_code")


        workflow.set_entry_point("write_plan")

        graph = workflow.compile()

        graph.invoke({ "graph_state": "none" })

    