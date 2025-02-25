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

from agents.coder_agent import CoderAgent
from agents.analyst_agent import AnalystAgent


import re


#_______________________________________________
# Helper Functions
#_______________________________________________


# Preprocessing HumanEval

def prompt_details_segregator(input_text: str):
    # Step 1: Extract imports (all text before the first 'def')
    imports_match = re.search(r"(.*?)(?=def)", input_text, re.DOTALL)
    imports = imports_match.group(1).strip() if imports_match else ""

    # Step 2: Extract function definition (from 'def' until the first colon ':' followed by triple quotes)
    func_def_match = re.search(r"(def .*?:)(?=\s*\"\"\")", input_text, re.DOTALL)
    func_def = func_def_match.group(1).strip() if func_def_match else ""

    # Step 3: Extract instructions between triple double quotes ("""...""")
    instructions_match = re.search(r'"""\s*(.*?)\s*"""', input_text, re.DOTALL)
    instructions = instructions_match.group(1).strip() if instructions_match else ""

    return imports, func_def, instructions

# Example usage:
input_text = """from typing import List

def separate_paren_groups(paren_string: str) -> List[str]:
    \"\"\"
    Input to this function is a string containing multiple groups of nested parentheses.
    Your goal is to separate those groups into separate strings and return the list of those.
    Separate groups are balanced (each open brace is properly closed) and not nested within each other.
    Ignore any spaces in the input string.

    >>> separate_paren_groups('( ) (( )) (( )( ))')
    ['()', '(())', '(()())']
    \"\"\"
"""

# Call the function
imports, func_def, instructions = prompt_details_segregator(input_text)

# Output the results
# print("Imports:\n", imports)
# print("Function Definition:\n", func_def)
# print("Instructions:\n", instructions)



def remove_metadata_from_test(input_text: str) -> str:
    # Use regular expression to find and remove anything before the 'def' keyword
    result = re.search(r"(def.*)", input_text, re.DOTALL)
    # Return only the function definition and the rest of the code
    return result.group(1).strip() if result else input_text

# Example usage:
input_text = """METADATA = { 'author': 'jt', 'dataset': 'test' } def check(candidate): assert candidate('(()()) ((())) () ((())()())') == [ '(()())', '((()))', '()', '((())()())' ] assert candidate('() (()) ((())) (((())))') == [ '()', '(())', '((()))', '(((())))' ] assert candidate('(()(())((())))') == [ '(()(())((())))' ] assert candidate('( ) (( )) (( )( ))') == ['()', '(())', '(()())']
"""

# Call the function
clean_test = remove_metadata_from_test(input_text)

# Output the result
# print(clean_test)


#_______________________________________________
# Graph State
#_______________________________________________


class GraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    question: str
    documents: list[Document]
    candidate_answer: str
    retries: int
    web_fallback: bool

    intent: str
    grades: list[str]
    grade_retries: int
    
    summary: str
    verdict: str

    requirement: str
    plan : str
    result: str


class GraphConfig(TypedDict):
    max_retries: int


#_______________________________________________
# Graph Functions
#_______________________________________________

model = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0)
num_comps = 1024  # Define max token length

# Roles
analyst = AnalystAgent(model=model)
coder = CoderAgent(model=model, flow="analyst_coder_flow")


# Function to write a "high level plan to guide coder"
def write_plan(state):

    requirement = convert_to_messages(state["messages"])[-1].content
    plan = analyst.implement(requirement)

    return {"plan": plan, "requirement": requirement}
        
# Function to generate the code completion
def simple_text2code(state: GraphState):

    plan = state["plan"]
    requirement = state["requirement"]

    naivecode = coder.implement(function_description=requirement, plan=plan)
    
    # return func_body
    return {"result": naivecode}



#_______________________________________________
# Compile Graph
#_______________________________________________


workflow = StateGraph(GraphState, config_schema=GraphConfig)

# Define Nodes
workflow.add_node("write_plan", write_plan)
workflow.add_node("simple_text2code", simple_text2code)


# Define Edges
workflow.add_edge("write_plan", "simple_text2code")

workflow.set_entry_point("write_plan")


# Compile with memory
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)



