
# "llm_debugger_flow": "./graphflows/llm_debugger_flow.py:graph"


# from roles import Analyst, Coder, Tester
# from utils import find_method_name
import time
# from utils import code_truncate
from typing_extensions import TypedDict

from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_google_vertexai import VertexAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint



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

from flows.execution import check_correctness

from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, Counter
import tqdm
import numpy as np

from typing import Optional, Callable, Dict
import ast
import contextlib
import faulthandler
import io
import os
import multiprocessing
import platform
import signal
import tempfile

from langchain_ollama.llms import OllamaLLM
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic


import time

def wait_for_10_seconds():
    print("Waiting for 10 seconds...")
    time.sleep(10) 


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

    solution_before_execution: list
    solution_after_execution:  list

    debug_cur_func_impl: str

    needs_block_flow: str
    retries_block: int

    trace_blocks :list

    blocks_verdict_and_explanation: str
    refined_code_after_blocking: str

    blocked_process_happened: bool

# before fixing retries___

# This setting causes some lines to "Errored out at main.py, runtime error"
# MAX_RETRIES = 3
# MAX_RETRIES_BLOCK = 4 # doing 1 actually

# MAX_RETRIES = 2 # causes 4 reports maximum
# MAX_RETRIES_BLOCK = 3 # doing 1 actually

# after fixing retries___

# MAX_RETRIES = 1 # mean 2 times max retries, passes tester noder 3 times 
# MAX_RETRIES_BLOCK = 1 # mean 2 times max retries, passes executioner noder 3 times 

# MAX_RETRIES = 2 # mean 3 times max retries, passes tester noder 4 times 
# MAX_RETRIES_BLOCK = 2 # mean 3 times max retries, passes executioner noder 4 times 

#mr________________________________________________
# MAX_RETRIES = 2 # means 3 times max retries, passes tester node 4 times 
#________________________________________________
#mrb_______________________________________________
MAX_RETRIES_BLOCK = 3 # means 4 times max retries, passes executioner node 5 times # DEFAULT

# MAX_RETRIES_BLOCK = 1 # means 2 times max retries, passes executioner node 3 times 

# MAX_RETRIES_BLOCK = 9 # means 10 times max retries, passes executioner node 11 times 

# MAX_RETRIES_BLOCK = 4 # means 5 times max retries, passes executioner node 6 times 

#________________________________________________


# Create a thread
config = {"configurable": {"thread_id": "1"}}

class GraphConfig(TypedDict):
    max_retries: int


class AC_Debug_Flow(object):
    def __init__(self, CODER_MAIN, CODER_IMPROVER, ANALYST, TESTER,
                 requirement, method_name, test, OUTPUT_PATH, task_id, provider, model, API_KEY):

        # Our memory log
        self.session_history = {}

        self.method_name = method_name
        self.test = test
        self.OUTPUT_PATH = OUTPUT_PATH
        self.task_id = task_id

        # Task
        self.requirement = requirement

        # Roles
        self.analyst = ANALYST
        self.coder_main = CODER_MAIN
        self.coder_improver = CODER_IMPROVER
        self.tester = TESTER

        self.solution_before_exec = []
        self.solution_after_exec = []

        # This LLM will be used for:
            # - Explain the execution of each block and indicate whether the block is correct. If not, provide an explanation of what is wrong.
            # - Use the verdict and explanation to provide a better program

        self.provider = provider
        self.model = model
        self.API_KEY = API_KEY


        if provider == "HuggingFace":
        
            print(f"Using HuggingFace model: {model} as debugger")
            # Initialize your HuggingFaceEndpoint here

            llm = HuggingFaceEndpoint(
                # 1. repo_id="HuggingFaceH4/zephyr-7b-beta",
                # 2. repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
                # 3. repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
                # 4. repo_id="Qwen/QwQ-32B-Preview",
                # 5. repo_id="microsoft/Phi-3.5-mini-instruct",
                # 6. repo_id="mistralai/Mistral-7B-Instruct-v0.2",

                repo_id = model,
                

                task="text-generation",
                temperature= 0,
                # max_new_tokens=512,
                do_sample=False,
                # repetition_penalty=1.03,
            )


            model = ChatHuggingFace(llm=llm)

        elif provider == "deepseek":

            print(f"Selected model for debugging trace and regeneration::")
            print(model)
                
            model = ChatOpenAI(
                # 7. model='deepseek-chat', 
                model=model, 
                openai_api_key=API_KEY,
                openai_api_base='https://api.deepseek.com',
                # max_tokens=1024,
                temperature = 0.0,
            )

        elif provider == "openai":

            print(f"Selected model for debugging trace and regeneration::")
            print(model)

            # 8. model = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0)

            # 9. model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)  # DEFAULT
            ### (points to gpt-4o-mini-2024-07-18)

            # 10. model = ChatOpenAI(model_name="gpt-4o", temperature=0)
            ### (points to gpt-4o-2024-08-06)

            model = ChatOpenAI(model_name=model, temperature=0)

        elif provider == "anthropic":

            print(f"Selected model for debugging trace and regeneration::")
            print(model)

            model = ChatAnthropic(
                model=model, 
                temperature=0,
                api_key=API_KEY,
            )

        elif provider == "groq":

            print(f"Selected model for debugging trace and regeneration::")
            print(model)


            model = ChatGroq(temperature=0,
            model_name=model)

        elif provider == "vertex":

            print(f"Selected model for debugging trace and regeneration::")
            print(model)

            # 18. model = VertexAI(model_name="gemini-2.0-flash-exp",temperature=0.0)

            # 19. model = VertexAI(model_name="gemini-1.0-pro",temperature=0.0)

            model = VertexAI(model_name=model,temperature=0.0)


        else:
            raise ValueError(f"Unsupported provider: {provider}")
        

        self.model = model
        
        

        

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
        def decide_to_block_code(state: GraphState, config):
            """
            Determines whether the code should be blocked

            Args:
                state (dict): The current graph state

            Returns:
                str: Binary decision for next node to call
            """

            # Initializing "retries" key (if it does not exist) in state (but not storing)
            retries_block = state["retries_block"] if state.get("retries_block") is not None else -1

    
            needs_block_flow_verdict = state["needs_block_flow"]

            max_retries = config.get("configurable", {}).get("max_retries", MAX_RETRIES_BLOCK)

            print("_______________________Needs Block? Verdict_______________________")
            print(needs_block_flow_verdict)

            if needs_block_flow_verdict == "NO":
                return "create_log"
            else:
                
                if retries_block < max_retries:
                    return "block_code"
                else:
                    # return "log"
                    return "create_log"

                
        def create_log(state):

            result = state.get("solution_after_execution", "No result available")

            result = result

            entry_point = self.method_name

            # tester_verdict = state["tester_verdict"]

            # Checking if state["retries"] exists. IE if retries were needed at all in the flow.
            retries = state["retries"] if state.get("retries") is not None else -1

            solution = result

            with open(self.OUTPUT_PATH, 'a') as f:
                f.write(json.dumps(solution) + '\n')
                f.flush()

            
            return {"logger": True}
        
        def populate_answer(state):

            result = state.get("result", "No result available")

            entry_point = self.method_name


            self.solution_before_exec = {
                'task_id': self.task_id,
                'prompt': self.requirement+"\n",
                'test': self.test,
                'entry_point': entry_point,
                'completion': result,

                # 'session_history': session_history,
            }

            
            return {"solution_before_execution": self.solution_before_exec}
        
        

        import json

        def get_given_tests(task_id: str, file_path: str = './data/seed.jsonl'):
            """
            This function reads the seed.jsonl file, matches the task_id, and retrieves the given_tests field.
            
            Args:
                task_id (str): The task_id to search for in the JSONL file.
                file_path (str): The path to the seed.jsonl file (defaults to 'seed.jsonl').
            
            Returns:
                list: The list of given_tests for the matching task_id, or None if not found.
            """
            # Open the JSONL file for reading
            with open(file_path, 'r') as f:
                # Loop through each line (each line is a JSON object)
                for line in f:
                    try:
                        # Parse the JSON data
                        data = json.loads(line.strip())
                        
                        # Check if the task_id matches
                        if data.get('task_id') == self.task_id:
                            # Return the given_tests field if found
                            return data.get('given_tests', [])
                        
                    except json.JSONDecodeError:
                        # Handle invalid JSON in the line (if needed)
                        print("Error decoding JSON in line:", line)
                        continue
            
                        # If no match is found, return None or empty list
                        return None

        def clean_code_function(original_code: str) -> str:
            # Use regex to remove the triple quotes and ```python\n from the string
            clean_code = re.sub(r'```python\n|```|"""', '', original_code)
            return clean_code.strip()  # Strip any leading/trailing whitespace
               
        

        def executor_node(state):
            solution_before_execution = state.get("solution_before_execution")
            needs_block_flow = state.get("needs_block_flow") if state.get("needs_block_flow") is not None else "NO"

            blocked_process_happened = state.get("blocked_process_happened") if state.get("blocked_process_happened") is not None else False
            refined_code_after_blocking = state.get("refined_code_after_blocking") if state.get("refined_code_after_blocking") is not None else ""

            retries_block = state["retries_block"] if state.get("retries_block") is not None else -1

            if needs_block_flow == "YES":
                blocked_process_happened = True


            # print(solution_before_execution["test"])

            # tests = extract_tests(solution_before_execution["prompt"])
            execution_results = []
            timeout: float = 3.0
            n_workers: int = 4
            completion_id = Counter()

            results = defaultdict(list)

            futures = []
            total, correct = [], []

            # We need to get the test from seed.json and match it with the task_is column
            # task_id = "HumanEval/3"
            tests = get_given_tests(solution_before_execution["task_id"])

            tests_and_results = []

            for test in tests:
                
                problem = {
                    "prompt": solution_before_execution["prompt"],
                    "test": str(test),
                    "entry_point": solution_before_execution["entry_point"]
                }

                # check_program = (
                #     solution_before_execution["completion"] + "\n" +
                #     problem["test"] + "\n" +
                #     f"check({problem['entry_point']})"
                # )

                # if blocked_process_happened
                #refined_code_after_blocking

                if blocked_process_happened:
                    check_program = (
                        refined_code_after_blocking + "\n" +
                        problem["test"]
                    )
                else:
                    check_program = (
                        clean_code_function(solution_before_execution["completion"]) + "\n" +
                        problem["test"]
                    )



                exec_globals = {}

                try:
                    print("___start test___")
                    exec(check_program, exec_globals)
                    print(check_program)
                    print("___passed___")

                    tests_and_results.append({
                        "test":str(test), 
                        "test_res_by_exec": "pass",
                        "error": "None"
                    })

                    needs_block_flow = "NO"

                    # Do execution verdict = pass (ie. all tests passed)
                except TimeoutError:
                    # Do execution verdict = fail
                    tests_and_results.append({
                        "test":str(test), 
                        "test_res_by_exec": "fail",
                        "error": "timeout"
                    })

                    # TODO: increment retries_block
                    # if blocked_process_happened:
                    #     retries_block = retries_block + 1

                    needs_block_flow = "YES"

                    
                    # colled the first failed test and run the blocking
                    # return "TIMEOUT"
                except Exception as e:
                    # Do execution verdict = fail
                    tests_and_results.append({
                        "test":str(test), 
                        "test_res_by_exec": "fail",
                        "error": str(e)
                    })

                    # if blocked_process_happened:
                    #     retries_block = retries_block + 1

                    needs_block_flow = "YES"
                    # return str(e)

                
                # with ThreadPoolExecutor(max_workers=n_workers) as executor:

                #     args = (problem, solution_before_execution["completion"], timeout, completion_id[solution_before_execution["task_id"]])
                #     future = executor.submit(check_correctness, *args)

                #     for future in tqdm.tqdm(as_completed(future), total=1):
                #         result = future.result()
                #         results[result["task_id"]].append((result["completion_id"], result))
            
            if needs_block_flow == "YES":
                if blocked_process_happened:
                    retries_block = retries_block + 1

            if blocked_process_happened:
                self.solution_after_exec = {
                    'task_id': self.task_id,
                    'prompt': self.requirement+"\n",
                    'test': self.test,
                    'entry_point': solution_before_execution["entry_point"],
                    'completion': clean_code_function(refined_code_after_blocking),
                    'visible_test_status': tests_and_results,

                    'blocking_happened': "yes",
                    'retries_block': retries_block

                    # 'session_history': session_history,
                }
            else:
                self.solution_after_exec = {
                    'task_id': self.task_id,
                    'prompt': self.requirement+"\n",
                    'test': self.test,
                    'entry_point': solution_before_execution["entry_point"],
                    'completion': clean_code_function(solution_before_execution["completion"]),
                    'visible_test_status': tests_and_results,

                    # 'session_history': session_history,
                }


            # print("solution_after_exec:___")
            # print(self.solution_after_exec)

            return {"solution_after_execution": self.solution_after_exec,
                    "needs_block_flow" : needs_block_flow,
                    "retries_block" : retries_block}
                    

            # print("totals:___")
            # print(results.values())
            # print("end:___")
            
        # create a well the formatted wrong program
        def prepare_function_for_blocking(state):

            # wait_for_10_seconds()

            solution_after_execution = state.get("solution_after_execution")

            cur_func_impl = prepare_function_from_seed("HumanEval", 
                                           solution_after_execution["prompt"],
                                           solution_after_execution["completion"], 
                                           solution_after_execution["entry_point"])
            
            if not find_comment(cur_func_impl, solution_after_execution["entry_point"]):
                debug_cur_func_impl = insert_comment(cur_func_impl, extrace_comment(solution_after_execution["prompt"]), solution_after_execution["entry_point"])
            else:
                debug_cur_func_impl = cur_func_impl

            # print("debug_cur_func_impl__________________")
            # print(debug_cur_func_impl)

            return {"debug_cur_func_impl": debug_cur_func_impl}
        


        def parse_first_failed_test(solution):
            # Get the visible_test_status field
            test_statuses = solution.get('visible_test_status', [])
            
            # Iterate through the test results
            for test_status in test_statuses:
                # Check if the test has failed
                if test_status.get('test_res_by_exec') == 'fail':
                    return test_status  # Return the first failed test

            return None  # Return None if no failed test is found


        def create_blocks_and_trace(state):

            debug_cur_func_impl = state.get("debug_cur_func_impl")
            solution_after_execution = state.get("solution_after_execution")


            # trace block takes in: the formatted wrong program, the test that produced this wrong program, and entry point
            trace_blocks = get_code_traces_block(IMPORT_HEADER + 
                                                 debug_cur_func_impl, 
                                                 parse_first_failed_test(solution_after_execution)["test"], 
                                                 solution_after_execution["entry_point"])

            # 10 is a hyperparam (ie. we are limiting to 10 block traces)
            selected_blocks = trace_blocks[:int(10/2)] + trace_blocks[-int(10/2):]
            trace_blocks  = selected_blocks
            
            print("our 10 trace blocks__________________")
            print(trace_blocks)

            return {"trace_blocks": trace_blocks}


        def get_block_correctness_and_explanation(state):

            debug_cur_func_impl = state.get("debug_cur_func_impl")
            solution_after_execution = state.get("solution_after_execution")

            wrong_code = solution_after_execution["completion"]
            failed_test = parse_first_failed_test(solution_after_execution)["test"]


            BLOCK_DEBUG_REQUEST_PROMPT = ChatPromptTemplate.from_template(
                template="""
                
                {wrong_code}

                The code above fails the given unit test:
                {failed_test}

                Execution Trace:
                {debug_cur_func_impl}
                
                I need help debugging this. Above is the code execution trace, block by block, with intermediate variable values. Please explain the execution of each block and indicate whether the block is correct. If not, provide an explanation of what is wrong.

                Please wrap your response into a JSON object that contains keys `block` with the name of each block, key `content` with the content of the block, key `correct` with value False or True, and key `explanation` with an explanation on the bug. 
                Example Answers 1: 
                "block": "BLOCK-1", "correct": "True", "explanation": "The block initializes variable `a` and `b`."
                Example Answers 2: 
                "block": "BLOCK-2", "correct": "False", "explanation": "The block is incorrect because the code does not add the two integers together, but instead subtracts the second integer from the first. To fix this issue, we should change the operator from `-` to `+` in the return statement."

                This will ensure that the function returns the correct output for the given input.
                """
            )

            print("Debugging execution trace with :")
            print(self.model)

            corr_chain = BLOCK_DEBUG_REQUEST_PROMPT | self.model | StrOutputParser()
            generation = corr_chain.invoke({"wrong_code": wrong_code, "failed_test": failed_test, "debug_cur_func_impl": debug_cur_func_impl})


            # print("------------------------Block verdict and Explaination Generation------------------------")
            # print(generation)

            return {"blocks_verdict_and_explanation": generation}
        

        
        def code_regen_with_block_feedback(state):
            solution_after_execution = state.get("solution_after_execution")
            task_description = solution_after_execution["prompt"]

            wrong_code = solution_after_execution["completion"]

            failed_test = parse_first_failed_test(solution_after_execution)["test"]


            blocks_verdict_and_explanation = state.get("blocks_verdict_and_explanation")

            CODE_REGEN_WITH_BLOCK_FEEDBACK_PROMPT = ChatPromptTemplate.from_template(
                template="""

                Task Description:
                {task_description}
                
                Code:
                {wrong_code}

                The code above fails the given unit test:
                {failed_test}

                Execution Trace with explanation:
                {blocks_verdict_and_explanation}

                You are an expert programming assistant. Generate the refined program based on the information provided which will pass the unit test.
                Please respond with code only.

                Do not include any comments. Return only the code.
                Do not start with "Here is the refined code that passes the unit test:". Just answer with the code.

                """
            )

            #### This extra lines was used in CODE_REGEN_WITH_BLOCK_FEEDBACK_PROMPT when used used "Claude Haiku" to ensure model gives code only.
            # Do not include any comments. Return only the code.
            # Do not start with "Here is the refined code that passes the unit test:". Just answer with the code.

            print("Regenerating code with :")
            print(self.model)

            refine_chain = CODE_REGEN_WITH_BLOCK_FEEDBACK_PROMPT | self.model | StrOutputParser()
            generation = refine_chain.invoke(
                {"task_description":task_description, 
                 "wrong_code": wrong_code, 
                 "failed_test": failed_test,
                 "blocks_verdict_and_explanation": blocks_verdict_and_explanation
                 })
            
            print("------------------------Regenerated Code------------------------")
            print(clean_code_function(str(generation)))


            return {"refined_code_after_blocking": clean_code_function(str(generation)), "blocked_process_happened": True}



            

        ## Compile graph

        workflow = StateGraph(GraphState, config_schema=GraphConfig)

        # Define Nodes
        workflow.add_node("write_plan", write_plan)
        workflow.add_node("simple_text2code", simple_text2code)
        # workflow.add_node("test_code", test_code)
        # workflow.add_node("parse_verdict_node", parse_verdict_node)
        # workflow.add_node("feedback2code", feedback2code)
        workflow.add_node("populate_answer", populate_answer)
        workflow.add_node("executor_node", executor_node)
        workflow.add_node("prepare_function_for_blocking", prepare_function_for_blocking)
        workflow.add_node("create_blocks_and_trace", create_blocks_and_trace)
        workflow.add_node("get_block_correctness_and_explanation", get_block_correctness_and_explanation)
        workflow.add_node("code_regen_with_block_feedback", code_regen_with_block_feedback)

        workflow.add_node("create_log", create_log)




        # workflow.add_node("create_log", create_log)

        

        workflow.add_conditional_edges(
            "executor_node",
            decide_to_block_code,
            {
                # path: node name
                "block_code": "prepare_function_for_blocking",
                "create_log": "create_log"
            },
        )

        # Define Edges
        workflow.add_edge("write_plan", "simple_text2code")
        workflow.add_edge("simple_text2code", "populate_answer")
        workflow.add_edge("populate_answer", "executor_node")

        workflow.add_edge("prepare_function_for_blocking", "create_blocks_and_trace")
        workflow.add_edge("create_blocks_and_trace", "get_block_correctness_and_explanation")
        workflow.add_edge("get_block_correctness_and_explanation", "code_regen_with_block_feedback")

        workflow.add_edge("code_regen_with_block_feedback", "executor_node")




        workflow.set_entry_point("write_plan")

        graph = workflow.compile()

        # Graph recursion limit increases from 25 (default) to 10000
        graph.invoke({ "graph_state": "none" }, {"recursion_limit": 10000})




# HELPERS for LDB Process ___________________________________________________________________________

IMPORT_HEADER = "from typing import *\nimport math\nfrom heapq import *\nimport itertools\nimport re\nimport typing\nimport heapq\n_str=str\nimport re\n"

# HELPERS (prepare_function_from_seed)
def find_comment(func_impl: str, entry: str ) -> bool:
    func_impl_lines = func_impl.split('\n')
    for i, line in enumerate(func_impl_lines):
        if line.startswith('def ' + entry + "("):
            break
    func_body = "\n".join(func_impl_lines[i:])
    if func_body.find('\"\"\"') != -1 or func_body.find('\'\'\'') != -1:
        return True
    return False

def get_function(prompt):
    lines = prompt.split('\n')
    cur_func = ""
    funcs = []
    for i, l in enumerate(lines):
        if l.startswith("def "):
            if cur_func == "":
                cur_func = l
            else:
                funcs.append([func_name, cur_func])
                cur_func = l
            func_name = l.split("def ")[1].split("(")[0]
        elif cur_func != "":
            cur_func += "\n" + l
    return funcs

def fix_func_impl_comments(func_impl: str, prompt: str, entry) -> str:
    # extract comments from prompt and insert them into func_impl after the function header
    if prompt.find('\"\"\"') != -1:
        comments = prompt.split('\"\"\"')[1]
    elif prompt.find('\'\'\'') != -1:
        comments = prompt.split('\'\'\'')[1]
    # Get the function header
    func_impl_lines = func_impl.split('\n')
    for i, line in enumerate(func_impl_lines):
        if line.startswith('def') and entry in line:
            break
    # Insert comments after the function header
    func_impl_lines.insert(i+1, '    \"\"\"' + comments + '\"\"\"')
    return '\n'.join(func_impl_lines)


# MAIN
# prepare_function_from_seed
def prepare_function_from_seed(dataset_type, prompt, seed, entry_point):
    if dataset_type in ["HumanEval", "MBPP"]:
        if (prompt in seed) or (('def ' + entry_point + '(') in seed):
            # It has the function header, no need to add
            cur_func_impl = seed
        else:
            cur_func_impl = prompt + "\n" + seed
        # Add auxilary function
        funcs = get_function(prompt)
        seed_funcs = [func[0] for func in get_function(seed)]
        for func in funcs:
            if func[0] not in seed_funcs:
                cur_func_impl = func[1] + "\n" + cur_func_impl
        # Add comments
        if not find_comment(cur_func_impl, entry_point):
            cur_func_impl = fix_func_impl_comments(cur_func_impl, prompt, entry_point)
    elif dataset_type in ["TransCoder"]:
        # It contains a whole program
        cur_func_impl = seed
    # Add import header
    if IMPORT_HEADER not in cur_func_impl:
        cur_func_impl = IMPORT_HEADER + cur_func_impl
    assert isinstance(cur_func_impl, str)
    return cur_func_impl


# HELPER (Add comments)
def insert_comment(func_impl: str, comment: str, entry: str) -> str:
    func_impl_lines = func_impl.split('\n')
    for i, line in enumerate(func_impl_lines):
        if line.startswith('def ' + entry + '('):
            break
    func_impl_lines.insert(i + 1, '    \"\"\"' + comment + '\"\"\"')
    return '\n'.join(func_impl_lines)

def remove_comment(old_block: List[str]) -> str:
    new_block = []
    old_block_lines = old_block.split('\n')
    for line in old_block_lines:
        if line.lstrip().startswith('#'):
            continue
        new_block.append(line)
    if len(new_block) == 1:
        return new_block[0]
    else:
        return '\n'.join(new_block)

def extrace_comment(prompt: str) -> str:
    if prompt.find('\"\"\"') != -1:
        comments = prompt.split('\"\"\"')[-2]
    elif prompt.find('\'\'\'') != -1:
        comments = prompt.split('\'\'\'')[-2]
    return comments


# MAIN
# Add comments
def add_comments(cur_func_impl, item):
    if not find_comment(cur_func_impl, item["entry_point"]):
        debug_cur_func_impl = insert_comment(cur_func_impl, extrace_comment(item["prompt"]), item["entry_point"])
    else:
        debug_cur_func_impl = cur_func_impl

    return debug_cur_func_impl



# CREATE BLOCKS AND TRACE FUNCTIONALITY >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Split into blocks
# import jsonl
import json
import os.path
import sys
from typing import Any, Dict, Optional, Set
import astroid
from astroid import nodes
from astroid.builder import AstroidBuilder
import time
import ast
import re
import os
import random
from staticfg import CFGBuilder

DEBUG = False


# HELPERS
def divide(prog):
    try:
        cfg = CFGBuilder().build_from_src('block', prog)
    except Exception as e:
        return None, str(e)
    divided_block = []
    prog_lines = prog.split("\n")
    for block in cfg:
        divided_block.append([block, prog_lines[block.at():block.end()+1], block.id])
    return divided_block, None

def get_error_msg(error):
    error_lines = error.split('\n')
    error_msg = ""
    last_l = ""
    code = ""
    for l in error_lines:
        if "File \"" in last_l:
            code = l
        elif "Error: " in l:
            error_msg = ("This line is wrong: ```" + code + "```\n" + l) if "__var_list" not in code else l
            break
        last_l = l
    return error_msg

# Get trace
def get_trace_line(trace, funcname, fname):
    mark = f"--- modulename: .tmp.py, funcname: {funcname}" + "\n"
    lines = trace.split(mark)[1].split("\n")
    traces = []
    for l in lines:
        # trace also record comment lines for some reason
        if l.lstrip().startswith("\'\'\'") or l.lstrip().startswith("\"\"\"") or l.lstrip().startswith("#"):
            continue
        traces.append(l)
    return traces

# Return: "*timeout*" or "*execution fail*{error_msg}" or "*parse fail*{ferr}" or line_traces(List)
def get_trace(prog, funcname):
    fname = '.tmp.py.' + str(random.randint(0, 10000))
    f = open(fname, "w")
    f.write(prog)
    f.close()
    # run in command line python -m trace -t tmp.py > trace
    import subprocess
    try:
        res=subprocess.run(["python3", "-m", "trace", "-t", fname], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
    except AssertionError:
        # This is expected if fail the test assetion
        pass
    except subprocess.TimeoutExpired:
        return "*timeout*"
    except Exception as e:
        error_msg = get_error_msg(res.stderr.decode('utf-8'))
        print("Trace Execution Fail:" + error_msg)
        return "*execution fail*" + error_msg
    finally:
        os.remove(fname)
    trace = res.stdout.decode('utf-8')
    # Find --- modulename: tmp, funcname: {funcname}
    try:
        trace = get_trace_line(trace, funcname, fname)
    except IndexError:
        ferr_name = "../error/.error.py" + str(time.time())
        if DEBUG:
            ferr = open(ferr_name, 'w')
            ferr.write(prog)
            ferr.close()
        return f"*parse fail*{ferr_name}" 
    # Find all lines with .tmp.py
    line_trace = []
    for l in trace:
        if l.startswith(fname):
            import re
            m = re.search(f"^{fname}", l)
            if (not line_trace) or (line_trace[-1] not in l):
                line_trace.append(l[m.end():])
    return line_trace

def collect_runtime_value_simple(value_prof_prog):
    hook = ""
    import sys
    hooked_prog = hook + "\n" + value_prof_prog
    fname = "tmp_line.py" + f".{random.randint(0,10000)}"
    with open(fname, "w") as f:
        f.write(hooked_prog)
    import subprocess
    try:
        res=subprocess.run(["python3", fname], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
    except subprocess.TimeoutExpired:
        return "*timeout*"
    finally:
        os.remove(fname)
    output = res.stderr.decode('utf-8')
    if "Traceback (most recent call last):" in output and ("AssertionError" not in output):
        output = get_error_msg(output)
        return "*execution fail*" + output
    output = res.stdout.decode('utf-8')
    return output

def get_lineno(trace_line):
    match = re.search("\([0-9]+\):", trace_line)
    return int(trace_line[match.start()+1:match.end()-2])

def get_line(trace_line):
    match = re.search("\([0-9]+\):", trace_line)
    return trace_line[match.end()+1:]

def get_indent(trace_line):
    match = re.search("\([0-9]+\):", trace_line)
    len1 = len(trace_line[match.end()+1:])
    len2 = len(trace_line[match.end()+1:].lstrip())
    return int((len1-len2)/4)

def extract_value(output):
    output = output.split("\n")[:-1]
    output = [x for x in output if x.startswith('Value_')]
    return output

#
def parse_runtime_value_simple_block(output, trace_lines):
    trace_idx = 0
    blocks = []
    blk = []
    value_profiles = extract_value(output)
    trace_len = len(trace_lines)
    trace_linenos = [get_lineno(l) for l in trace_lines]
    last_bp = ""
    trace_idx = 0
    for i, l in enumerate(value_profiles):
        if trace_idx >= trace_len:
            break
        lineno = int(l.split(':')[1].split('|')[0])
        values = '\t'.join(l.split('|')[1:])
        values = values if len(values) < 100 else (values[:50] + "..." + values[-50:])
        if lineno not in trace_linenos:    
            #payload = "    "*get_indent(trace_lines[trace_idx]) + "# " + values
            last_bp = values
            continue
        blk.append("    "*get_indent(trace_lines[trace_idx]) + "# " + last_bp)
        while trace_idx < trace_len and get_lineno(trace_lines[trace_idx]) != lineno:
            trace_l = trace_lines[trace_idx]
            blk.append(get_line(trace_l))
            trace_idx += 1
        if trace_idx == trace_len:
            break
        blk.append(get_line(trace_lines[trace_idx]))
        blk.append("    "*get_indent(trace_lines[trace_idx]) + "# " + values)
        last_bp = values
        blocks.append(blk)
        blk = []
        trace_idx += 1
    if trace_idx < trace_len:
        blk = ["    "*get_indent(trace_lines[trace_idx]) + "# " + last_bp] + blk
        while trace_idx < trace_len:
            blk.append(get_line(trace_lines[trace_idx]))
            trace_idx += 1
        blocks.append(blk)
    return blocks


 

def get_range(prog, entry):
    tree = AstroidBuilder().string_build(prog)
    for ele in tree.body:
        if isinstance(ele, nodes.FunctionDef) and ele.name == entry:
            return [ele.lineno-1, ele.end_lineno-1] # Lineno start from 0
    return None

def get_after(stmts):
    for s in stmts:
        if s == "":
            continue
        else:
            return s.strip(), int((len(s) - len(s.lstrip()))/4)



def instrument_simple_block(prog, entry, divided_blocks):
    stmts = prog.split("\n")
    # Get range of entry function
    rang = get_range(prog, entry)
    block_insert = set([b[0].at() - 1 for b in divided_blocks] + [b[0].end() for b in divided_blocks])
    if rang is None:
        assert False, f"{entry} not in {prog}!"
    res = []
    for i, stmt in enumerate(stmts):
        if i < rang[0]:
            res.append(stmt)
            continue
        elif i > rang[1]:
            res.append(stmt)
            break
        if (i+1) not in block_insert:
            res.append(stmt)
            continue
        # indent the same as this statement
        refs, indent_after = get_after(reversed(stmts[:i+1]))
        # Unless 
        if refs.startswith("else:") or refs.startswith("elif ") or refs.startswith("if ") or refs.startswith("while ") or refs.startswith("for ") or refs.startswith("def "):
            refs, indent_after = get_after(stmts[i+1:])
        payload = "    "*indent_after + f"__var_list = vars();print(f'Value_After:{i+1}|' + '|'.join([(x + '=' + _str(__var_list[x])) for x in __var_list if not x.startswith('__')]));"
        if stmt.find(" return ") != -1:
            stmt = stmt.replace(" return ", " _ret = ")
            payload = payload + " return _ret"
        res.append(stmt)
        res.append(payload)
    return "\n".join(res)



# MAIN Function
def get_code_traces_block(prog, test, entry):
    log_of_tracing = ""
    # Divide program into basic block units
    divided_blocks, error = divide(prog)
    prog_lines = prog.split("\n")

    # print(divided_blocks)
    if divided_blocks is None:
        return "*execution fail*" + error
    # Collect Execution Traces
    if test.find("assert ") != -1:
        test = test.replace("assert ", "print(").split(" == ")[0] + ")"
    exec_prog = prog + "\n" + test
    trace_lines = get_trace(exec_prog, entry)
    if isinstance(trace_lines, str):
        if trace_lines == "*timeout*" or trace_lines.startswith("*execution fail*") or trace_lines.startswith("*parse fail*"):
            return trace_lines
    log_of_tracing += str("Trace:\n"+ '\n'.join(trace_lines[:10]))
    value_prof_prog = instrument_simple_block(prog, entry, divided_blocks)
    log_of_tracing += str("\nValue Profile Program:\n" + value_prof_prog + "\n" + test + "\n")
    output = collect_runtime_value_simple(value_prof_prog + "\n" + test)
    if output == "*timeout*" or output.startswith("*execution fail*"):
        return output
    log_of_tracing += "\n" + str("Value Profile Output:\n" + output)
    runtime_value = parse_runtime_value_simple_block(output, trace_lines)
    
    if not os.path.exists("./tracing_log"):
        os.makedirs("./tracing_log")
    log_file = "./tracing_log/trace.log."+str(random.randint(0, 10000))
    with open(log_file, 'w') as f:
        f.write(log_of_tracing)
        print(f"Writing tracing logs to {log_file}")
    return runtime_value





