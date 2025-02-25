import os
import copy
import json
import argparse
import tqdm
from datasets import load_dataset
from utils import prompt_split_humaneval, mbpp_test_formatter

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_vertexai import VertexAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from agents.coder_agent import CoderAgent
from agents.analyst_agent import AnalystAgent
from agents.tester_agent import TesterAgent

from flows.flow import GraphState
from flows.flow import Flow
from flows.analyst_coder_flow import Analyst_Coder_Flow
from flows.analyst_coder_tester_flow import Analyst_Coder_Tester_Flow
from flows.llm_debugger_flow import LDB_Flow
from flows.debugger_only_flow import Debugger_Only_Flow
from flows.AC_Debug_flow import AC_Debug_Flow

from langchain_groq import ChatGroq
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from langchain_ollama.llms import OllamaLLM
from datasets import DatasetDict


parser = argparse.ArgumentParser()

parser.add_argument('--output_path', type=str, default='output.jsonl')
parser.add_argument('--dataset', type=str, default='humaneval')
parser.add_argument('--signature', action='store_true')
parser.add_argument('--lang', type=str, default='python')
parser.add_argument('--range', type=str, default='full')
parser.add_argument('--provider_and_model', type=str, required=True,
                    help="Provider and model in the format 'provider:model' (e.g., 'HuggingFace:HuggingFaceH4/zephyr-7b-beta')")
parser.add_argument('--api_key', type=str, default='full')
parser.add_argument('--flow', type=str, default='basic', required=True)

args = parser.parse_args()



if __name__ == '__main__':

    OUTPUT_PATH = args.output_path
    API_KEY = args.api_key
    FLOW = args.flow

    # load dataset
    if args.dataset == 'humaneval':
        if args.lang == 'python':

            # DATASET ___________ DATASET ___________ DATASET ___________ DATASET ___________ DATASET ___________ DATASET ___________

            dataset = load_dataset("openai_humaneval")
            
            # print(dataset)

            # RANGE SELECTOR ___________RANGE SELECTOR _____________________RANGE SELECTOR ____________________________RANGE SELECTOR
            # RANGE SELECTOR ____________________________RANGE SELECTOR ____________________________RANGE SELECTOR ____________________________

            # dataset = DatasetDict({'test':dataset['test'].select(range(82, 84))})
            # dataset = DatasetDict({'test':dataset['test'].select(range(0, 164))}) # full ds

            if args.range == 'full':
                dataset = DatasetDict({'test':dataset['test'].select(range(0, 164))})
            else:
                try:
                    # Parse the range, assuming it's provided as "start:end"
                    start, end = map(int, args.range.split(':'))
                    dataset = DatasetDict({'test': dataset['test'].select(range(start, end))})
                except ValueError:
                    raise ValueError("Invalid range format. Use 'start:end' (e.g., '0:10').")
                
            # Example Range Setting Usage
            # python main.py --range 0:10

            #  ____________________________ ____________________________ ____________________________
                
        

            # dataset = DatasetDict({'test':dataset['test'].select(range(42, 164))})


            # dataset = DatasetDict({'test':dataset['test'].select(range(153, 155))})

            # SPECIFIC ROWS SELECTOR ____________________________
            # Define the specific rows you want to select
            # indices = [53, 45]
            # indices = [53]
            # indices = [15]

            # # Select the specific rows from the 'test' set
            # dataset = DatasetDict({'test': dataset['test'].select(indices)})

            #___________

            dataset_key = ["test"]

    elif args.dataset == 'mbpp':
        if args.lang == 'python':
            dataset = load_dataset("mbpp")

            # print(dataset["test"]["text"][0])
            # Write a python function to remove first and last occurrence of a given character from the string.

            dataset_key = ["test"]

    # CODE GENERATION________________________________________________________________________________________________________________________

    # Opens the output file in write mode ('w+'), which creates the file if it doesn’t exist.
    with open(OUTPUT_PATH, 'w+') as f:
        
        for key in dataset_key:

            # Iterates over the dataset keys, creating a progress bar (pbar) to visualize the loop.
            if args.dataset == 'humaneval':
                pbar = tqdm.tqdm(dataset[key], total=len(dataset[key]))
            elif args.dataset == 'mbpp':
                pbar = tqdm.tqdm(dataset[key], total=len(dataset[key]))
            # pbar = tqdm.tqdm(dataset[key][:2], total=2)

            
            # Generating the completion
            # Input = intent
            # Output = imports + code
            for idx, task in enumerate(pbar):
                
                if args.dataset == 'humaneval':
                    method_name = task['entry_point']

                    # before_func, signature, comment, test_case
                    before_func, signature, intent, public_test_case = prompt_split_humaneval(task['prompt'],method_name)

                    args.signature = True
                    if args.signature:
                        intent = task['prompt']
                    
                    test = task['test']

                if args.dataset == 'mbpp':

                    # print(task)
                    
                    check_function, original_name = mbpp_test_formatter(str(task['test_list']))

                    test = check_function

                    method_name = original_name

                    args.signature = True
                    if args.signature:
                        # intent = task['text'] 
                        intent = task['text'] + "\n" + f"Function name must be {method_name}"
                

                try:

                    #    MODELS ______________________________ MODELS ______________________________ MODELS _________________ MODELS _____________ 
                    #   _______________ MODELS ______________________________ MODELS ______________________________ MODELS _______________________
                    # This model powers the agents

                    # Parse the provider and model
                    try:
                        provider, model_arg = args.provider_and_model.split(':')
                    except ValueError:
                        raise ValueError("Invalid format for provider and model. Use 'provider:model' (e.g., 'HuggingFace:HuggingFaceH4/zephyr-7b-beta')")


                    if provider == "HuggingFace":
    
                        print(f"Using HuggingFace model: {model_arg}")
                        # Initialize your HuggingFaceEndpoint here

                        print("Running model:")
                        print(model_arg)

                        llm = HuggingFaceEndpoint(
                            # 1. repo_id="HuggingFaceH4/zephyr-7b-beta",
                            # 2. repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
                            # 3. repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
                            # 4. repo_id="Qwen/QwQ-32B-Preview",
                            # 5. repo_id="microsoft/Phi-3.5-mini-instruct",
                            # 6. repo_id="mistralai/Mistral-7B-Instruct-v0.2",

                            repo_id = model_arg,
                            

                            task="text-generation",
                            temperature= 0,
                            # max_new_tokens=512,
                            do_sample=False,
                            # repetition_penalty=1.03,
                        )

                
                        model = ChatHuggingFace(llm=llm)

                    elif provider == "deepseek":

                        print("Running model:")
                        print(model_arg)
                            
                        model = ChatOpenAI(
                            # 7. model='deepseek-chat', 
                            model=model_arg, 
                            openai_api_key=API_KEY,
                            openai_api_base='https://api.deepseek.com',
                            # max_tokens=1024,
                            temperature = 0.0,
                        )

                    elif provider == "openai":

                        print("Running model:")
                        print(model_arg)

                        # 8. model = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0)

                        # 9. model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)  # DEFAULT
                        ### (points to gpt-4o-mini-2024-07-18)

                        # 10. model = ChatOpenAI(model_name="gpt-4o", temperature=0)
                        ### (points to gpt-4o-2024-08-06)

                        model = ChatOpenAI(model_name=model_arg, temperature=0)

                    elif provider == "anthropic":

                        print("Running model:")
                        print(model_arg)

                        # 11. 
                        # model = ChatAnthropic(
                        #     model='claude-3-haiku-20240307', 
                        #     temperature=0,
                        #     )

                        # 12. 
                        # model = ChatAnthropic(
                        #     model='claude-3-5-sonnet-20241022', 
                        #     temperature=0,
                        # )

                        # 13. 
                        # model = ChatAnthropic(
                        #     model='claude-3-5-haiku-20241022', 
                        #     temperature=0,
                        # )

                        model = ChatAnthropic(
                            model=model_arg, 
                            temperature=0,
                            api_key=API_KEY,
                        )

                    elif provider == "groq":

                        print("Running model:")
                        print(model_arg)

                        # 14.  model = ChatGroq(temperature=0,
                    #     model_name="llama-3.3-70b-versatile")


                    # 15.  model = ChatGroq(temperature=0,
                    #     model_name="llama-3.1-8b-instant")

                    # 16.  model = ChatGroq(temperature=0,
                    #     model_name="gemma2-9b-it")

                        # 17.  model = ChatGroq(temperature=0,
                    #     model_name="mixtral-8x7b-32768")


                        model = ChatGroq(temperature=0,
                        model_name=model_arg)

                    elif provider == "vertex":

                        print("Running model:")
                        print(model_arg)

                        # 18. model = VertexAI(model_name="gemini-2.0-flash-exp",temperature=0.0)

                        # 19. model = VertexAI(model_name="gemini-1.0-pro",temperature=0.0)

                        model = VertexAI(model_name=model_arg,temperature=0.0)


                    else:
                        raise ValueError(f"Unsupported provider: {provider}")


                    # _______________ ROLES _______________
                    basic_coder_agent = CoderAgent(model=model, flow="basic")
                    # basic_coder_agent = CoderAgent(model=model, flow="basic_optimized_few_shot")


                    analyst_agent = AnalystAgent(model=model)
                    coder_agent = CoderAgent(model=model, flow="analyst_coder_flow")
                    coder_improver = CoderAgent(model=model, flow="tester_coder_flow")
                    tester = TesterAgent(model=model)

                    # _______________ FLOWS  _______________ FLOWS _______________ FLOWS  _______________ FLOWS  _______________ FLOWS 
                    # FLOWS  _______________ FLOWS _______________ FLOWS  _______________ FLOWS _______________ FLOWS _______________ 

                    ### 1. BASIC_______________

                    if FLOW == "basic":

                        print("Running flow:")
                        print(FLOW)

                        flow = Flow(PYTHON_DEVELOPER=basic_coder_agent,
                                    requirement=intent, method_name=method_name, test=test, OUTPUT_PATH = OUTPUT_PATH, task=task)
                        flow.run_flow()

                    elif FLOW == "AC":

                        print("Running flow:")
                        print(FLOW)

                        ### 2. Analyst Coder Flow (AC)_______________
                        analyst_coder_flow = Analyst_Coder_Flow(PYTHON_DEVELOPER=coder_agent, ANALYST = analyst_agent,
                                    requirement=intent, method_name=method_name, test=test, OUTPUT_PATH = OUTPUT_PATH, task=task)
                        analyst_coder_flow.run_flow()

                    elif FLOW == "ACT":

                        print("Running flow:")
                        print(FLOW)

                        ### 3. Analyst Coder Tester Flow (ACT)_______________
                        analyst_coder_tester_flow = Analyst_Coder_Tester_Flow(CODER_MAIN=coder_agent, CODER_IMPROVER = coder_improver, ANALYST = analyst_agent, TESTER=tester,
                                    requirement=intent, method_name=method_name, test=test, OUTPUT_PATH = OUTPUT_PATH, task=task)
                        analyst_coder_tester_flow.run_flow()

                    elif FLOW == "debugger":

                        print("Running flow:")
                        print(FLOW)

                        ### 4. Degugger Only Flow (DB)_______________
                        debugger_only_flow = Debugger_Only_Flow(CODER_MAIN=basic_coder_agent, CODER_IMPROVER = coder_improver, ANALYST = analyst_agent, TESTER=tester,
                                    requirement=intent, 
                                    method_name=method_name, 
                                    test=test, 
                                    OUTPUT_PATH = OUTPUT_PATH, 
                                    task_id=task['task_id'],

                                    provider=provider,
                                    model=model_arg,
                                    API_KEY=API_KEY,
                                    )
                        debugger_only_flow.run_flow()

                    elif FLOW == "ac_debugger":

                        print("Running flow:")
                        print(FLOW)


                        ### 5. AC + Degugger Flow (DB)_______________
                        ac_debug_flow = AC_Debug_Flow(CODER_MAIN=basic_coder_agent, CODER_IMPROVER = coder_improver, ANALYST = analyst_agent, TESTER=tester,
                                    requirement=intent, 
                                    method_name=method_name, 
                                    test=test, 
                                    OUTPUT_PATH = OUTPUT_PATH, 
                                    task_id=task['task_id'],

                                    provider=provider,
                                    model=model_arg,
                                    API_KEY=API_KEY,
                                    )
                        ac_debug_flow.run_flow()

                    elif FLOW == "act_debugger":

                        print("Running flow:")
                        print(FLOW)


                        ### 6. LLM Degugger Flow (LDB) (ACT + Debugger)_______________
                        analyst_coder_tester_debugger_flow = LDB_Flow(CODER_MAIN=coder_agent, CODER_IMPROVER = coder_improver, ANALYST = analyst_agent, TESTER=tester,
                                    requirement=intent, 
                                    method_name=method_name, 
                                    test=test, 
                                    OUTPUT_PATH = OUTPUT_PATH, 
                                    task_id=task['task_id'],

                                    provider=provider,
                                    model=model_arg,
                                    API_KEY=API_KEY,


                                    )
                        analyst_coder_tester_debugger_flow.run_flow()

                    #________________________________________________________________#________________________________________________________________
                    #________________________________________________________________#________________________________________________________________
                    #__________________________________________________________END OF FLOWS___________________________________________________________

                except RuntimeError as e:
                    pass
                    print("Errored out at main.py, runtime error")
                    continue
