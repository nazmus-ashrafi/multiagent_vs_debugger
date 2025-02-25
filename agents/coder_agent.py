from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI


# Optimizer helpers

import os
import dspy
from random import sample
from dspy.datasets import DataLoader
import random


dl = DataLoader()

class CodeGenerationSignature(dspy.Signature):
    # The Prompt
    """You are an expert programming assistant. Complete the following task in Python. Remember to repeat all imports and function header. Do not include any comments. Return only the imports and the code.
    """
    # Type support for the prompt
    question: str = dspy.InputField(
        prefix="Task Description:",
        desc="task description to ask to generate code from",
        format=lambda x: x.strip(),
    )
        
    answer: str = dspy.OutputField(
        prefix="Answer Code:",
        desc="python code that satisfies the task",
    )

human_eval = dl.from_huggingface(
    "openai/openai_humaneval",
    split="test",
    input_keys=("prompt",),
)

# Set a random seed for reproducibility
random.seed(42)

human_eval_train = [
    dspy.Example(
        question=example.prompt, 
        answer=example.canonical_solution).with_inputs("question")
    for example in sample(human_eval, 100)
]
human_eval_test = [
    dspy.Example(
        question=example.prompt, 
        answer=example.canonical_solution).with_inputs("question")
    for example in sample(human_eval, 64)
]




# Agent

class CoderAgent:
    def __init__(self, model, max_tokens=512, temperature=0.0, top_p=1.0, flow="basic"):
        # Initialize the LLM using Langchain's OpenAI integration
        self.llm = model
        self.flow = flow


    # dspy.settings.configure(
    #     lm=dspy.OpenAI(
    #         model="gpt-4o-mini",
    #         api_key=os.getenv("OPENAI_API_KEY"),
    #         max_tokens=4000,
    #         temperature=0,
    #     )
    # )

    

    def implement(self, function_description, plan=None, report=None):
        """
        Takes the function definition and the relevant description, passes it to the model, and returns the result.
        OR
        Takes a report from the tester, passes it to the model, and returns the result.
        
        Args:
            function_description (str): The input string containing the function and description.
            
        Returns:
            str: Generated code or response from the LLM.
        """
        try:
            # Create the prompt using the function description
            chain = LLMChain(
                llm=self.llm,
                prompt=self.construct_prompt()
            )

            # Run the chain with the provided function description
            if plan:
                response = chain.run({
                    "function_description": function_description,
                    "plan": plan
                })

            elif report:
                response = chain.run({
                    "function_description": function_description,
                    "report": report
                })

            else:
                # If no plan is provided, only pass the function description
                response = chain.run({
                    "function_description": function_description
                })
            
        except Exception as e:
            print(f"Error: {e}")
            return "Error occurred during LLM execution"
        
        return response

    
    def construct_prompt(self):
        """
        Constructs the prompt template for processing the function description based on the flow.
        """
        if self.flow == "basic":
            template = """
                You are an expert programming assistant. Complete the following task in Python. Remember to repeat all imports and function header. Do not include any comments. Return only the imports and the code.

                {function_description}

                Do not include any comments. Return only the code.
                Do not start with "Here's the Python code that implements the function based on the provided user requirement and plan".
                Just answer with the code.
            """
        elif self.flow == "basic_optimized_few_shot":
            template = """
                You are an expert programming assistant. Complete the following task in Python. Remember to repeat all imports and function header. Do not include any comments. Return only the imports and the code.
                
                Task:
                {function_description}

                Here is an example:
                
                Example Task One:
                def strlen(string: str) -> int:
                \"\"\" Return length of given string
                >>> strlen('')
                0
                >>> strlen('abc')
                3
                "\"\"

                Python Program for Example Task One:
                def strlen(string: str) -> int:
                    return len(string)

            """

        elif self.flow == "analyst_coder_flow":
            template = '''
            I want you to act as a developer on our development team. You will receive user requirement and plans from a requirements analyst. Your job is to write code in Python that meets the requirements following the plan.
            Remember, do not need to explain the code you wrote. Remember to repeat all imports and function header. Do not include any comments. Return only the imports and the code.

            User requirement:
            {function_description}

            Plan:
            {plan}

            Do not include any comments. Return only the code.
            Do not start with "Here's the Python code that implements the function based on the provided user requirement and plan".
            Just answer with the code.
            '''

            #### This extra lines was used in "tester_coder_flow" template when used used "Claude Haiku" to ensure model gives code only.
            # Do not include any comments. Return only the code.
            # Do not start with "Here's the Python code that implements the function based on the provided user requirement and plan".
            # Just answer with the code.

        elif self.flow == "tester_coder_flow":
            template = '''
            I want you to act as a developer on our development team. You will receive test reports from a reviewer. Your job is to fix or improve the code based on the content of the report. Ensure that any changes made to the code do not introduce new bugs or negatively impact the performance of the code.
Remember, do not need to explain the code you wrote. Remember to repeat all imports and function header. Do not include any comments. Return only the imports and the code.

            User Requirement:
            {function_description}

            Report:
            {report}
            
            Do not include any comments. Return only the code.
            Do not start with "Here is the refined code.." or any such comments.
            Just answer with the code.
            '''

            #### This extra lines was used in "tester_coder_flow" template when used used "Claude Haiku" to ensure model gives code only.
            # Do not include any comments. Return only the code.
            # Do not start with "Here is the refined code.." or any such comments.
            # Just answer with the code.
        
        else:
            raise ValueError("Unsupported flow type")

        # Create and return the prompt template using the selected flow template
        prompt_template = PromptTemplate(
            input_variables=["function_description"],
            template=template
        )
        return prompt_template

# To run the agent
if __name__ == "__main__":
    function_description = (
        "'from typing import List\n\n',"
        "'def has_close_elements(numbers: List[float], threshold: float) -> bool:\n',"
        "'Check if in given list of numbers, are any two numbers closer to each other than given threshold.',"
        "'\tExample:\n',"
        "'\t>>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n',"
        "'\tFalse\n',"
        "'\t>>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n',"
        "'\tTrue'"
    )

    model = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0)
    
    coder_agent = CoderAgent(model=model)
    result = coder_agent.implement(function_description)
    print(result)


# from typing import List

# def has_close_elements(numbers: List[float], threshold: float) -> bool:
#     """
#     Check if in given list of numbers, are any two numbers closer to each other than given threshold.
    
#     Example:
#     >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
#     False
#     >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
#     True
#     """
#     for i in range(len(numbers)):
#         for j in range(i+1, len(numbers)):
#             if abs(numbers[i] - numbers[j]) < threshold:
#                 return True
#     return False