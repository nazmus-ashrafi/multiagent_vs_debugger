from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI


# ANALYST = '''I want you to act as a requirement analyst on our development team. Given a user requirement, your task is to analyze, decompose, and develop a high-level plan to guide our developer in writing programs. The plan should include the following information:
# 1. Decompose the requirement into several easy-to-solve subproblems that can be more easily implemented by the developer.
# 2. Develop a high-level plan that outlines the major steps of the program.
# Remember, your plan should be high-level and focused on guiding the developer in writing code, rather than providing implementation details.
# '''


class TesterAgent:
    def __init__(self, model, max_tokens=512, temperature=0.0, top_p=1.0):
        # Initialize the LLM using Langchain's OpenAI integration
        self.llm = model

    def implement(self, generated_code, function_description):
        """
        Takes the generated code and function_description, passes it to the model, and returns the result.
        
        Args:
            function_description (str): The input string containing the function and description.
            
        Returns:
            str: Generated a response from the LLM (Breaks down requirements and creates a high level plan to guide coder)
        """
        try:
            # Create the prompt using the function description
            chain = LLMChain(
                llm=self.llm,
                prompt=self.construct_prompt()
            )

            # Run the chain with the provided function description
            response = chain.run({"function_description": function_description, "generated_code": generated_code})
            
        except Exception as e:
            print(f"Error: {e}")
            return "Error occurred during LLM execution"
        
        return response
    
    def construct_prompt(self):
        """
        Constructs the prompt template for processing the function description.
        """

        prompt_template = PromptTemplate(
            input_variables=["function_description", "generated_code"],
            template='''I want you to act as a tester in the team. You will receive the code written by the developer, and your job is to complete a report as follows:

"Code": {generated_code}
"Code Review": Evaluate the structure and syntax of the code to ensure that it conforms to the specifications of the programming language, that the APIs used are correct, and that the code does not contain syntax errors or logic holes.
"Code Description": Briefly describe what the code is supposed to do. This helps identify differences between the code implementation and the requirement.
"Satisfying the requirements": Ture or False. This indicates whether the code satisfies the requirement.
"Edge cases": Edge cases are scenarios where the code might not behave as expected or where inputs are at the extreme ends of what the code should handle.
"Conclusion": "Code Test Passed" or "Code Test Failed". This is a summary of the test results.
Return the final verdict at the end of your reply with verdict: "PASS" or "FAIL". Do not style this verdict.

Original user requirements:
{function_description}

Here is the code which was written by the developer to satisfy the requirements:
{generated_code}
'''
        )
        return prompt_template

# To run the agent
if __name__ == "__main__":
    generated_code = (
        """from typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i + 1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False"""
    )


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

    # model = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0)

    model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    
    coder_agent = TesterAgent(model=model)
    result = coder_agent.implement(generated_code, function_description)
    print(result)

