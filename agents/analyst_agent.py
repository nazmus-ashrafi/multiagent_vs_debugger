from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI


# ANALYST = '''I want you to act as a requirement analyst on our development team. Given a user requirement, your task is to analyze, decompose, and develop a high-level plan to guide our developer in writing programs. The plan should include the following information:
# 1. Decompose the requirement into several easy-to-solve subproblems that can be more easily implemented by the developer.
# 2. Develop a high-level plan that outlines the major steps of the program.
# Remember, your plan should be high-level and focused on guiding the developer in writing code, rather than providing implementation details.
# '''


class AnalystAgent:
    def __init__(self, model, max_tokens=512, temperature=0.0, top_p=1.0):
        # Initialize the LLM using Langchain's OpenAI integration
        self.llm = model

    def implement(self, function_description):
        """
        Takes the function definition and the relevant description, passes it to the model, and returns the result.
        
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
            response = chain.run({"function_description": function_description})
            
        except Exception as e:
            print(f"Error: {e}")
            return "Error occurred during LLM execution"
        
        return response
    
    def construct_prompt(self):
        """
        Constructs the prompt template for processing the function description.
        """
        # Prompt template to inject the function description and generate the assistant's response
        prompt_template = PromptTemplate(
            input_variables=["function_description"],
            template="""
            I want you to act as a requirement analyst on our development team. Given a user requirement, your task is to analyze, decompose, and develop a high-level plan to guide our developer in writing programs. The plan should include the following information:
            1. Decompose the requirement into several easy-to-solve subproblems that can be more easily implemented by the developer.
            2. Develop a high-level plan that outlines the major steps of the program.
            Remember, your plan should be high-level and focused on guiding the developer in writing code, rather than providing implementation details.
            
            User requirement:
            {function_description}
            """
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
    
    coder_agent = AnalystAgent(model=model)
    result = coder_agent.implement(function_description)
    print(result)

