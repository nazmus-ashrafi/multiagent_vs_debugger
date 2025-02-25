import json
import re
import ast
import time
import difflib
import copy
from pprint import pprint as pp


def code_truncate_regex(code):
    code_regex = r"```(.*?|)\n(?P<code>.*?)```"
    match = re.search(code_regex, code, re.DOTALL)
    code = match.group("code") if match else ""
    return code
    
def code_truncate(response):
    code = code_truncate_regex(response)
    if code == "":
        generation = response[response.find("def"):]
        tem = [s for s in generation.split('\n\n') if 'def ' in s or s[:1] == ' ']
        code = '\n\n'.join(tem).strip('```').strip()
    return code

# EXAMPLE PROMPT
# from typing import List def has_close_elements(numbers: List[float], threshold: float) -> 
# bool: """ Check if in given list of numbers, are any two numbers closer to each other than given threshold. 
# >>> has_close_elements([1.0, 2.0, 3.0], 0.5) False >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) True """

# Parse and extract different components from a given code prompt
def prompt_split_humaneval(prompt, mehotd_name):

    # Clean up the prompt
    prompt = prompt.strip()
    prompt = prompt.replace("\r\n", "\n")

    # Extracts everything before the last occurrence of the function definition (def), which could include imports
    before_func = prompt[:prompt.rfind("def ")]

    # Captures everything starting from the last function definition def onwards.
    code = prompt[prompt.rfind("def "):]

    # Searches for the starting location of the docstring, which can be enclosed in triple double quotes (""") or triple single quotes (''')
    comment_start_1 = re.search("\"\"\"", code)
    comment_start_2 = re.search("\'\'\'", code)
    if comment_start_1:
        comment_start = comment_start_1.end()
    elif comment_start_2:
        comment_start = comment_start_2.end()

    # •	example_start_1: Looks for variations of the word “Example”.
	# •	example_start_2: Looks for potential typos like “For Example”.
	# •	example_start_3: Searches for Python’s interactive shell prompt (>>>), often used to indicate examples.
	# •	example_start_4: Searches for function calls using the method name (as given by mehotd_name).

    example_start_1 = re.search("[eE]xample(:)?", code)
    example_start_2 = re.search("[fF]or [eE]xamble(:)?", code)
    example_start_3 = re.search(">>>", code)
    example_start_4 = re.search(mehotd_name+"\(.+\)", code[comment_start:])


    if example_start_1:
        comment = code[comment_start:example_start_1.start()]
        example = code[example_start_1.start():-4]
    elif example_start_2:
        comment = code[comment_start:example_start_2.start()]
        example = code[example_start_2.start():-4]
    elif example_start_3:
        comment = code[comment_start:example_start_3.start()]
        example = "Example:\n"+code[example_start_3.start():-4]
    elif example_start_4:
        comment = code[comment_start:example_start_4.start()+comment_start]
        example = "Example:\n"+code[example_start_4.start()+comment_start:-4]
    else:
        comment = code[comment_start:-4]
        example = ""

    # OUR EXTRACTIONS

    # Removes extra spaces and line breaks from the comment, ensuring it’s a clean single-line string.
    comment = comment.strip().replace("\n", " ")
    comment = re.sub("\s+", " ", comment)
    # Formats the example to have proper indentation.
    example = re.sub("\n(\s)*","\n\t",example)
    test_case = "\t"+example.strip()
    # the first line of the def function statement
    signature = code[:code.index("\n")+1]

    # RETURNING THE EXTRACTIONS
    return before_func, signature, comment, test_case

# Example run
# run_output = prompt_split_humaneval(
#     """from typing import List

# def has_close_elements(numbers: List[float], threshold: float) -> bool:
#     \"""
#     Check if in given list of numbers, are any two numbers closer to each other than given threshold.

#     >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
#     False

#     >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
#     True
#     \"""
# """,
# "has_close_elements"
# )

# pp(run_output)




def build_test_method(test_list, test_imports, method_name):
    if test_imports:
        test_imports = "\n".join(test_imports)
        test_method = test_imports + "\n"
    else:
        test_method = ""
    test_method = "def check(" + method_name + "):\n"
    if len(test_list) == 0:
        return test_method + "\treturn True" + "\n"
    for test in test_list:
        test_method += '\t' + test + "\n"
    return test_method.strip("\n")

def find_method_name(code, lang="python"):
    try:
        parsed = ast.parse(code)
        function_defs = [node for node in parsed.body if isinstance(node, ast.FunctionDef)]
        if function_defs:
            if len(function_defs) == 1:
                method_name = function_defs[0].name
            else:
                method_name = function_defs[-1].name if function_defs[-1].name != "main" else function_defs[-2].name
        else:
            method_name = None
    except:
        method_name = None

    return method_name


def code_split(func):
    '''
    Split code into signature, comment and function body
    '''
    func = func.replace("\r\n", "\n")
    before_func = func[:func.rfind("def ")]
    code = func[func.rfind("def "):]

    is_comment = False
    comments = []
    
    statements = code.split("\n")
    for s_idx, s in enumerate(statements):
        s = s.strip()
        if s.startswith("def"):
            signature = statements[:s_idx+1]
            method_name = s.split("def ")[1].split("(")[0]
            func_body_idx = s_idx+1
            tmp_statement = statements[func_body_idx].strip()
            if not tmp_statement.startswith("'''"):
                break
        elif s.startswith("'''") and not is_comment:
            is_comment = True

        elif is_comment:
            if s.startswith("'''"):
                is_comment = False
                func_body_idx = s_idx+1
                break
            comments.append(s)
    func_body = statements[func_body_idx:]
    return method_name, "\n".join(signature), "\n".join(comments), "\n".join(func_body), before_func

def construct_system_message(requirement, role, team=''):
    if team == '':
        system_message = "The requirement from users is: \n{'requirement':\n"  +  "'"+ requirement.replace('\n\n','\n').strip(".") + "'\n}\n\n" + role
    else:
        system_message = team + '\n '+ \
                    "The requirement from users is: \n{'requirement':\n"  +  "'"+ requirement.replace('\n\n','\n').strip(".") + "'\n}\n\n" + \
                    role
                
    return system_message


# Helper function to change "MBBP test_list format" to "Human Eval test format"
def mbpp_test_formatter(test_cases_array, new_name="candidate"):
    """
    Convert an array of test case strings into a function containing those assertions,
    automatically detecting the original function name and replacing it
    
    Args:
        test_cases_array (str): A string containing an array of test case assertions
        new_name (str): The new function name to use (default: "candidate")
        
    Returns:
        tuple: A tuple containing (function_str, original_function_name)
    """
    # Clean up the input string to get individual test cases
    test_cases = test_cases_array.strip('[]"\\ ').split('", "')
    
    # Extract the original function name from the first test case
    # Look for a word followed by an opening bracket
    import re
    first_test = test_cases[0]
    match = re.search(r'assert\s+(\w+)\(', first_test)
    old_name = match.group(1) if match else None
    
    if not old_name:
        raise ValueError("Could not detect original function name in test cases")
    
    # Create the function definition
    function_str = f"def check({new_name}):\n"
    
    # Add each test case as an assertion
    for test_case in test_cases:
        # Clean up the test case, replace function name, and add proper indentation
        test_case = test_case.strip().replace(old_name, new_name)
        function_str += f"    {test_case}\n"
        
    return function_str, old_name
    

# # Example usage
# test_array = '''[ "assert first_repeated_char(\"abcabc\") == \"a\"", "assert first_repeated_char(\"abc\") == \"None\"", "assert first_repeated_char(\"123123\") == \"1\"" ]'''

# # Convert array to function and get original name
# check_function, original_name = mbpp_test_formattern(test_array)

# # Print the results
# print("Generated function:")
# print(check_function)
# print("\nOriginal function name:", original_name)