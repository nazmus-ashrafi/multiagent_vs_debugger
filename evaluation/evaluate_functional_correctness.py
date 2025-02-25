import fire
import sys

from data import HUMAN_EVAL
from data import HUMAN_EVAL_PLUS

from evaluation import evaluate_functional_correctness


def entry_point(
        
    sample_file: str,
    ## SWITCH between HumanEval datasets used for evaluation here: _______________________________________________________
    ## ___________________________________________________________________________________________________________________
    # problem_file: str = HUMAN_EVAL,
    # problem_file: str = HUMAN_EVAL_PLUS,
    problem_file: str,


    k: str = "1,10,100",
    n_workers: int = 4,
    timeout: float = 3.0,

    
    

    ## ___________________________________________________________________________________________________________________
    ## ___________________________________________________________________________________________________________________

):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """
    k = list(map(int, k.split(",")))
    results = evaluate_functional_correctness(sample_file, k, n_workers, timeout, problem_file)
    print(results)


def main():
    fire.Fire(entry_point)


# sys.exit(main())

if __name__ == '__main__':
    sys.exit(main())
    
