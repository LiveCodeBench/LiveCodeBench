import os
import argparse
from enum import Enum

class Scenario(Enum):
    generation = "generation"
    repair = "repair"
    testoutput = "testoutput"
    execution = "execution"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo-0301")
    parser.add_argument("--scenario", type=Scenario, default=Scenario.generation)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=1200)
    parser.add_argument("--multiprocess", default=0, type=int)
    parser.add_argument("--timeout", default=60, type=int)
    parser.add_argument("--stop", default="###", type=str)
    parser.add_argument("--continue_existing", action="store_true")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    args.stop = args.stop.split(",")

    if args.multiprocess == -1:
        args.multiprocess = os.cpu_count()

    output_path = f"output/{args.model}/{args.scenario}/output_{args.n}_{args.temperature}.json" 
    args.output_path = output_path

    return args

def test():
    args = get_args()
    print(args)

if __name__ == "__main__":
    test()