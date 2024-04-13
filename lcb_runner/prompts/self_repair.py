import json

from anthropic import HUMAN_PROMPT, AI_PROMPT

from lcb_runner.lm_styles import LMStyle

class PromptConstants:
    SYSTEM_MESSAGE_GENERIC = f"You are a helpful programming assistant and an expert Python programmer. You are helping a user write a program to solve a problem. The user has written some code, but it has some errors and is not passing the tests. You will help the user by first giving a concise (at most 2-3 sentences) textual explanation of what is wrong with the code. After you have pointed out what is wrong with the code, you will then generate a fixed version of the program. You must put the entired fixed program within code delimiters only for once."

    SYSTEM_MESSAGE_DEEPSEEK = f"You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you are helping a user correct a error program for code competition. The user has written some code, but it has some errors and is not passing the tests. You will help the user by first giving a concise (at most 2-3 sentences) textual explanation of what is wrong with the code. After you have pointed out what is wrong with the code, you will then generate a fixed version of the entire executable program. You must put the entire fixed executable program within code delimiters."

    SYSTEM_MESSAGE_MAGIC = f"You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.\n\n@@ Instruction\n"

    SYSTEM_MESSAGE_WIZARD = "Below is an instruction that describes a task. Write a response that appropriately completes the request."

    SYSTEM_MESSAGE_PHIND = f"""You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program. You must put the entired fixed program within code delimiters only for once., for example: 
```python 
# YOUR CODE HERE
```"""
    FORMATTING_MESSAGE = "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."

    FORMATTING_WITHOUT_STARTER_MESSAGE = "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows."

def truncate_io(io):
    if len(str(io)) > 200:
        io = str(io)[:200] + "...."
    return io

def get_check_prompt(question: str, result,metadata):
    result_by_test_case = result
    if "error_code:-2" in metadata:
        # time limit exceeded
        message = f"The above code is incorrect and exceeds the time limit.\n{metadata}\n"
    elif "error_code:-1" in metadata:
        # runtime error
        runtime_message = metadata
        message = f"The above code is incorrect and has a runtime error.\n{runtime_message}\n"
    else:
        message = f"The above code is incorrect and does not pass the testcase.\n{truncate_io(metadata)}\n"
    return message

def get_generic_question_template_answer(question: str,code, result,metadata):
    prompt = f"### Question:\n{question}\n\n"
    prompt += f"### Answer:\n```python\n{code}\n```\n\n"
    prompt += get_check_prompt(question, result,metadata)
    return prompt

def get_cllama_question_template_answer(question: str, code, result,metadata):
    prompt = f"### Question\n{question}\n\n"
    prompt += f"### Answer\n```python\n{code}\n```\n\n"
    prompt += get_check_prompt(question, result,metadata)
    return prompt

def get_deepseekcode_question_template_answer(question:str, code, result, metadata):
    prompt = f"### Instruction: You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program.\n\n"
    prompt += f"Question:\n{question}\n\n"
    prompt += f"### Response:\n```python\n{code}\n```\n\n"
    prompt += get_check_prompt(question, result,metadata)
    return prompt

def get_magicoder_question_template_answer(question: str, code, result, metadata):
    prompt = f"You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program.\n\n"
    prompt += f"Question:\n{question}\n\n"
    prompt += f"@@ Response \n```python\n{code}\n```\n\n"
    prompt += get_check_prompt(question, result,metadata)
    return prompt

def get_mixtral_question_template_answer(question:str, code, result, metadata):
    prompt = f"Question:\n"
    prompt += f"{question}\n\n"
    prompt += f"Answer:\n\n"
    prompt += f"```python\n\n{code}\n``\n\n"
    prompt += get_check_prompt(question, result,metadata)
    return prompt

def get_wizard_question_template_answer(question:str, code, result, metadata):
    prompt = f"""### Instruction: You are a helpful programming assistant and an expert Python programmer. You are helping a user write a program to solve a problem. The user has written some code, but it has some errors and is not passing the tests. You will help the user by first giving a concise (at most 2-3 sentences) textual explanation of what is wrong with the code. After you have pointed out what is wrong with the code, you will then generate a fixed version of the program. You must put the entired fixed program within code delimiters only for once., for example:
    ```python
    # YOUR CODE HERE
    ```
"""
    prompt += f"{question}\n\n"
    prompt += f"### Response:```python\n\n{code}\n```\n\n"
    prompt += get_check_prompt(question, result,metadata)
    return prompt

def get_phind_question_template_answer(question: str, code, result, metadata):
    prompt = f"{question}\n\n"
    prompt += f'```python\n{code}\n``` \n\n'
    prompt += get_check_prompt(question, result,metadata)
    prompt += f"\n\n### Assistant"
    return prompt


def format_prompt_self_repair(question: str, LanguageModelStyle: LMStyle, code, result,metadata) -> str:
    if result :
        # The code is accepted, no need to change anything.
        return ""
    if LanguageModelStyle == LMStyle.OpenAIChat:
        chat_messages = [
            {
                "role": "system",
                "content": PromptConstants.SYSTEM_MESSAGE_GENERIC,
            },
        ]
        chat_messages += [
            {
                "role": "user",
                "content": get_generic_question_template_answer(question, code ,result,metadata),
            },
        ]
        return chat_messages
    elif LanguageModelStyle == LMStyle.Anthropic:
        prompt = f"{HUMAN_PROMPT}\n{PromptConstants.SYSTEM_MESSAGE_GENERIC}\n\n{get_generic_question_template_answer(question, code, result, metadata).rstrip()}\n{AI_PROMPT}"
        return prompt
    elif LanguageModelStyle == LMStyle.Gemini:
        prompt = f"{PromptConstants.SYSTEM_MESSAGE_GENERIC}\n{get_generic_question_template_answer(question, code, result,metadata)}"
        return prompt
    elif LanguageModelStyle == LMStyle.DeepSeekCodeInstruct:
        prompt = f"{PromptConstants.SYSTEM_MESSAGE_DEEPSEEK}\n\n{get_deepseekcode_question_template_answer(question, code, result,metadata)}"
        return prompt
    elif LanguageModelStyle == LMStyle.MistralWeb:
        prompt = f"[INST] {PromptConstants.SYSTEM_MESSAGE_GENERIC}\n{get_mixtral_question_template_answer(question, code, result,metadata)}[/INST]\n\n### Response\n\n"
        return prompt
    elif LanguageModelStyle == LMStyle.CodeLLaMaInstruct:
        prompt = f"[INST] <<SYS>>\n{PromptConstants.SYSTEM_MESSAGE_GENERIC}\n<</SYS>>\n\n{get_cllama_question_template_answer(question, code, result,metadata)}\n[/INST]"
        return prompt
    elif LanguageModelStyle == LMStyle.MagiCoder:
        prompt = f"{PromptConstants.SYSTEM_MESSAGE_MAGIC}\n{get_magicoder_question_template_answer(question, code, result,metadata)}"
        return prompt
    elif LanguageModelStyle == LMStyle.WizardCoder:
        prompt = f"{PromptConstants.SYSTEM_MESSAGE_WIZARD}\n\n{get_wizard_question_template_answer(question, code, result,metadata)}"
        return prompt
    elif LanguageModelStyle == LMStyle.Phind:
        prompt = f"### System Prompt\n\n{PromptConstants.SYSTEM_MESSAGE_PHIND}\n\n### User Message\n\n{get_phind_question_template_answer(question, code, result,metadata)}"
        return prompt
    else:
        raise NotImplementedError(f"LanguageModelStyle {LanguageModelStyle} not implemented")

def extract_code(model_output: str, lmstyle : LMStyle):
    outputlines = model_output.split("\n")
    if lmstyle == LMStyle.CodeLLaMa:
        indexlines = [i for i, line in enumerate(outputlines) if "PYTHON]" in line]
    else:
        indexlines = [i for i, line in enumerate(outputlines) if "```" in line]
    if len(indexlines) < 2:
        return ""
    return "\n".join(outputlines[indexlines[0] + 1 : indexlines[1]])

def test():
    def write_str_or_json(prompt):
        if isinstance(prompt, str):
            fp.write(prompt)
        else:
            fp.write(json.dumps(prompt))
        return
    
    for lm_style in [LMStyle.OpenAIChat]:
        with open("output/GPT-3.5-Turbo-0125/Scenario.codegeneration_10_0.2_eval_all.json") as f:
            check_metadata = json.load(f)[0]
        checked_base_question_cotent = check_metadata["question_content"]
        checked_base_codes = check_metadata["code_list"][0]
        checked_base_results = check_metadata["graded_list"][0]
        checked_base_metadata = check_metadata["metadata"][0]
        leetcode_prompt = format_prompt_self_repair(checked_base_question_cotent, lm_style,checked_base_codes,checked_base_results,checked_base_metadata)

        with open(f"/tmp/leetcode_{lm_style}.txt", "w") as fp:
            write_str_or_json(leetcode_prompt)
    return

if __name__ == '__main__':
    test()