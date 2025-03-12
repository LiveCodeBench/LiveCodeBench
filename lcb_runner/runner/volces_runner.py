import os

try:
    import openai
    from openai import OpenAI
except ImportError:
    pass

from lcb_runner.runner.siliconflow_runner import SiliconFlowRunner


class VolcesRunner(SiliconFlowRunner):
    client = OpenAI(
        api_key=os.getenv("API_KEY"),
        base_url="https://ark.cn-beijing.volces.com/api/v3/",
    )
