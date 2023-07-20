from pandasai.callbacks.base import StdoutCallBack, BaseCallback
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
import pandas as pd
from pandasai.prompts.base import Prompt

llm = OpenAI(api_token="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")

df = pd.DataFrame({
    "country": ["United States", "United Kingdom", "France", "Germany", "Italy", "Spain", "Canada", "Australia", "Japan", "China"],
    "gdp": [19294482071552, 2891615567872, 2411255037952, 3435817336832, 1745433788416, 1181205135360, 1607402389504, 1490967855104, 4380756541440, 14631844184064],
    "happiness_index": [6.94, 7.16, 6.66, 7.07, 6.38, 6.4, 7.23, 7.22, 5.87, 5.12]
})


class Test(Prompt):
    text = """
This is a test prmpt {vals}"""
class Test2(Prompt):
    text = """
This is a test prmpt for first prompt {vals}"""

vals = {"vals" : "test"}
t = Test(**vals)
t2 = Test(**vals)
class MyCustomCallback(BaseCallback):
    def on_code(self, response: str):
        print("My custom call back")
        print(response)

m = MyCustomCallback()
p = PandasAI(llm=llm, non_default_prompts={"multiple_dataframes" : t2, "generate_python_code" : t}, enable_cache=False, callback=m)



print(p.run(df,prompt="give me all countries"))

