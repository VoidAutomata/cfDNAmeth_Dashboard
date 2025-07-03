import pandas as pd

class BaseAgent:
    def __init__(self, name: str):
        self.name = name

    def run(self, input_data: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError