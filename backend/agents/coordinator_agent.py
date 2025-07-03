from .base_agent import BaseAgent

# Coordinates multiple agents in a pipeline
# Example: Classifier -> LLM Interpreter
class CoordinatorAgent(BaseAgent):

    def __init__(self, classifier_agent: BaseAgent, llm_agent: BaseAgent):
        super().__init__("CoordinatorAgent")
        self.classifier_agent = classifier_agent
        self.llm_agent = llm_agent

    def run(self, input_data: dict) -> dict:

        # 1. Classification model
        # 2. Pass result to LLM for interpretation

        # Step 1: Prediction
        prediction_output = self.classifier_agent.run(input_data)

        # Step 2: Interpretation via LLM
        llm_input = {
            "prediction": prediction_output["prediction"],
            "confidence": prediction_output["confidence"],
            "clinical_data": input_data  # optional, for context
        }
        interpretation_output = self.llm_agent.run(llm_input)

        return {
            "prediction": prediction_output,
            "interpretation": interpretation_output
        }
