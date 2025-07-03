from .base_agent import BaseAgent
import requests

# Agent that interprets model predictions using a local LLM
# Currently supports: GatorTron (default)

class LLMInterpreterAgent(BaseAgent):

    def __init__(self, model_backend="gatortron"):
        super().__init__("LLMInterpreterAgent")
        self.model_backend = model_backend.lower()
        self.available_models = {
            "gatortron": self._run_gatortron
            # Add more models like "model": self._run_model
        }

        if self.model_backend not in self.available_models:
            raise ValueError(f"Unsupported LLM backend: {self.model_backend}")

    # input_data should contain:
    # - prediction: str (predicted class or label)
    # - confidence: float
    # - clinical_data: dict (optional, for context)
    def run(self, input_data: dict) -> dict:

        return self.available_models[self.model_backend](input_data)

    def _run_gatortron(self, input_data: dict) -> dict:
        prediction = input_data["prediction"]
        confidence = input_data["confidence"]
        clinical_data = input_data.get("clinical_data", {})

        # Format prompt
        clinical_items = "\n".join([f"- {k}: {v}" for k, v in clinical_data.items()])
        prompt = (
            f"A liver disease prediction model has classified the patient as '{prediction}' "
            f"with {confidence*100:.1f}% confidence.\n\n"
            f"Clinical features:\n{clinical_items}\n\n"
            "Explain why this prediction might have been made."
        )

        # Call local server
        try:
            response = requests.post("http://localhost:8000/generate", json={"prompt": prompt})
            response.raise_for_status()
            explanation = response.json()["response"]
        except Exception as e:
            explanation = f"[GatorTron ERROR] Failed to call model: {str(e)}"

        return {"explanation": explanation, "model_used": "GatorTron"}
