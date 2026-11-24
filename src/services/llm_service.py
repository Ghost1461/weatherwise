import requests


class LLMService:
    def __init__(self):
        self.url = "http://localhost:11434/api/generate"
        self.model = "phi3:mini"

    def generate_text(self, prompt: str) -> str:
        try:
            response = requests.post(
                self.url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": 140,
                        "temperature": 0.4
                    }
                },
                timeout=90
            )

            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip() or "Could not generate advice."

        except Exception as e:
            print("OLLAMA ERROR:", repr(e))
            return "Could not generate advice."