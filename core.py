from typing import Final
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class ConversationHistory:
    MAX_HISTORY_MESSAGES: Final[int] = 10
    START_HISTORY_MESSAGES: Final[int] = 2
    END_HISTORY_MESSAGES: Final[int] = 6

    def __init__(self, system_prompt_path: str = "system_prompt.txt") -> None:
        self.system_prompt: str = self._load_system_prompt(system_prompt_path)
        self.messages: list[dict[str, str]] = []

    def _load_system_prompt(self, path: str) -> str:
        try:
            with open(path, "r", encoding="utf-8") as file:
                return file.read().strip()
        except FileNotFoundError:
            return "Ти — помічник із програмування Python. Відповідай українською мовою."

    def add_request(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})
        self._trim_history()

    def add_response(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})
        self._trim_history()

    def _trim_history(self) -> None:
        if len(self.messages) > self.MAX_HISTORY_MESSAGES:
            start_slice = self.messages[:self.START_HISTORY_MESSAGES]
            end_slice = self.messages[-self.END_HISTORY_MESSAGES:]
            self.messages = start_slice + end_slice

    def get_messages(self) -> list[dict[str, str]]:
        system_message = {"role": "system", "content": self.system_prompt}
        return [system_message] + self.messages


class Bot:
    def __init__(self, model_name: str = "Qwen/Qwen3-1.7B") -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Використовується пристрій: {self.device}")
        print(f"Завантаження моделі '{model_name}'...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        self.model.to(self.device)
        self.model.eval()

        self.history = ConversationHistory()
        print("Модель успішно завантажено та переведено в режим оцінювання!\n")

    def generate_response(self, user_text: str) -> str:
        self.history.add_request(user_text)
        messages = self.history.get_messages()

        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = self.tokenizer([prompt_text], return_tensors="pt").to(self.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=300,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        input_length = model_inputs.input_ids.shape[1]
        response_ids = generated_ids[0][input_length:]
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()

        self.history.add_response(response_text)
        return response_text