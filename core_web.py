from typing import Final
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from web_search import WebSearch


class ConversationHistory:
    MAX_HISTORY_MESSAGES: Final[int] = 10
    START_HISTORY_MESSAGES: Final[int] = 2
    END_HISTORY_MESSAGES: Final[int] = 6

    def __init__(self, system_prompt_path: str = "system_prompt_web.txt") -> None:
        self.system_prompt: str = self._load_text_file(
            system_prompt_path,
            "Ти – AI-асистент, який відповідає українською мовою."
        )
        self.messages: list[dict[str, str]] = []

    @staticmethod
    def _load_text_file(path: str, fallback: str = "") -> str:
        try:
            with open(path, "r", encoding="utf-8") as file:
                return file.read().strip()
        except FileNotFoundError:
            return fallback

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


class BotWeb:
    def __init__(self, model_name: str = "google/gemma-3-4b-it") -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Використовується пристрій: {self.device}")
        print(f"Завантаження моделі '{model_name}'...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype
        )

        self.model.to(self.device)
        self.model.eval()

        self.history = ConversationHistory("system_prompt_web.txt")
        self.web_prompt_instruction = ConversationHistory._load_text_file(
            "web_prompt.txt",
            "Використай наведені вебматеріали для відповіді на запитання. Додавай URL джерел."
        )
        self.search_engine = WebSearch()
        print("Модель успішно завантажено та готова до роботи з вебпошуком!\n")

    def _build_web_context(self, user_text: str) -> str:
        search_results = self.search_engine.search(user_text)

        if not search_results:
            return (
                f"Запитання користувача: {user_text}\n\n"
                f"{self.web_prompt_instruction}\n"
                "Вебматеріали: Пошук в інтернеті не дав результатів."
            )

        materials: list[str] = []
        for idx, item in enumerate(search_results, start=1):
            url = item["url"]
            title = item["title"]
            snippet = item["snippet"]

            page_content = self.search_engine.read_page(url)
            content_block = page_content if page_content else snippet

            materials.append(
                f"Джерело {idx}: {title}\n"
                f"URL: {url}\n"
                f"Вміст: {content_block}"
            )

        joined_materials = "\n\n---\n\n".join(materials)

        return (
            f"Запитання користувача: {user_text}\n\n"
            f"{self.web_prompt_instruction}\n\n"
            f"=== ВЕБМАТЕРІАЛИ ===\n{joined_materials}\n=== КІНЕЦЬ ВЕБМАТЕРІАЛІВ ==="
        )

    def generate_response(self, user_text: str) -> str:
        enriched_user_message = self._build_web_context(user_text)
        self.history.add_request(enriched_user_message)
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
                max_new_tokens=400,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        input_length = model_inputs.input_ids.shape[1]
        response_ids = generated_ids[0][input_length:]
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()

        self.history.add_response(response_text)
        return response_text