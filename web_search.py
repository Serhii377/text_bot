from typing import Final
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS


class WebSearch:
    REGION: Final[str] = "ua-uk"
    SAFESEARCH: Final[str] = "moderate"
    MAX_RESULTS: Final[int] = 3
    REQUEST_TIMEOUT: Final[int] = 10
    MAX_CONTENT_LENGTH: Final[int] = 3000
    USER_AGENT: Final[str] = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )

    def search(self, query: str) -> list[dict[str, str]]:
        if not query or not query.strip():
            return []

        results_list: list[dict[str, str]] = []

        try:
            with DDGS() as ddgs:
                search_results = ddgs.text(
                    query,
                    region=self.REGION,
                    safesearch=self.SAFESEARCH,
                    max_results=self.MAX_RESULTS
                )

                if not search_results:
                    return []

                for item in search_results:
                    results_list.append({
                        "title": item.get("title", "Без назви"),
                        "url": item.get("href", ""),
                        "snippet": item.get("body", "")
                    })
        except Exception:
            return []

        return results_list

    def _is_valid_url(self, url: str) -> bool:
        try:
            parsed = urlparse(url)
            return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
        except Exception:
            return False

    def read_page(self, url: str) -> str:
        if not url or not self._is_valid_url(url):
            return ""

        headers = {"User-Agent": self.USER_AGENT}

        try:
            response = requests.get(url, headers=headers, timeout=self.REQUEST_TIMEOUT)
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "").lower()
            if "text/html" not in content_type:
                return ""

            soup = BeautifulSoup(response.text, "html.parser")

            unwanted_tags = [
                "script", "style", "nav", "footer", "header",
                "aside", "form", "noscript", "iframe", "svg"
            ]
            for tag in soup(unwanted_tags):
                tag.decompose()

            raw_text = soup.get_text(separator=" ", strip=True)
            clean_text = " ".join(raw_text.split())

            return clean_text[:self.MAX_CONTENT_LENGTH]

        except (requests.RequestException, Exception):
            return ""


if __name__ == "__main__":
    web_search = WebSearch()
    test_query = "Що таке Python мова програмування"

    print(f"Пошук за запитом: '{test_query}'...\n")
    found_results = web_search.search(test_query)

    if not found_results:
        print("За запитом нічого не знайдено.")
    else:
        for idx, res in enumerate(found_results, start=1):
            print(f"{idx}. {res['title']} -> {res['url']}")

        target_url = found_results[0]["url"]
        print(f"\nЗавантаження вмісту сторінки: {target_url}\n")
        page_text = web_search.read_page(target_url)

        if page_text:
            print("Отриманий текст сторінки:")
            print("-" * 60)
            print(page_text)
            print("-" * 60)
        else:
            print("Не вдалося отримати текст сторінки.")