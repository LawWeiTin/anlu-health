import re
from abc import ABC, abstractmethod
from functools import lru_cache

import httpx

from app.config import Settings, get_settings


class ModelError(RuntimeError):
    pass


class ModelProvider(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError


class MockMedicalModel(ModelProvider):
    """Safe deterministic copy for local development; it contains no medical intelligence."""

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        del system_prompt
        question_match = re.search(r"<question>(.*?)</question>", user_prompt, re.S)
        question = question_match.group(1).casefold() if question_match else ""
        urgency_match = re.search(r"<minimum_urgency>(.*?)</minimum_urgency>", user_prompt)
        urgency = urgency_match.group(1) if urgency_match else "routine"
        integrative = "<care_mode>integrative</care_mode>" in user_prompt
        source_titles = re.findall(r"\[(S\d+)\] ([^\n]+)", user_prompt)

        def citation_for(*terms: str) -> str:
            for source_id, title in source_titles:
                if any(term in title.casefold() for term in terms):
                    return f" [{source_id}]"
            return ""

        general_citation = f" [{source_titles[0][0]}]" if source_titles else ""
        lump_citation = citation_for("lump", "swelling") or general_citation
        tcm_citation = citation_for("traditional chinese", "herb", "proprietary medicine")

        if any(term in question for term in ("lump", "mass", "swelling", "肿块", "结节")):
            answer = (
                "What to do now\n"
                f"The minimum urgency is {urgency}. Arrange an in-person examination; a text "
                "description cannot identify the cause of a lump. Seek faster care if it is rapidly "
                "growing, hard/fixed, in the breast or testicle, red/hot, or affects swallowing or "
                f"breathing.{lump_citation}\n\n"
                "What this may mean\n"
                "Lumps have many possible causes, and examination—sometimes imaging or sampling—is "
                "needed to distinguish them. I cannot tell whether yours is benign or serious."
                f"{lump_citation}\n\n"
                "What to watch\n"
                "Note the location, size, duration, growth, pain, redness/warmth, mobility, fever, night "
                "sweats, unexplained weight loss, and any swallowing or breathing difficulty."
            )
        else:
            answer = (
                "What to do now\n"
                "Use this as general information and arrange a clinician or pharmacist review when "
                f"symptoms are new, persistent, worsening, or medicine-related.{general_citation}\n\n"
                "What this may mean\n"
                "I cannot determine a diagnosis from this message. The source cards below are the "
                f"evidence available for this response.{general_citation}\n\n"
                "What to watch\n"
                "Seek urgent help for severe breathing difficulty, fainting, stroke-like symptoms, "
                "uncontrolled bleeding, or a severe allergic reaction."
            )

        if integrative:
            answer += (
                "\n\nTraditional Chinese medicine perspective\n"
                "A licensed TCM practitioner may describe symptoms using a traditional pattern "
                "framework. That framework is not a confirmed biomedical diagnosis. Do not start a "
                "personalized herbal formula without checking pregnancy, surgery plans, liver/kidney "
                "conditions, allergies, and medicine interactions with a clinician or pharmacist."
                f"{tcm_citation}"
            )
        answer += (
            "\n\nHelpful follow-up questions\n"
            "When did this start, and is it changing? What other symptoms are present? What medicines, "
            "supplements, or herbs do you take?"
        )
        return answer


class OpenAICompatibleModel(ModelProvider):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        headers = {"Content-Type": "application/json"}
        if self.settings.model_api_token:
            headers["Authorization"] = f"Bearer {self.settings.model_api_token}"
        url = self.settings.model_api_url.rstrip("/")
        if not url.endswith("/chat/completions"):
            url += "/chat/completions"
        payload = {
            "model": self.settings.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 800,
        }
        try:
            response = httpx.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.settings.model_timeout_seconds,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise ModelError("The medical information model is temporarily unavailable") from exc
        if not isinstance(content, str) or not content.strip():
            raise ModelError("The medical information model returned an empty response")
        return content.strip()


@lru_cache
def get_model_provider() -> ModelProvider:
    settings = get_settings()
    if settings.model_provider == "openai_compatible":
        return OpenAICompatibleModel(settings)
    return MockMedicalModel()
