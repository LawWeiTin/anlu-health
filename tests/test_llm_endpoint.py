import httpx

from app.config import Settings
from app.llm import OpenAICompatibleModel, endpoint_messages


def endpoint_settings() -> Settings:
    return Settings(
        _env_file=None,
        model_provider="openai_compatible",
        model_api_url="https://private-model.example/v1",
        model_api_token="scoped-test-token",
        model_name="lawwt/anlu-health-medgemma-v32-adapter",
    )


def test_private_endpoint_retries_scale_to_zero_warmup(
    monkeypatch,
) -> None:
    statuses = iter((502, 503, 200))
    sleeps: list[float] = []

    def fake_post(*args, **kwargs) -> httpx.Response:  # type: ignore[no-untyped-def]
        del args, kwargs
        status = next(statuses)
        request = httpx.Request("POST", "https://private-model.example/v1/chat/completions")
        if status == 200:
            return httpx.Response(
                status,
                request=request,
                json={"choices": [{"message": {"content": "Verified answer"}}]},
            )
        return httpx.Response(status, request=request)

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr("app.llm.time.sleep", sleeps.append)

    answer = OpenAICompatibleModel(endpoint_settings()).generate("system", "user")

    assert answer == "Verified answer"
    assert sleeps == [1.0, 2.0]


def test_private_endpoint_honors_bounded_retry_after(monkeypatch) -> None:
    statuses = iter((503, 200))
    sleeps: list[float] = []

    def fake_post(*args, **kwargs) -> httpx.Response:  # type: ignore[no-untyped-def]
        del args, kwargs
        status = next(statuses)
        request = httpx.Request("POST", "https://private-model.example/v1/chat/completions")
        if status == 200:
            return httpx.Response(
                status,
                request=request,
                json={"choices": [{"message": {"content": "Ready"}}]},
            )
        return httpx.Response(status, request=request, headers={"Retry-After": "30"})

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr("app.llm.time.sleep", sleeps.append)

    answer = OpenAICompatibleModel(endpoint_settings()).generate("system", "user")

    assert answer == "Ready"
    assert sleeps == [15.0]


def test_private_endpoint_cold_start_retry_budget_is_bounded() -> None:
    assert OpenAICompatibleModel._MAX_ATTEMPTS == 12
    delays = [
        OpenAICompatibleModel._retry_delay(
            httpx.Response(
                503,
                request=httpx.Request(
                    "POST", "https://private-model.example/v1/chat/completions"
                ),
            ),
            retry_number,
        )
        for retry_number in range(1, OpenAICompatibleModel._MAX_ATTEMPTS)
    ]

    assert sum(delays) == 85.0
    assert max(delays) == 10.0


def test_v32_endpoint_uses_training_aligned_evidence_labels() -> None:
    system, user = endpoint_messages(
        "anlu-v32",
        "generic system prompt",
        """<care_mode>biomedical</care_mode>
<minimum_urgency>soon</minimum_urgency>
<safety_flags>lump_or_swelling</safety_flags>
<question>A lump has stayed for 12 days.</question>
<approved_sources>
[S1] Skin lumps
Excerpt: Track growth and arrange an examination.
</approved_sources>""",
    )

    assert "180-280 words" in system
    assert "3-6 possible causes" in system
    assert "practical meal or snack examples" in system
    assert "<question>" not in user
    assert "Supplied source records:" in user
    assert "User issue: A lump has stayed for 12 days." in user
    assert "[S1] Skin lumps" in user
    assert '"The supplied source is relevant to"' in user
    assert "cite every source used" in user
