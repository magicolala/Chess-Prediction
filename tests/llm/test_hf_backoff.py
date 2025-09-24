from types import SimpleNamespace

import pytest

from oracle.llm.hf_serverless import HuggingFaceServerlessProvider


class DummyError(Exception):
    def __init__(self, status_code=None):
        super().__init__("boom")
        self.status_code = status_code


class FlakyClient:
    def __init__(self, results):
        self.results = results
        self.calls = 0

    def text_generation(self, *args, **kwargs):
        result = self.results[self.calls]
        self.calls += 1
        if isinstance(result, Exception):
            raise result
        return result


FAKE_SUCCESS = SimpleNamespace(
    generated_text="e4",
    details=SimpleNamespace(
        tokens=[
            SimpleNamespace(
                id=42,
                text="e",
                logprob=-0.05,
                top_tokens=[SimpleNamespace(token="e", logprob=-0.05)],
            )
        ],
        prefill=[],
    ),
)


def test_hf_serverless_retry_exponential_backoff():
    client = FlakyClient([DummyError(), DummyError(503), FAKE_SUCCESS])
    sleeps = []

    def fake_sleep(duration):
        sleeps.append(duration)

    provider = HuggingFaceServerlessProvider(
        model_id="mistral",
        client=client,
        top_n_tokens=5,
        max_retries=3,
        retry_base_delay=0.1,
        rate_limit_delay=0.3,
        sleep=fake_sleep,
    )

    results = provider.get_top_sequences("Prompt", ["e4"], depth=1, prob_threshold=0.0, top_k=3)

    assert results == [("e", -0.05)]
    assert sleeps == [0.1, 0.2]
    assert client.calls == 3


def test_hf_serverless_rate_limit_delay():
    client = FlakyClient([DummyError(429), DummyError(429), FAKE_SUCCESS])
    sleeps = []

    def fake_sleep(duration):
        sleeps.append(duration)

    provider = HuggingFaceServerlessProvider(
        model_id="mistral",
        client=client,
        top_n_tokens=5,
        max_retries=3,
        retry_base_delay=0.1,
        rate_limit_delay=0.5,
        sleep=fake_sleep,
    )

    results = provider.get_top_sequences("Prompt", ["e4"], depth=1, prob_threshold=0.0, top_k=3)

    assert results == [("e", -0.05)]
    assert sleeps == [0.5, 0.5]


def test_hf_serverless_raises_after_max_retries():
    client = FlakyClient([DummyError(429), DummyError(503), DummyError(400)])

    provider = HuggingFaceServerlessProvider(
        model_id="mistral",
        client=client,
        top_n_tokens=5,
        max_retries=3,
        retry_base_delay=0.01,
        rate_limit_delay=0.01,
        sleep=lambda _: None,
    )

    with pytest.raises(DummyError):
        provider.get_top_sequences("Prompt", ["e4"], depth=1, prob_threshold=0.0, top_k=2)
