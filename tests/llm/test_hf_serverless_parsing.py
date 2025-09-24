from oracle.llm.hf_serverless import HuggingFaceServerlessProvider


FAKE_RESP = {
    "generated_text": "e4",
    "details": {
        "prefill": [{"id": 1, "text": "..."}],
        "tokens": [
            {
                "id": 99,
                "text": "e",
                "logprob": -0.05,
                "top_tokens": [
                    {"token": "e", "logprob": -0.05},
                    {"token": "d", "logprob": -0.25},
                ],
            }
        ],
    },
}


class DummyClient:
    def __init__(self, response):
        self.response = response

    def text_generation(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return self.response


def test_hf_serverless_parses_top_tokens():
    provider = HuggingFaceServerlessProvider(
        model_id="primary", client=DummyClient(FAKE_RESP), top_n_tokens=5
    )

    results = provider.get_top_sequences(
        "Prompt", ["e4"], depth=1, prob_threshold=0.0, top_k=5
    )

    assert results, "Expected at least one token from parsing response"
    for token, logprob in results:
        assert isinstance(token, str)
        assert isinstance(logprob, float)
