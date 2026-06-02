"""Unit tests for LLM service."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from services import llm_service


pytestmark = pytest.mark.unit


class _DummyMessage:
    def __init__(self, content: str):
        self.content = content
        self.tool_calls = None


class _DummyChoice:
    def __init__(self, message):
        self.message = message


class _DummyResponse:
    def __init__(self, content: str, total_cost: float = 0.0):
        self.choices = [_DummyChoice(_DummyMessage(content))]
        self.usage = SimpleNamespace(total_cost=total_cost)


@pytest.mark.asyncio
async def test_call_llm_acompletion_success(monkeypatch):
    async def fake_acompletion(**_kwargs):
        return _DummyResponse("hello", total_cost=1.23)

    monkeypatch.setattr(llm_service.litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(llm_service, "get_observability_service", lambda: Mock())

    user_service = Mock()

    content, thoughts = await llm_service.call_llm(
        provider="openai",
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
        parameters={},
        user_service=user_service,
        user_id="user-1",
    )

    assert content == "hello"
    assert thoughts is None
    user_service.require_allowance.assert_called_once_with("user-1", basic_info=None)
    user_service.add_usage.assert_called_once_with("user-1", 1.23, basic_info=None)


@pytest.mark.asyncio
async def test_call_llm_acompletion_error(monkeypatch):
    async def fake_acompletion(**_kwargs):
        raise RuntimeError("boom")

    observability = Mock()
    monkeypatch.setattr(llm_service.litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(llm_service, "get_observability_service", lambda: observability)

    user_service = Mock()

    with pytest.raises(RuntimeError, match="boom"):
        await llm_service.call_llm(
            provider="openai",
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
            parameters={},
            user_service=user_service,
            user_id="user-2",
        )

    observability.log_exception.assert_called_once()


class _DummyDelta:
    def __init__(self, content: str):
        self.content = content


class _DummyStreamChoice:
    def __init__(self, delta):
        self.delta = delta


class _DummyStreamChunk:
    def __init__(self, content: str):
        self.choices = [_DummyStreamChoice(_DummyDelta(content))]


@pytest.mark.asyncio
async def test_stream_llm_success(monkeypatch):
    """Test successful streaming LLM call."""
    chunks_to_yield = [
        _DummyStreamChunk("Hello"),
        _DummyStreamChunk(" "),
        _DummyStreamChunk("World"),
    ]

    # For streaming, litellm.acompletion returns an awaitable that resolves
    # to an async generator, so we need to return an awaitable
    async def async_gen():
        for chunk in chunks_to_yield:
            yield chunk

    async def fake_acompletion(**_kwargs):
        return async_gen()

    monkeypatch.setattr(llm_service.litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(llm_service, "get_observability_service", lambda: Mock())

    user_service = Mock()

    received_chunks = []
    async for chunk in llm_service.stream_llm(
        provider="openai",
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
        parameters={},
        user_service=user_service,
        user_id="user-1",
    ):
        received_chunks.append(chunk)

    assert len(received_chunks) == 3
    assert received_chunks[0].choices[0].delta.content == "Hello"
    assert received_chunks[1].choices[0].delta.content == " "
    assert received_chunks[2].choices[0].delta.content == "World"
    user_service.require_allowance.assert_called_once_with("user-1", basic_info=None)


@pytest.mark.asyncio
async def test_stream_llm_error(monkeypatch):
    """Test streaming LLM call with error."""
    async def fake_acompletion(**_kwargs):
        raise RuntimeError("stream error")

    observability = Mock()
    monkeypatch.setattr(llm_service.litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(llm_service, "get_observability_service", lambda: observability)

    user_service = Mock()

    with pytest.raises(RuntimeError, match="stream error"):
        async for _ in llm_service.stream_llm(
            provider="openai",
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
            parameters={},
            user_service=user_service,
            user_id="user-2",
        ):
            pass

    observability.log_exception.assert_called_once()


@pytest.mark.asyncio
async def test_stream_llm_filters_none_params(monkeypatch):
    """Test that stream_llm filters out None values from parameters."""
    call_kwargs = {}

    async def async_gen():
        yield _DummyStreamChunk("test")

    async def fake_acompletion(**kwargs):
        call_kwargs.update(kwargs)
        return async_gen()

    monkeypatch.setattr(llm_service.litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(llm_service, "get_observability_service", lambda: Mock())
    monkeypatch.setattr(llm_service, "get_user_service", lambda _: Mock())
    monkeypatch.setattr(llm_service, "get_database_service", lambda _: Mock())

    async for _ in llm_service.stream_llm(
        provider="openai",
        model="gpt-4",
        messages=[{"role": "user", "content": "test"}],
        parameters={"temperature": 0.7, "top_p": None, "max_tokens": 100},
    ):
        pass

    # None values should be filtered out
    assert "temperature" in call_kwargs
    assert call_kwargs["temperature"] == 0.7
    assert "top_p" not in call_kwargs
    assert "max_tokens" in call_kwargs
    assert call_kwargs["stream"] is True


class TestLLMServiceApiKeys:
    """Test suite for LLM service API key integration."""

    @pytest.mark.asyncio
    async def test_call_llm_uses_user_api_key_when_set(self, monkeypatch):
        """Test that user API key is passed to litellm when set."""
        call_kwargs = {}

        async def fake_acompletion(**kwargs):
            call_kwargs.update(kwargs)
            return _DummyResponse("hello", total_cost=1.0)

        monkeypatch.setattr(llm_service.litellm, "acompletion", fake_acompletion)
        monkeypatch.setattr(llm_service, "get_observability_service", lambda: Mock())

        user_service = Mock()
        user_service.get_effective_api_key.return_value = "user-specific-api-key"

        content, thoughts = await llm_service.call_llm(
            provider="openai",
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
            parameters={},
            user_service=user_service,
            user_id="user-1",
        )

        # Verify the user API key was passed to litellm
        assert "api_key" in call_kwargs
        assert call_kwargs["api_key"] == "user-specific-api-key"
        user_service.get_effective_api_key.assert_called_once_with("user-1", "openai", basic_info=None)

    @pytest.mark.asyncio
    async def test_call_llm_no_api_key_when_not_set(self, monkeypatch):
        """Test that no api_key is passed when user has no key set."""
        call_kwargs = {}

        async def fake_acompletion(**kwargs):
            call_kwargs.update(kwargs)
            return _DummyResponse("hello", total_cost=1.0)

        monkeypatch.setattr(llm_service.litellm, "acompletion", fake_acompletion)
        monkeypatch.setattr(llm_service, "get_observability_service", lambda: Mock())

        user_service = Mock()
        user_service.get_effective_api_key.return_value = None

        content, thoughts = await llm_service.call_llm(
            provider="openai",
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
            parameters={},
            user_service=user_service,
            user_id="user-1",
        )

        # Verify no api_key was passed (litellm will use env var)
        assert "api_key" not in call_kwargs

    @pytest.mark.asyncio
    async def test_stream_llm_uses_user_api_key_when_set(self, monkeypatch):
        """Test that user API key is used for streaming when set."""
        call_kwargs = {}

        async def async_gen():
            yield _DummyStreamChunk("test")

        async def fake_acompletion(**kwargs):
            call_kwargs.update(kwargs)
            return async_gen()

        monkeypatch.setattr(llm_service.litellm, "acompletion", fake_acompletion)
        monkeypatch.setattr(llm_service, "get_observability_service", lambda: Mock())

        user_service = Mock()
        user_service.get_effective_api_key.return_value = "user-streaming-key"

        async for _ in llm_service.stream_llm(
            provider="anthropic",
            model="claude-3",
            messages=[{"role": "user", "content": "hi"}],
            parameters={},
            user_service=user_service,
            user_id="user-2",
        ):
            pass

        # Verify the user API key was passed
        assert "api_key" in call_kwargs
        assert call_kwargs["api_key"] == "user-streaming-key"
        user_service.get_effective_api_key.assert_called_once_with("user-2", "anthropic", basic_info=None)

    @pytest.mark.asyncio
    async def test_stream_llm_no_api_key_when_not_set(self, monkeypatch):
        """Test that no api_key is passed for streaming when user has no key."""
        call_kwargs = {}

        async def async_gen():
            yield _DummyStreamChunk("test")

        async def fake_acompletion(**kwargs):
            call_kwargs.update(kwargs)
            return async_gen()

        monkeypatch.setattr(llm_service.litellm, "acompletion", fake_acompletion)
        monkeypatch.setattr(llm_service, "get_observability_service", lambda: Mock())

        user_service = Mock()
        user_service.get_effective_api_key.return_value = None

        async for _ in llm_service.stream_llm(
            provider="openai",
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            parameters={},
            user_service=user_service,
            user_id="user-3",
        ):
            pass

        # Verify no api_key was passed
        assert "api_key" not in call_kwargs

    @pytest.mark.asyncio
    async def test_call_llm_with_user_basic_info(self, monkeypatch):
        """Test that user_basic_info is passed to get_effective_api_key."""
        call_kwargs = {}

        async def fake_acompletion(**kwargs):
            call_kwargs.update(kwargs)
            return _DummyResponse("hello", total_cost=1.0)

        monkeypatch.setattr(llm_service.litellm, "acompletion", fake_acompletion)
        monkeypatch.setattr(llm_service, "get_observability_service", lambda: Mock())

        user_service = Mock()
        user_service.get_effective_api_key.return_value = "user-key"

        user_basic_info = {"email": "test@example.com", "name": "Test User"}

        await llm_service.call_llm(
            provider="openai",
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            parameters={},
            user_service=user_service,
            user_id="user-1",
            user_basic_info=user_basic_info,
        )

        # Verify user_basic_info was passed to get_effective_api_key
        user_service.get_effective_api_key.assert_called_once_with("user-1", "openai", basic_info=user_basic_info)

    @pytest.mark.asyncio
    async def test_call_llm_different_providers_use_different_keys(self, monkeypatch):
        """Test that different providers use their respective API keys."""
        call_log = []

        async def fake_acompletion(**kwargs):
            call_log.append({"provider": kwargs.get("model").split("/")[0], "api_key": kwargs.get("api_key")})
            return _DummyResponse("hello", total_cost=1.0)

        monkeypatch.setattr(llm_service.litellm, "acompletion", fake_acompletion)
        monkeypatch.setattr(llm_service, "get_observability_service", lambda: Mock())

        # Test multiple providers
        providers_and_keys = [
            ("openai", "gpt-4", "openai-user-key"),
            ("anthropic", "claude-3", "anthropic-user-key"),
            ("gemini", "gemini-pro", "gemini-user-key"),
        ]

        for provider, model, expected_key in providers_and_keys:
            user_service = Mock()
            user_service.get_effective_api_key.return_value = expected_key

            await llm_service.call_llm(
                provider=provider,
                model=model,
                messages=[{"role": "user", "content": "hi"}],
                parameters={},
                user_service=user_service,
                user_id="user-1",
            )

            # Verify get_effective_api_key was called with the correct provider
            user_service.get_effective_api_key.assert_called_once_with("user-1", provider, basic_info=None)

        # Verify all calls had their respective API keys
        for i, (provider, _, expected_key) in enumerate(providers_and_keys):
            assert call_log[i]["provider"] == provider
            assert call_log[i]["api_key"] == expected_key

    @pytest.mark.asyncio
    async def test_call_llm_empty_string_key_not_passed(self, monkeypatch):
        """Test that empty string API key is not passed to litellm (treated as no key)."""
        call_kwargs = {}

        async def fake_acompletion(**kwargs):
            call_kwargs.update(kwargs)
            return _DummyResponse("hello", total_cost=1.0)

        monkeypatch.setattr(llm_service.litellm, "acompletion", fake_acompletion)
        monkeypatch.setattr(llm_service, "get_observability_service", lambda: Mock())

        user_service = Mock()
        # Empty string is falsy, so the code shouldn't pass it
        user_service.get_effective_api_key.return_value = ""  # Empty string

        await llm_service.call_llm(
            provider="openai",
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            parameters={},
            user_service=user_service,
            user_id="user-1",
        )

        # Empty string is falsy in Python, so api_key should not be in kwargs
        # (the code checks `if api_key:` which is False for empty string)
        assert "api_key" not in call_kwargs


class TestBedrockProvider:
    """Test suite for Bedrock provider integration."""

    @pytest.mark.asyncio
    async def test_call_llm_bedrock_uses_api_key(self, monkeypatch):
        """Test that bedrock provider uses AWS_BEARER_TOKEN_BEDROCK via api_key."""
        call_kwargs = {}

        async def fake_acompletion(**kwargs):
            call_kwargs.update(kwargs)
            return _DummyResponse("hello from bedrock", total_cost=0.5)

        monkeypatch.setattr(llm_service.litellm, "acompletion", fake_acompletion)
        monkeypatch.setattr(llm_service, "get_observability_service", lambda: Mock())

        user_service = Mock()
        user_service.get_effective_api_key.return_value = "bedrock-api-key-12345"

        content, thoughts = await llm_service.call_llm(
            provider="bedrock",
            model="us.anthropic.claude-opus-4-5-20251101-v1:0",
            messages=[{"role": "user", "content": "hi"}],
            parameters={},
            user_service=user_service,
            user_id="user-1",
        )

        assert content == "hello from bedrock"
        assert "api_key" in call_kwargs
        assert call_kwargs["api_key"] == "bedrock-api-key-12345"
        assert call_kwargs["model"] == "bedrock/us.anthropic.claude-opus-4-5-20251101-v1:0"
        user_service.get_effective_api_key.assert_called_once_with("user-1", "bedrock", basic_info=None)

    @pytest.mark.asyncio
    async def test_call_llm_bedrock_model_string_format(self, monkeypatch):
        """Test that bedrock model string is correctly formatted as bedrock/model-id."""
        call_kwargs = {}

        async def fake_acompletion(**kwargs):
            call_kwargs.update(kwargs)
            return _DummyResponse("test", total_cost=0.1)

        monkeypatch.setattr(llm_service.litellm, "acompletion", fake_acompletion)
        monkeypatch.setattr(llm_service, "get_observability_service", lambda: Mock())

        user_service = Mock()
        user_service.get_effective_api_key.return_value = None

        await llm_service.call_llm(
            provider="bedrock",
            model="us.amazon.nova-pro-v1:0",
            messages=[{"role": "user", "content": "test"}],
            parameters={},
            user_service=user_service,
            user_id="user-1",
        )

        # Model string should be "bedrock/model-id" format for litellm
        assert call_kwargs["model"] == "bedrock/us.amazon.nova-pro-v1:0"

    @pytest.mark.asyncio
    async def test_stream_llm_bedrock_uses_api_key(self, monkeypatch):
        """Test that bedrock streaming uses the API key."""
        call_kwargs = {}

        async def async_gen():
            yield _DummyStreamChunk("bedrock stream")

        async def fake_acompletion(**kwargs):
            call_kwargs.update(kwargs)
            return async_gen()

        monkeypatch.setattr(llm_service.litellm, "acompletion", fake_acompletion)
        monkeypatch.setattr(llm_service, "get_observability_service", lambda: Mock())

        user_service = Mock()
        user_service.get_effective_api_key.return_value = "bedrock-stream-key"

        async for _ in llm_service.stream_llm(
            provider="bedrock",
            model="us.meta.llama4-maverick-17b-instruct-v1:0",
            messages=[{"role": "user", "content": "hi"}],
            parameters={},
            user_service=user_service,
            user_id="user-1",
        ):
            pass

        assert "api_key" in call_kwargs
        assert call_kwargs["api_key"] == "bedrock-stream-key"
        assert call_kwargs["model"] == "bedrock/us.meta.llama4-maverick-17b-instruct-v1:0"

    @pytest.mark.asyncio
    async def test_call_llm_bedrock_with_parameters(self, monkeypatch):
        """Test that bedrock calls pass through parameters correctly."""
        call_kwargs = {}

        async def fake_acompletion(**kwargs):
            call_kwargs.update(kwargs)
            return _DummyResponse("test", total_cost=0.1)

        monkeypatch.setattr(llm_service.litellm, "acompletion", fake_acompletion)
        monkeypatch.setattr(llm_service, "get_observability_service", lambda: Mock())

        user_service = Mock()
        user_service.get_effective_api_key.return_value = "bedrock-key"

        await llm_service.call_llm(
            provider="bedrock",
            model="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            messages=[{"role": "user", "content": "test"}],
            parameters={"temperature": 0.7, "max_tokens": 1000},
            user_service=user_service,
            user_id="user-1",
        )

        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_tokens"] == 1000

class TestIsOpus47OrLater:
    """Tests for the _is_opus_4_7_or_later helper."""

    @pytest.mark.parametrize("model", [
        "us.anthropic.claude-opus-4-7",
        "us.anthropic.claude-opus-4-8",
        # Tolerate either bare or prefixed model strings.
        "bedrock/us.anthropic.claude-opus-4-7",
        "bedrock/us.anthropic.claude-opus-4-8",
    ])
    def test_returns_true_for_opus_4_7_and_4_8(self, model):
        assert llm_service._is_opus_4_7_or_later(model) is True

    @pytest.mark.parametrize("model", [
        # Older Opus generations use the legacy thinking format.
        "us.anthropic.claude-opus-4-6-v1",
        "us.anthropic.claude-opus-4-5-20251101-v1:0",
        "us.anthropic.claude-opus-4-1-20250805-v1:0",
        # Non-Opus Claude models never go through the new format,
        # regardless of generation.
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        # Completely unrelated providers / models.
        "gpt-5",
        "",
    ])
    def test_returns_false_for_older_opus_and_non_opus(self, model):
        assert llm_service._is_opus_4_7_or_later(model) is False


class TestCallLlmAdaptiveThinkingKwargs:
    """Verify the kwargs that reach litellm.acompletion when reasoning_effort
    is set on Opus 4.7+ vs. older models. Locks in the parameter-shape
    contract that Bedrock's converse API enforces."""

    @pytest.fixture
    def captured(self):
        """Holds the kwargs the fake acompletion saw, for assertion."""
        return {}

    @pytest.fixture
    def patched_llm(self, monkeypatch, captured):
        async def capturing_acompletion(**kwargs):
            captured.update(kwargs)
            return _DummyResponse("ok", total_cost=0.0)
        monkeypatch.setattr(llm_service.litellm, "acompletion", capturing_acompletion)
        monkeypatch.setattr(llm_service, "get_observability_service", lambda: Mock())

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model,effort", [
        ("us.anthropic.claude-opus-4-7", "low"),
        ("us.anthropic.claude-opus-4-7", "medium"),
        ("us.anthropic.claude-opus-4-7", "high"),
        ("us.anthropic.claude-opus-4-7", "xhigh"),
        ("us.anthropic.claude-opus-4-7", "max"),
        ("us.anthropic.claude-opus-4-8", "high"),
    ])
    async def test_opus_4_7_plus_with_effort_uses_adaptive_thinking(
        self, patched_llm, captured, model, effort,
    ):
        """For Opus 4.7+, reasoning_effort must be translated into
        thinking={"type":"adaptive"} and output_config={"effort":...}
        before litellm sees the call. The legacy `reasoning_effort`
        kwarg must NOT be passed through, otherwise litellm's internal
        translation re-injects the rejected thinking.type.enabled shape."""
        await llm_service.call_llm(
            provider="bedrock",
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            parameters={"reasoning_effort": effort},
            user_service=Mock(),
            user_id="user-1",
        )
        assert captured.get("thinking") == {"type": "adaptive"}
        assert captured.get("output_config") == {"effort": effort}
        assert "reasoning_effort" not in captured, (
            "reasoning_effort must be stripped on 4.7+; if it leaks "
            "through, litellm re-translates it to thinking.type.enabled "
            "and Bedrock 400s."
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model", [
        "us.anthropic.claude-opus-4-6-v1",
        "us.anthropic.claude-opus-4-5-20251101-v1:0",
        "us.anthropic.claude-opus-4-1-20250805-v1:0",
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    ])
    async def test_older_models_pass_reasoning_effort_through(
        self, patched_llm, captured, model,
    ):
        """4.6 and earlier (and all non-Opus Claudes) keep the legacy
        path: reasoning_effort passes through to litellm unchanged, and
        no adaptive-thinking params are set."""
        await llm_service.call_llm(
            provider="bedrock",
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            parameters={"reasoning_effort": "high"},
            user_service=Mock(),
            user_id="user-1",
        )
        assert captured.get("reasoning_effort") == "high"
        assert "thinking" not in captured
        assert "output_config" not in captured

    @pytest.mark.asyncio
    async def test_opus_4_8_with_disable_sets_no_thinking_params(
        self, patched_llm, captured,
    ):
        """reasoning_effort='disable' must skip BOTH branches — no
        thinking dict, no output_config, no reasoning_effort kwarg.
        Otherwise the model thinks when the caller asked it not to."""
        await llm_service.call_llm(
            provider="bedrock",
            model="us.anthropic.claude-opus-4-8",
            messages=[{"role": "user", "content": "hi"}],
            parameters={"reasoning_effort": "disable"},
            user_service=Mock(),
            user_id="user-1",
        )
        assert "thinking" not in captured
        assert "output_config" not in captured
        assert "reasoning_effort" not in captured

    @pytest.mark.asyncio
    async def test_opus_4_8_without_reasoning_effort_parameter(
        self, patched_llm, captured,
    ):
        """When reasoning_effort isn't supplied at all (no key in the
        parameters dict), behavior should match 'disable' — no thinking
        params injected."""
        await llm_service.call_llm(
            provider="bedrock",
            model="us.anthropic.claude-opus-4-8",
            messages=[{"role": "user", "content": "hi"}],
            parameters={},
            user_service=Mock(),
            user_id="user-1",
        )
        assert "thinking" not in captured
        assert "output_config" not in captured
        assert "reasoning_effort" not in captured