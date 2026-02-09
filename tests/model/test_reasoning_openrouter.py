import json
from unittest.mock import patch

import pytest

from inspect_ai._util.content import ContentReasoning, ContentText
from inspect_ai.model._chat_message import ChatMessageAssistant
from inspect_ai.model._providers.openrouter import (
    OPENROUTER_REASONING_DETAILS_SIGNATURE,
    OpenRouterAPI,
    openrouter_reasoning_details_to_reasoning,
    reasoning_to_openrouter_reasoning_details,
)

# =============================================================================
# Tests for openrouter_reasoning_details_to_reasoning()
# =============================================================================


class TestOpenrouterReasoningDetailsToReasoning:
    """Tests for converting OpenRouter reasoning_details to ContentReasoning."""

    def test_text_type(self):
        """reasoning.text type extracts text as reasoning."""
        details = [
            {
                "type": "reasoning.text",
                "text": "Let me think about this step by step...",
                "id": "r1",
                "format": "anthropic-claude-v1",
            }
        ]
        result = openrouter_reasoning_details_to_reasoning(details)

        assert result.reasoning == "Let me think about this step by step..."
        assert result.summary is None
        assert result.redacted is False
        assert result.signature is not None
        assert result.signature.startswith(OPENROUTER_REASONING_DETAILS_SIGNATURE)

    def test_summary_type_only(self):
        """reasoning.summary alone becomes the reasoning (fallback behavior)."""
        details = [
            {
                "type": "reasoning.summary",
                "summary": "The model analyzed the problem",
                "id": "s1",
                "format": "anthropic-claude-v1",
            }
        ]
        result = openrouter_reasoning_details_to_reasoning(details)

        # When only summary exists, it becomes the reasoning
        assert result.reasoning == "The model analyzed the problem"
        assert result.summary is None  # summary moved to reasoning
        assert result.redacted is False

    def test_text_and_summary_combined(self):
        """Both text and summary preserves both fields."""
        details = [
            {
                "type": "reasoning.summary",
                "summary": "High-level summary",
                "id": "s1",
                "format": "anthropic-claude-v1",
            },
            {
                "type": "reasoning.text",
                "text": "Detailed reasoning content",
                "id": "t1",
                "format": "anthropic-claude-v1",
            },
        ]
        result = openrouter_reasoning_details_to_reasoning(details)

        assert result.reasoning == "Detailed reasoning content"
        assert result.summary == "High-level summary"
        assert result.redacted is False

    def test_encrypted_type(self):
        """reasoning.encrypted sets redacted=True."""
        details = [
            {
                "type": "reasoning.encrypted",
                "data": "eyJlbmNyeXB0ZWQiOiJ0cnVlIn0=",
                "id": "e1",
                "format": "anthropic-claude-v1",
            }
        ]
        result = openrouter_reasoning_details_to_reasoning(details)

        assert result.reasoning == "eyJlbmNyeXB0ZWQiOiJ0cnVlIn0="
        assert result.redacted is True

    def test_empty_list_logs_warning(self):
        """Empty reasoning_details list logs warning and returns raw JSON."""
        with patch("inspect_ai.model._providers.openrouter.logger") as mock_logger:
            details = []
            result = openrouter_reasoning_details_to_reasoning(details)

            mock_logger.warning.assert_called_once()
            call_args = mock_logger.warning.call_args[0][0]
            assert "Reasoning content not provided" in call_args
            assert result.reasoning == "[]"

    def test_invalid_format_logs_warning(self):
        """Invalid/malformed data logs warning and returns raw JSON."""
        with patch("inspect_ai.model._providers.openrouter.logger") as mock_logger:
            details = [{"type": "unknown.type", "foo": "bar"}]
            result = openrouter_reasoning_details_to_reasoning(details)

            mock_logger.warning.assert_called_once()
            call_args = mock_logger.warning.call_args[0][0]
            assert "Error parsing OpenRouter reasoning details" in call_args
            # Falls back to raw JSON
            assert result.signature is not None

    def test_signature_contains_original_json(self):
        """Signature preserves full original JSON for round-tripping."""
        details = [
            {
                "type": "reasoning.text",
                "text": "Some reasoning",
                "id": "r1",
                "format": "anthropic-claude-v1",
                "index": 0,
            }
        ]
        result = openrouter_reasoning_details_to_reasoning(details)

        # Extract JSON from signature
        assert result.signature is not None
        json_str = result.signature.replace(OPENROUTER_REASONING_DETAILS_SIGNATURE, "")
        recovered = json.loads(json_str)

        assert recovered == details


# =============================================================================
# Tests for reasoning_to_openrouter_reasoning_details()
# =============================================================================


class TestReasoningToOpenrouterReasoningDetails:
    """Tests for converting ContentReasoning back to OpenRouter format."""

    def test_valid_signature_returns_details(self):
        """ContentReasoning with OpenRouter signature returns reasoning_details."""
        original_details = [
            {
                "type": "reasoning.text",
                "text": "My reasoning",
                "id": "r1",
                "format": "anthropic-claude-v1",
            }
        ]
        signature = (
            f"{OPENROUTER_REASONING_DETAILS_SIGNATURE}{json.dumps(original_details)}"
        )
        content = ContentReasoning(
            reasoning="My reasoning",
            signature=signature,
        )

        result = reasoning_to_openrouter_reasoning_details(content)

        assert result is not None
        assert "reasoning_details" in result
        assert result["reasoning_details"] == original_details

    def test_no_signature_returns_none(self):
        """ContentReasoning without signature returns None."""
        content = ContentReasoning(reasoning="Some reasoning")

        result = reasoning_to_openrouter_reasoning_details(content)

        assert result is None

    def test_wrong_signature_returns_none(self):
        """ContentReasoning with non-OpenRouter signature returns None."""
        content = ContentReasoning(
            reasoning="Some reasoning",
            signature="some-other-signature-format",
        )

        result = reasoning_to_openrouter_reasoning_details(content)

        assert result is None

    def test_empty_signature_returns_none(self):
        """ContentReasoning with empty signature returns None."""
        content = ContentReasoning(
            reasoning="Some reasoning",
            signature="",
        )

        result = reasoning_to_openrouter_reasoning_details(content)

        assert result is None


# =============================================================================
# Round-trip tests
# =============================================================================


class TestRoundTrip:
    """Tests that reasoning_details survives round-trip conversion."""

    def test_text_round_trip(self):
        """Text type round-trips correctly."""
        original = [
            {
                "type": "reasoning.text",
                "text": "Step by step reasoning",
                "id": "r1",
                "format": "anthropic-claude-v1",
                "index": 0,
            }
        ]

        # Convert to ContentReasoning
        content = openrouter_reasoning_details_to_reasoning(original)

        # Convert back to reasoning_details
        result = reasoning_to_openrouter_reasoning_details(content)

        assert result is not None
        assert result["reasoning_details"] == original

    def test_encrypted_round_trip(self):
        """Encrypted type round-trips correctly."""
        original = [
            {
                "type": "reasoning.encrypted",
                "data": "encrypted-base64-data",
                "id": "e1",
                "format": "anthropic-claude-v1",
            }
        ]

        content = openrouter_reasoning_details_to_reasoning(original)
        result = reasoning_to_openrouter_reasoning_details(content)

        assert result is not None
        assert result["reasoning_details"] == original

    def test_complex_round_trip(self):
        """Complex multi-element reasoning_details round-trips correctly."""
        original = [
            {
                "type": "reasoning.summary",
                "summary": "Analyzed the problem",
                "id": "s1",
                "format": "anthropic-claude-v1",
                "index": 0,
            },
            {
                "type": "reasoning.text",
                "text": "First, let me consider...\nThen, I'll analyze...",
                "signature": None,
                "id": "t1",
                "format": "anthropic-claude-v1",
                "index": 1,
            },
        ]

        content = openrouter_reasoning_details_to_reasoning(original)
        result = reasoning_to_openrouter_reasoning_details(content)

        assert result is not None
        assert result["reasoning_details"] == original


# =============================================================================
# Tests for xAI reasoning replay handling
# =============================================================================


def _make_openrouter_api(model_name: str) -> OpenRouterAPI:
    with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
        return OpenRouterAPI(model_name=model_name)


def _encrypted_reasoning() -> ContentReasoning:
    details = [
        {
            "type": "reasoning.encrypted",
            "data": "encrypted-base64-data",
            "id": "e1",
            "format": "openrouter-v1",
        }
    ]
    signature = f"{OPENROUTER_REASONING_DETAILS_SIGNATURE}{json.dumps(details)}"
    return ContentReasoning(
        reasoning="encrypted-base64-data",
        redacted=True,
        signature=signature,
    )


class TestXaiReasoningReplay:
    """Tests that xAI models skip reasoning_details replay."""

    @pytest.mark.asyncio
    async def test_xai_encrypted_reasoning_dropped(self):
        """Encrypted reasoning from xAI should not appear in messages."""
        api = _make_openrouter_api("x-ai/grok-4")
        messages = [
            ChatMessageAssistant(
                content=[_encrypted_reasoning(), ContentText(text="Hello!")],
                source="generate",
            )
        ]

        result = await api.messages_to_openai(messages)

        for msg in result:
            assert "reasoning_details" not in msg

    @pytest.mark.asyncio
    async def test_non_xai_encrypted_reasoning_replayed(self):
        """Encrypted reasoning from non-xAI models should be replayed."""
        api = _make_openrouter_api("openai/o4-mini")
        messages = [
            ChatMessageAssistant(
                content=[_encrypted_reasoning(), ContentText(text="Hello!")],
                source="generate",
            )
        ]

        result = await api.messages_to_openai(messages)

        has_reasoning_details = any("reasoning_details" in msg for msg in result)
        assert has_reasoning_details
