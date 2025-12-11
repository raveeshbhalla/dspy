from unittest import mock

import pydantic
import pytest
from litellm.utils import Choices, Message, ModelResponse

import dspy


class TestBasicAdapterFormatting:
    """Test message formatting in BasicAdapter."""

    def test_single_input_no_tags(self):
        """Single input should be just the raw value without tags."""

        class MySignature(dspy.Signature):
            question: str = dspy.InputField()
            answer: str = dspy.OutputField()

        adapter = dspy.BasicAdapter()
        messages = adapter.format(MySignature, [], {"question": "What is 2+2?"})

        assert len(messages) == 2
        user_content = messages[1]["content"]
        # Should be just the raw value, no XML tags
        assert user_content == "What is 2+2?"
        assert "<question>" not in user_content

    def test_multiple_inputs_with_xml_tags(self):
        """Multiple inputs should use XML tags with field names."""

        class MySignature(dspy.Signature):
            context: str = dspy.InputField()
            question: str = dspy.InputField()
            answer: str = dspy.OutputField()

        adapter = dspy.BasicAdapter()
        messages = adapter.format(
            MySignature, [], {"context": "Paris is the capital of France.", "question": "What is the capital?"}
        )

        assert len(messages) == 2
        user_content = messages[1]["content"]
        # Should have XML tags
        assert "<context>Paris is the capital of France.</context>" in user_content
        assert "<question>What is the capital?</question>" in user_content
        # Should NOT have wrapper <input> tag
        assert "<input>" not in user_content

    def test_multiple_inputs_with_markdown_format(self):
        """Multiple inputs should use markdown headers when format_type='markdown'."""

        class MySignature(dspy.Signature):
            context: str = dspy.InputField()
            question: str = dspy.InputField()
            answer: str = dspy.OutputField()

        adapter = dspy.BasicAdapter(format_type="markdown")
        messages = adapter.format(
            MySignature, [], {"context": "Paris is the capital of France.", "question": "What is the capital?"}
        )

        assert len(messages) == 2
        user_content = messages[1]["content"]
        # Should have markdown headers
        assert "## context" in user_content
        assert "## question" in user_content
        # Should NOT have XML tags
        assert "<context>" not in user_content

    def test_system_message_no_markers(self):
        """System message should not contain [[ ## field ## ]] markers."""

        class MySignature(dspy.Signature):
            """Answer the question."""

            question: str = dspy.InputField()
            answer: str = dspy.OutputField()

        adapter = dspy.BasicAdapter()
        messages = adapter.format(MySignature, [], {"question": "What is 2+2?"})

        system_content = messages[0]["content"]
        # Should NOT have DSPy markers
        assert "[[ ##" not in system_content
        assert "## ]]" not in system_content
        # Should have the task description
        assert "Answer the question" in system_content


class TestBasicAdapterOutputHandling:
    """Test output handling (plain text vs structured)."""

    def test_single_str_output_plain_text(self):
        """Single str output should not use structured output."""

        class MySignature(dspy.Signature):
            question: str = dspy.InputField()
            answer: str = dspy.OutputField()

        adapter = dspy.BasicAdapter()
        assert not adapter._should_use_structured_output(MySignature)

    def test_single_int_output_plain_text(self):
        """Single int output should not use structured output."""

        class MySignature(dspy.Signature):
            question: str = dspy.InputField()
            count: int = dspy.OutputField()

        adapter = dspy.BasicAdapter()
        assert not adapter._should_use_structured_output(MySignature)

    def test_single_float_output_plain_text(self):
        """Single float output should not use structured output."""

        class MySignature(dspy.Signature):
            question: str = dspy.InputField()
            score: float = dspy.OutputField()

        adapter = dspy.BasicAdapter()
        assert not adapter._should_use_structured_output(MySignature)

    def test_single_bool_output_plain_text(self):
        """Single bool output should not use structured output."""

        class MySignature(dspy.Signature):
            question: str = dspy.InputField()
            is_valid: bool = dspy.OutputField()

        adapter = dspy.BasicAdapter()
        assert not adapter._should_use_structured_output(MySignature)

    def test_multiple_outputs_structured(self):
        """Multiple outputs should use structured output."""

        class MySignature(dspy.Signature):
            question: str = dspy.InputField()
            answer: str = dspy.OutputField()
            confidence: float = dspy.OutputField()

        adapter = dspy.BasicAdapter()
        assert adapter._should_use_structured_output(MySignature)

    def test_complex_single_output_structured(self):
        """Single complex type output should use structured output."""

        class Answer(pydantic.BaseModel):
            text: str
            sources: list[str]

        class MySignature(dspy.Signature):
            question: str = dspy.InputField()
            answer: Answer = dspy.OutputField()

        adapter = dspy.BasicAdapter()
        assert adapter._should_use_structured_output(MySignature)

    def test_list_output_structured(self):
        """Single list output should use structured output."""

        class MySignature(dspy.Signature):
            question: str = dspy.InputField()
            answers: list[str] = dspy.OutputField()

        adapter = dspy.BasicAdapter()
        assert adapter._should_use_structured_output(MySignature)


class TestBasicAdapterParsing:
    """Test response parsing in BasicAdapter."""

    def test_parse_simple_str(self):
        """Parse simple string output."""

        class MySignature(dspy.Signature):
            question: str = dspy.InputField()
            answer: str = dspy.OutputField()

        adapter = dspy.BasicAdapter()
        result = adapter.parse(MySignature, "Paris")
        assert result == {"answer": "Paris"}

    def test_parse_simple_int(self):
        """Parse simple int output."""

        class MySignature(dspy.Signature):
            question: str = dspy.InputField()
            count: int = dspy.OutputField()

        adapter = dspy.BasicAdapter()
        result = adapter.parse(MySignature, "42")
        assert result == {"count": 42}

    def test_parse_simple_float(self):
        """Parse simple float output."""

        class MySignature(dspy.Signature):
            question: str = dspy.InputField()
            score: float = dspy.OutputField()

        adapter = dspy.BasicAdapter()
        result = adapter.parse(MySignature, "0.95")
        assert result == {"score": 0.95}

    def test_parse_simple_bool(self):
        """Parse simple bool output."""

        class MySignature(dspy.Signature):
            question: str = dspy.InputField()
            is_valid: bool = dspy.OutputField()

        adapter = dspy.BasicAdapter()
        result = adapter.parse(MySignature, "True")
        assert result == {"is_valid": True}

    def test_parse_json_output(self):
        """Parse JSON output for multiple fields."""

        class MySignature(dspy.Signature):
            question: str = dspy.InputField()
            answer: str = dspy.OutputField()
            confidence: float = dspy.OutputField()

        adapter = dspy.BasicAdapter()
        result = adapter.parse(MySignature, '{"answer": "Paris", "confidence": 0.95}')
        assert result == {"answer": "Paris", "confidence": 0.95}

    def test_parse_json_with_extra_text(self):
        """Parse JSON output even if surrounded by extra text."""

        class MySignature(dspy.Signature):
            question: str = dspy.InputField()
            answer: str = dspy.OutputField()
            confidence: float = dspy.OutputField()

        adapter = dspy.BasicAdapter()
        result = adapter.parse(MySignature, 'Here is my answer:\n{"answer": "Paris", "confidence": 0.95}')
        assert result == {"answer": "Paris", "confidence": 0.95}


class TestBasicAdapterDemos:
    """Test few-shot examples formatting."""

    def test_demos_in_system_message_xml(self):
        """Demos should be included in system message as XML examples."""

        class MySignature(dspy.Signature):
            question: str = dspy.InputField()
            answer: str = dspy.OutputField()

        demos = [
            {"question": "What is 1+1?", "answer": "2"},
            {"question": "What is 2+2?", "answer": "4"},
        ]

        adapter = dspy.BasicAdapter()
        messages = adapter.format(MySignature, demos, {"question": "What is 3+3?"})

        # Should only be system + user messages (demos in system)
        assert len(messages) == 2
        system_content = messages[0]["content"]

        # Demos should be in XML format
        assert "<examples>" in system_content
        assert "<example>" in system_content
        assert "<question>What is 1+1?</question>" in system_content
        assert "<output>2</output>" in system_content

    def test_demos_in_system_message_markdown(self):
        """Demos should be included in system message as markdown examples."""

        class MySignature(dspy.Signature):
            question: str = dspy.InputField()
            answer: str = dspy.OutputField()

        demos = [
            {"question": "What is 1+1?", "answer": "2"},
        ]

        adapter = dspy.BasicAdapter(format_type="markdown")
        messages = adapter.format(MySignature, demos, {"question": "What is 3+3?"})

        system_content = messages[0]["content"]

        # Demos should be in markdown format
        assert "## Examples" in system_content
        assert "### Example 1" in system_content
        assert "**Input:**" in system_content
        assert "**Output:**" in system_content


class TestBasicAdapterSyncAsyncCall:
    """Test sync and async adapter calls."""

    def test_basic_adapter_sync_call_simple_output(self):
        """Test sync call with simple string output."""
        signature = dspy.make_signature("question->answer")
        adapter = dspy.BasicAdapter()

        with mock.patch("litellm.completion") as mock_completion:
            mock_completion.return_value = ModelResponse(
                choices=[Choices(message=Message(content="Paris"))],
                model="openai/gpt-4o-mini",
            )
            lm = dspy.LM("openai/gpt-4o-mini", cache=False)
            result = adapter(lm, {}, signature, [], {"question": "What is the capital of France?"})
            assert result == [{"answer": "Paris"}]

    def test_basic_adapter_sync_call_structured_output(self):
        """Test sync call with structured output."""

        class MySignature(dspy.Signature):
            question: str = dspy.InputField()
            answer: str = dspy.OutputField()
            confidence: float = dspy.OutputField()

        adapter = dspy.BasicAdapter()

        with mock.patch("litellm.completion") as mock_completion:
            mock_completion.return_value = ModelResponse(
                choices=[Choices(message=Message(content='{"answer": "Paris", "confidence": 0.95}'))],
                model="openai/gpt-4o-mini",
            )
            lm = dspy.LM("openai/gpt-4o-mini", cache=False)
            result = adapter(lm, {}, MySignature, [], {"question": "What is the capital of France?"})
            assert result == [{"answer": "Paris", "confidence": 0.95}]

    @pytest.mark.asyncio
    async def test_basic_adapter_async_call(self):
        """Test async call."""
        signature = dspy.make_signature("question->answer")
        adapter = dspy.BasicAdapter()

        with mock.patch("litellm.acompletion") as mock_completion:
            mock_completion.return_value = ModelResponse(
                choices=[Choices(message=Message(content="Paris"))],
                model="openai/gpt-4o-mini",
            )
            lm = dspy.LM("openai/gpt-4o-mini", cache=False)
            result = await adapter.acall(lm, {}, signature, [], {"question": "What is the capital of France?"})
            assert result == [{"answer": "Paris"}]


class TestBasicAdapterWithImages:
    """Test BasicAdapter with image inputs."""

    def test_formats_image(self):
        """Test basic image formatting."""
        image = dspy.Image(url="https://example.com/image.jpg")

        class MySignature(dspy.Signature):
            image: dspy.Image = dspy.InputField()
            text: str = dspy.OutputField()

        adapter = dspy.BasicAdapter()
        messages = adapter.format(MySignature, [], {"image": image})

        assert len(messages) == 2
        user_message_content = messages[1]["content"]

        # The image should be formatted correctly
        expected_image_content = {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
        assert expected_image_content in user_message_content


class TestBasicAdapterWithHistory:
    """Test BasicAdapter with conversation history."""

    def test_formats_conversation_history(self):
        """Test conversation history formatting."""

        class MySignature(dspy.Signature):
            question: str = dspy.InputField()
            history: dspy.History = dspy.InputField()
            answer: str = dspy.OutputField()

        history = dspy.History(
            messages=[
                {"question": "What is the capital of France?", "answer": "Paris"},
            ]
        )

        adapter = dspy.BasicAdapter()
        messages = adapter.format(MySignature, [], {"question": "And Germany?", "history": history})

        # Should have: system, history user, history assistant, current user
        assert len(messages) == 4


class TestBasicAdapterExceptionHandling:
    """Test error handling in BasicAdapter."""

    def test_parse_error_on_invalid_json(self):
        """Should raise AdapterParseError on invalid JSON for structured output."""

        class MySignature(dspy.Signature):
            question: str = dspy.InputField()
            answer: str = dspy.OutputField()
            score: float = dspy.OutputField()

        adapter = dspy.BasicAdapter()
        with pytest.raises(dspy.utils.exceptions.AdapterParseError):
            adapter.parse(MySignature, "not valid json at all {{{")

    def test_parse_error_on_missing_fields(self):
        """Should raise AdapterParseError when required fields are missing."""

        class MySignature(dspy.Signature):
            question: str = dspy.InputField()
            answer: str = dspy.OutputField()
            score: float = dspy.OutputField()

        adapter = dspy.BasicAdapter()
        with pytest.raises(dspy.utils.exceptions.AdapterParseError):
            adapter.parse(MySignature, '{"answer": "Paris"}')  # missing score
