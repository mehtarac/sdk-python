import asyncio
import base64
import unittest.mock

import pytest
import pytest_asyncio

from strands.experimental.bidi.io.audio import BidiAudioIO, _BidiAudioBuffer
from strands.experimental.bidi.types.events import BidiAudioInputEvent, BidiAudioStreamEvent, BidiInterruptionEvent


@pytest.fixture
def audio_buffer():
    buffer = _BidiAudioBuffer(size=1)
    buffer.start()
    yield buffer
    buffer.stop()


@pytest.fixture
def mock_agent():
    """Create a mock agent with model that has default audio_config."""
    agent = unittest.mock.MagicMock()
    agent.model.audio_config = {
        "input_rate": 16000,
        "output_rate": 16000,
        "channels": 1,
        "format": "pcm",
        "voice": "matthew",
    }
    return agent


@pytest.fixture
def mock_agent_custom_config():
    """Create a mock agent with custom audio_config."""
    agent = unittest.mock.MagicMock()
    agent.model.audio_config = {
        "input_rate": 48000,
        "output_rate": 24000,
        "channels": 2,
        "format": "pcm",
        "voice": "alloy",
    }
    return agent


@pytest.fixture
def py_audio():
    with unittest.mock.patch("strands.experimental.bidi.io.audio.pyaudio.PyAudio") as mock:
        yield mock.return_value

@pytest.fixture
def audio_io(py_audio):
    _ = py_audio
    return BidiAudioIO()


@pytest_asyncio.fixture
async def audio_input(audio_io, mock_agent):
    input_ = audio_io.input()
    await input_.start(mock_agent)
    yield input_
    await input_.stop()


@pytest_asyncio.fixture
async def audio_output(audio_io, mock_agent):
    output = audio_io.output()
    await output.start(mock_agent)
    yield output
    await output.stop()


def test_bidi_audio_buffer_put(audio_buffer):
    audio_buffer.put(b"test-chunk")

    tru_chunk = audio_buffer.get()
    exp_chunk = b"test-chunk"
    assert tru_chunk == exp_chunk


def test_bidi_audio_buffer_put_full(audio_buffer):
    audio_buffer.put(b"test-chunk-1")
    audio_buffer.put(b"test-chunk-2")

    tru_chunk = audio_buffer.get()
    exp_chunk = b"test-chunk-2"
    assert tru_chunk == exp_chunk


def test_bidi_audio_buffer_get_padding(audio_buffer):
    audio_buffer.put(b"test-chunk")

    tru_chunk = audio_buffer.get(11)
    exp_chunk = b"test-chunk\x00"
    assert tru_chunk == exp_chunk


def test_bidi_audio_buffer_clear(audio_buffer):
    audio_buffer.put(b"test-chunk")
    audio_buffer.clear()

    tru_byte = audio_buffer.get(1)
    exp_byte = b"\x00"
    assert tru_byte == exp_byte


@pytest.mark.asyncio
async def test_bidi_audio_io_input(audio_input):
    audio_input._callback(b"test-audio")

    tru_event = await audio_input()
    exp_event = BidiAudioInputEvent(
        audio=base64.b64encode(b"test-audio").decode("utf-8"),
        channels=1,
        format="pcm",
        sample_rate=16000,
    )
    assert tru_event == exp_event


@pytest.mark.asyncio
async def test_bidi_audio_io_output(audio_output):
    audio_event = BidiAudioStreamEvent(
        audio=base64.b64encode(b"test-audio").decode("utf-8"),
        channels=1,
        format="pcm",
        sample_rate=1600,
    )
    await audio_output(audio_event)

    tru_data, _ = audio_output._callback(None, frame_count=4)
    exp_data = b"test-aud"
    assert tru_data == exp_data


@pytest.mark.asyncio
async def test_bidi_audio_io_output_interrupt(audio_output):
    audio_event = BidiAudioStreamEvent(
        audio=base64.b64encode(b"test-audio").decode("utf-8"),
        channels=1,
        format="pcm",
        sample_rate=1600,
    )
    await audio_output(audio_event)
    interrupt_event = BidiInterruptionEvent(reason="user_speech")
    await audio_output(interrupt_event)

    tru_data, _ = audio_output._callback(None, frame_count=1)
    exp_data = b"\x00\x00"
    assert tru_data == exp_data


# Audio Configuration Tests


# @pytest.mark.asyncio
# async def test_audio_input_uses_model_config(audio_input):
#     microphone = unittest.mock.Mock()
#     microphone.read.return_value = b"test-audio"
#     py_audio.open.return_value = microphone

#     # Model config should be used
#     py_audio.open.assert_called_once()
#     call_kwargs = py_audio.open.call_args.kwargs
#     assert call_kwargs["rate"] == 16000  # From mock_agent.model.audio_config
#     assert call_kwargs["channels"] == 1  # From mock_agent.model.audio_config


# @pytest.mark.asyncio
# async def test_audio_input_uses_custom_model_config(py_audio, audio_io, mock_agent_custom_config):
#     """Test that audio input uses custom model audio_config."""
#     audio_input = audio_io.input()

#     microphone = unittest.mock.Mock()
#     microphone.read.return_value = b"test-audio"
#     py_audio.open.return_value = microphone

#     await audio_input.start(mock_agent_custom_config)

#     # Custom model config should be used
#     py_audio.open.assert_called_once()
#     call_kwargs = py_audio.open.call_args.kwargs
#     assert call_kwargs["rate"] == 48000  # From custom config
#     assert call_kwargs["channels"] == 2  # From custom config

#     await audio_input.stop()


# @pytest.mark.asyncio
# async def test_audio_output_uses_model_config(py_audio, audio_io, mock_agent):
#     """Test that audio output uses model's audio_config."""
#     audio_output = audio_io.output()

#     speaker = unittest.mock.Mock()
#     py_audio.open.return_value = speaker

#     await audio_output.start(mock_agent)

#     # Model config should be used
#     py_audio.open.assert_called_once()
#     call_kwargs = py_audio.open.call_args.kwargs
#     assert call_kwargs["rate"] == 16000  # From mock_agent.model.audio_config
#     assert call_kwargs["channels"] == 1  # From mock_agent.model.audio_config

#     await audio_output.stop()


# @pytest.mark.asyncio
# async def test_audio_output_uses_custom_model_config(py_audio, audio_io, mock_agent_custom_config):
#     """Test that audio output uses custom model audio_config."""
#     audio_output = audio_io.output()

#     speaker = unittest.mock.Mock()
#     py_audio.open.return_value = speaker

#     await audio_output.start(mock_agent_custom_config)

#     # Custom model config should be used
#     py_audio.open.assert_called_once()
#     call_kwargs = py_audio.open.call_args.kwargs
#     assert call_kwargs["rate"] == 24000  # From custom config
#     assert call_kwargs["channels"] == 2  # From custom config

#     await audio_output.stop()


# # Device Configuration Tests


# @pytest.mark.asyncio
# async def test_audio_input_respects_user_device_config(py_audio, mock_agent):
#     """Test that user-provided device config overrides defaults."""
#     audio_io = BidiAudioIO(input_device_index=5, input_frames_per_buffer=1024)
#     audio_input = audio_io.input()

#     microphone = unittest.mock.Mock()
#     microphone.read.return_value = b"test-audio"
#     py_audio.open.return_value = microphone

#     await audio_input.start(mock_agent)

#     # User device config should be used
#     py_audio.open.assert_called_once()
#     call_kwargs = py_audio.open.call_args.kwargs
#     assert call_kwargs["input_device_index"] == 5  # User config
#     assert call_kwargs["frames_per_buffer"] == 1024  # User config
#     # Model config still used for audio parameters
#     assert call_kwargs["rate"] == 16000  # From model
#     assert call_kwargs["channels"] == 1  # From model

#     await audio_input.stop()


# @pytest.mark.asyncio
# async def test_audio_output_respects_user_device_config(py_audio, mock_agent):
#     """Test that user-provided device config overrides defaults."""
#     audio_io = BidiAudioIO(output_device_index=3, output_frames_per_buffer=2048, output_buffer_size=50)
#     audio_output = audio_io.output()

#     speaker = unittest.mock.Mock()
#     py_audio.open.return_value = speaker

#     await audio_output.start(mock_agent)

#     # User device config should be used
#     py_audio.open.assert_called_once()
#     call_kwargs = py_audio.open.call_args.kwargs
#     assert call_kwargs["output_device_index"] == 3  # User config
#     assert call_kwargs["frames_per_buffer"] == 2048  # User config
#     # Model config still used for audio parameters
#     assert call_kwargs["rate"] == 16000  # From model
#     assert call_kwargs["channels"] == 1  # From model
#     # Buffer size should be set
#     assert audio_output._buffer_size == 50  # User config

#     await audio_output.stop()


# @pytest.mark.asyncio
# async def test_audio_io_uses_defaults_when_no_config(py_audio, mock_agent):
#     """Test that defaults are used when no config provided."""
#     audio_io = BidiAudioIO()  # No config
#     audio_input = audio_io.input()

#     microphone = unittest.mock.Mock()
#     microphone.read.return_value = b"test-audio"
#     py_audio.open.return_value = microphone

#     await audio_input.start(mock_agent)

#     # Defaults should be used
#     py_audio.open.assert_called_once()
#     call_kwargs = py_audio.open.call_args.kwargs
#     assert call_kwargs["input_device_index"] is None  # Default
#     assert call_kwargs["frames_per_buffer"] == 512  # Default

#     await audio_input.stop()
