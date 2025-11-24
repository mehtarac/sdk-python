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
def py_audio():
    with unittest.mock.patch("strands.experimental.bidi.io.audio.pyaudio.PyAudio") as mock:
        yield mock.return_value

@pytest.fixture
def audio_io(py_audio):
    _ = py_audio
    return BidiAudioIO()


@pytest_asyncio.fixture
async def audio_input(audio_io):
    input_ = audio_io.input()
    await input_.start()
    yield input_
    await input_.stop()


@pytest_asyncio.fixture
async def audio_output(audio_io):
    output = audio_io.output()
    await output.start()
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
