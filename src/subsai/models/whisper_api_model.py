#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Whisper API Model

See [openai/whisper](https://platform.openai.com/docs/guides/speech-to-text)
"""

import os
import ffmpeg
import tempfile
from subsai.models.abstract_model import AbstractModel
from subsai.utils import _load_config
from openai import OpenAI
from pysubs2 import SSAFile
from pydub import AudioSegment
from joblib import Parallel, delayed

TMPDIR = tempfile.gettempdir()
OPENAI_API_SIZE_LIMIT_MB = 24

def split_filename(filepath):
    path, full_filename = os.path.split(filepath)
    filename, ext = os.path.splitext(full_filename)
    return path,filename,ext

def convert_video_to_audio_ffmpeg(video_file, output_ext="mp3"):
    # Construct the output file name
    path,filename,ext = split_filename(video_file)
    output_file = os.path.join(TMPDIR,f"{filename}.{output_ext}")
    

    print('Saving audio to {} with ffmpeg...'.format(output_file))
    # Execute the ffmpeg conversion
    (
        ffmpeg
        .input(video_file)
        .output(output_file)
        .overwrite_output()
        .run(quiet=False)
    )
    return output_file

class WhisperAPIModel(AbstractModel):
    model_name = 'openai/whisper'
    config_schema = {
            # load model config
            'model_type': {
                'type': list,
                'description': "OpenAI Whisper API, currently only supports large-v2 which is named as whisper-1/ \
                                There is a 25mb upload limit so audio is chunked locally, this may lead to lower performance.",
                'options': ['whisper-1', 'gpt-4o-mini-transcribe', 'gpt-4o-transcribe'],
                'default': 'whisper-1'
            },
            'api_key': {
                'type': str,
                'description': "Your OpenAI API key",
                'options': None,
                'default': os.environ.get('OPENAI_KEY', None)
            },
            'language': {
                'type': str,
                'description': "The language of the input audio. Supplying the input language in ISO-639-1 format will improve accuracy and latency.",
                'options': None,
                'default': None
            },
            'prompt': {
                'type': str,
                'description': "An optional text to guide the model's style or continue a previous audio segment. The prompt should match the audio language.",
                'options': None,
                'default': None
            },
            'temperature': {
                'type': float,
                'description': "The sampling temperature, between 0 and 1. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic. If set to 0, the model will use log probability to automatically increase the temperature until certain thresholds are hit.",
                'options': None,
                'default': 0
            },
            'base_url': {
                'type': str,
                'description': "The base URL for the API. Useful if you're already self hosting whisper for example.",
                'options': None,
                'default': "https://api.openai.com/v1/"
            },
            "n_jobs": {
                "type": int,
                "description": "Number of calls to do in parallel (1 to not use parallel call)",
                "options": None,
                "default": 1,
            }
    }

    def __init__(self, model_config):
        # config
        self.model_type = _load_config('model_type', model_config, self.config_schema)
        self.api_key = _load_config('api_key', model_config, self.config_schema)
        self.language = _load_config('language', model_config, self.config_schema)
        self.prompt = _load_config('prompt', model_config, self.config_schema)
        self.temperature = _load_config('temperature', model_config, self.config_schema)
        self.base_url = _load_config('base_url', model_config, self.config_schema)
        if not self.base_url.endswith("/"):
            self.base_url += "/"
        self.n_jobs = _load_config("n_jobs", model_config, self.config_schema)

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def chunk_audio(self, audio_file_path) -> list:
        # Load the audio file
        audio = AudioSegment.from_mp3(audio_file_path)

        # Desired chunk size in megabytes (MB)
        chunk_size_bits = OPENAI_API_SIZE_LIMIT_MB * 1024 * 1024 * 8
        bitrate = audio.frame_rate * audio.frame_width
        chunk_duration_ms = ((chunk_size_bits) / bitrate) * 1000

        chunks = []

        # Split the audio into chunks
        current_ms = 0
        while current_ms < len(audio):
            # Calculate the end of the current chunk
            end_ms = current_ms + chunk_duration_ms
            # Create a chunk from the current position to the end position
            chunk = audio[current_ms : int(end_ms)]
            # Add the chunk to the list of chunks and include offset
            chunks.append((chunk, current_ms))
            # Update the current position
            current_ms = end_ms

        return chunks

    def _transcribe_chunk(self, chunk_data):
        """
        Transcribe a single audio chunk using OpenAI Whisper API.

        Parameters
        ----------
        chunk_data : tuple
            Tuple containing (chunk_index, chunk, offset)

        Returns
        -------
        tuple
            Tuple containing (chunk_index, transcription_result, offset)
        """
        i, chunk, offset = chunk_data
        chunk_path = os.path.join(TMPDIR, f"chunk_{i}.mp3")

        try:
            print("Transcribing audio chunk {}".format(i))
            chunk.export(chunk_path, format="mp3")

            with open(chunk_path, "rb") as audio_file:
                # Use OpenAI Whisper API
                result = self.client.audio.transcriptions.create(
                    model=self.model_type,
                    language=self.language,
                    prompt=self.prompt,
                    temperature=self.temperature,
                    file=audio_file,
                    response_format="srt",
                )

            # Clean up the temporary chunk file
            os.remove(chunk_path)

            return (i, result, offset)

        except Exception as e:
            # Clean up the temporary chunk file in case of error
            if os.path.exists(chunk_path):
                os.remove(chunk_path)
            raise e

    def transcribe(self, media_file: str) -> SSAFile:

        audio_file_path = convert_video_to_audio_ffmpeg(media_file)

        try:
            chunks = self.chunk_audio(audio_file_path)

            print(f"Processing {len(chunks)} audio chunks with {self.n_jobs} parallel jobs")

            # Prepare chunk data for parallel processing
            chunk_data = [(i, chunk, offset) for i, (chunk, offset) in enumerate(chunks)]

            # Use parallel processing if n_jobs > 1, otherwise process sequentially
            if self.n_jobs > 1:
                # Use threading backend since API calls are I/O-bound
                parallel_results = Parallel(n_jobs=self.n_jobs, backend="threading")(
                    delayed(self._transcribe_chunk)(data) for data in chunk_data
                )
            else:
                # Sequential processing for n_jobs=1
                parallel_results = [self._transcribe_chunk(data) for data in chunk_data]

            # Sort results by chunk index to maintain order
            parallel_results.sort(key=lambda x: x[0])

            # Process results and apply time offsets
            results = ""
            for i, result_text, offset in parallel_results:
                # Shift subtitles by offset
                result = SSAFile.from_string(result_text)
                result.shift(ms=offset)
                results += result.to_string("srt")

            return SSAFile.from_string(results)
        finally:
            if os.path.exists(audio_file_path):
                os.unlink(audio_file_path)
