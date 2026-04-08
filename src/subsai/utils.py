#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions
"""

import pathlib
from typing import Optional, Union

import torch
from pysubs2.formats import FILE_EXTENSION_TO_FORMAT_IDENTIFIER


def _load_config(config_name, model_config, config_schema):
    """
    Helper function to load default values if `config_name` is not specified

    :param config_name: the name of the config
    :param model_config: configuration provided to the model
    :param config_schema: the schema of the configuration

    :return: config value
    """
    if config_name in model_config:
        return model_config[config_name]
    return config_schema[config_name]['default']


def get_available_devices() -> list:
    """
    Get available devices (cpu and gpus)

    :return: list of available devices
    """
    return ['cpu', *[f'cuda:{i}' for i in range(torch.cuda.device_count())]]


def available_translation_models() -> list:
    """
    Returns available translation models
    from (dl-translate)[https://github.com/xhluca/dl-translate]

    :return: list of available models
    """
    models = [
        "facebook/m2m100_418M",
        "facebook/m2m100_1.2B",
        "facebook/mbart-large-50-many-to-many-mmt",
        "facebook/nllb-200-distilled-600M"
    ]
    return models


def available_subs_formats(include_extensions=True):
    """
    Returns available subtitle formats
    from :attr:`pysubs2.FILE_EXTENSION_TO_FORMAT_IDENTIFIER`

    :param include_extensions: include the dot separator in file extensions

    :return: list of subtitle formats
    """

    extensions = list(FILE_EXTENSION_TO_FORMAT_IDENTIFIER.keys())

    if include_extensions:
        return extensions
    else:
        # remove the '.' separator from extension names
        return [ext.split('.')[1] for ext in extensions]


def build_subtitle_path(
    media_file: Union[str, pathlib.Path],
    subs_format: str = 'srt',
    destination_folder: Optional[Union[str, pathlib.Path]] = None,
    output_suffix: Optional[str] = None,
    language_suffix: Optional[str] = None,
) -> pathlib.Path:
    """
    Build the output path for a subtitle file, optionally inserting a language
    code before the extension (e.g. ``video.en.srt``).

    This is useful for media servers like Plex and Jellyfin that auto-detect
    subtitle language from the filename pattern ``video.LANG.ext``.

    :param media_file: path to the source media file
    :param subs_format: subtitle format / extension (without dot), e.g. ``srt``
    :param destination_folder: optional output directory (defaults to same
        folder as *media_file*)
    :param output_suffix: optional string appended to the stem before the
        language code
    :param language_suffix: ISO-639 language code to insert before the
        extension (e.g. ``en``, ``ar``, ``ja``). If *None*, no language code
        is added.

    :return: :class:`pathlib.Path` of the subtitle file
    """
    media_path = pathlib.Path(media_file)

    if destination_folder is not None:
        folder = pathlib.Path(destination_folder).absolute()
    else:
        folder = media_path.parent

    stem = media_path.stem
    if output_suffix is not None:
        stem = stem + output_suffix
    if language_suffix is not None:
        stem = stem + '.' + language_suffix

    return folder / (stem + '.' + subs_format)
