# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import iree.runtime
import pytest
import platform
import torch

from parameterized import parameterized
from pathlib import Path
from sharktank.layers import create_model, model_config_presets
from sharktank.types import DefaultPrimitiveTensor
from sharktank.utils import chdir
from sharktank.utils.iree import (
    device_array_to_host,
    get_iree_devices,
    run_model_with_iree_run_module,
    trace_model_with_tracy,
    with_iree_device_context,
)
from sharktank.utils.testing import skip
from sharktank.models.dummy import DummyModel
from sharktank import ops
from unittest import TestCase


@pytest.fixture(scope="session")
def dummy_model_path(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("dummy_model")


@pytest.fixture(scope="session")
def dummy_model(dummy_model_path: Path) -> DummyModel:
    with chdir(dummy_model_path):
        model = create_model(model_config_presets["dummy-model-local-llvm-cpu"])
        model.export()
        model.compile()
        return model


def test_run_model_with_iree_run_module(
    dummy_model: DummyModel, dummy_model_path: Path
):
    with chdir(dummy_model_path):
        run_model_with_iree_run_module(dummy_model.config, function="forward_bs1")


@skip(
    reason=(
        "The test hangs. Probably during compilation or IREE module "
        "execution. We can't determine easily what is going on as running "
        "tests in parallel with pyest-xdist is incompatible with capture "
        "disabling with --capture=no. No live logs are available from the CI."
        " TODO: investigate"
    )
)
@pytest.mark.xfail(
    platform.system() == "Windows",
    raises=FileNotFoundError,
    strict=True,
    reason="The Python package for Windows does not include iree-tracy-capture.",
)
def test_trace_model_with_tracy(dummy_model: DummyModel, dummy_model_path: Path):
    with chdir(dummy_model_path):
        trace_path = Path(f"{dummy_model.config.iree_module_path}.tracy")
        assert not trace_path.exists()
        trace_model_with_tracy(dummy_model.config, function="forward_bs1")
        assert trace_path.exists()
