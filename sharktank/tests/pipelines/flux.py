# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest
import pytest

import torch

from sharktank.types import Dataset
from sharktank.utils.hf_datasets import get_dataset
from sharktank.models.clip.export import (
    export_clip_text_model_mlir,
)
from transformers import CLIPTextModel, CLIPTokenizer, T5TokenizerFast
from sharktank.layers.configs.llm_configs import ClipTextConfig
from sharktank.models.t5 import export_encoder_mlir
from sharktank.models.flux.flux import FluxModelV1, FluxParams
from sharktank.models.flux.export import export_flux_transformer_model_mlir

from sharktank.models.vae.model import VaeDecoderModel
from sharktank.models.vae.tools.run_vae import export_vae
from sharktank.models.vae.tools.sample_data import get_random_inputs

from sharktank.utils.iree import (
    get_iree_devices,
    load_iree_module,
)
import iree.compiler

clip_compile_flags = [
    "--iree-hal-target-backends=rocm",
    "--iree-hip-target=gfx942",
    "--iree-input-type=torch",
    "--iree-opt-const-eval=false",
    "--iree-opt-strip-assertions=true",
    "--iree-global-opt-propagate-transposes=true",
    "--iree-opt-outer-dim-concat=true",
    "--iree-hip-waves-per-eu=2",
    "--iree-llvmgpu-enable-prefetch",
    "--iree-dispatch-creation-enable-aggressive-fusion",
    "--iree-dispatch-creation-enable-fuse-horizontal-contractions=true",
    "--iree-opt-aggressively-propagate-transposes=true",
    "--iree-codegen-llvmgpu-use-vector-distribution=true",
    "--iree-execution-model=async-external",
    "--iree-preprocessing-pass-pipeline=builtin.module(iree-preprocessing-transpose-convolution-pipeline,iree-preprocessing-pad-to-intrinsics{pad-target-type=conv})",
    "--iree-scheduling-dump-statistics-format=json",
    "--iree-scheduling-dump-statistics-file=compilation_info.json",
]
flux_compile_flags = [
    "--iree-hal-target-backends=rocm",
    "--iree-hip-target=gfx942",
    "--iree-opt-const-eval=false",
    "--iree-opt-strip-assertions=true",
    "--iree-global-opt-propagate-transposes=true",
    "--iree-dispatch-creation-enable-fuse-horizontal-contractions=true",
    "--iree-dispatch-creation-enable-aggressive-fusion=true",
    "--iree-opt-aggressively-propagate-transposes=true",
    "--iree-opt-outer-dim-concat=true",
    "--iree-vm-target-truncate-unsupported-floats",
    "--iree-llvmgpu-enable-prefetch=true",
    "--iree-opt-data-tiling=false",
    "--iree-codegen-gpu-native-math-precision=true",
    "--iree-codegen-llvmgpu-use-vector-distribution",
    "--iree-hip-waves-per-eu=2",
    "--iree-execution-model=async-external",
    "--iree-scheduling-dump-statistics-format=json",
    "--iree-scheduling-dump-statistics-file=compilation_info.json",
]
vae_compile_flags = [
    "--iree-hal-target-backends=rocm",
    "--iree-hip-target=gfx942",
    "--iree-opt-const-eval=false",
    "--iree-opt-strip-assertions=true",
    "--iree-global-opt-propagate-transposes=true",
    "--iree-opt-outer-dim-concat=true",
    "--iree-llvmgpu-enable-prefetch=true",
    "--iree-hip-waves-per-eu=2",
    "--iree-dispatch-creation-enable-aggressive-fusion=true",
    "--iree-codegen-llvmgpu-use-vector-distribution=true",
    "--iree-execution-model=async-external",
    "--iree-preprocessing-pass-pipeline=builtin.module(iree-preprocessing-transpose-convolution-pipeline,iree-preprocessing-pad-to-intrinsics)",
]

sample_prompt = [
    "a cat under the snow with blue eyes, covered by snow, cinematic style, medium shot, professional photo, animal"
]
neg_prompt = ["Watermark, blurry, oversaturated, low resolution, pollution"]


class FluxPipelineTest(unittest.TestCase):
    def setUp(self):
        dtype = torch.bfloat16

        export_clip_text_model_mlir(
            "/data/shark/fluxtesting/openai__clip_vit_large_patch14_text_model_bf16.irpa",
            batch_sizes=[1],
            mlir_output_path="flux/clip_bf16.mlir",
        )
        iree.compiler.compile_file(
            "flux/clip_bf16.mlir",
            output_file="flux/clip_bf16.vmfb",
            extra_args=clip_compile_flags,
        )

        export_encoder_mlir(
            "/data/shark/fluxtesting/google__t5_v1_1_small_encoder_bf16.irpa",
            batch_sizes=[1],
            mlir_output_path="flux/t5_bf16.mlir",
        )
        iree.compiler.compile_file(
            "flux/t5_bf16.mlir",
            output_file="flux/t5_bf16.vmfb",
            extra_args=clip_compile_flags,
        )
        ds_flux = Dataset.load("/data/shark/fluxtesting/black-forest-labs--FLUX.1-schnell--transformer-single-layer-b16.irpa", file_type="irpa")
        flux_params = FluxParams(
            in_channels=64,
            out_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=1,
            depth_single_blocks=1,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=False,
        )
        flux_model = FluxModelV1(theta=ds_flux.root_theta, params=flux_params)
        export_flux_transformer_model_mlir(flux_model, output_path="flux/flux_bf16.mlir", batch_sizes=[1])
        iree.compiler.compile_file("flux/flux_bf16.mlir",
            output_file = "flux/flux_bf16.vmfb",
            extra_args = flux_compile_flags,
        )
        vae_inputs = get_random_inputs(dtype=dtype, device="cpu", bs=1, config="flux")
        ds_vae = Dataset.load("/data/shark/fluxtesting/vae_bf16.irpa", file_type="irpa")
        vae_model = VaeDecoderModel.from_dataset(ds_vae).to(device="cpu")
        vae_module = export_vae(vae_model, vae_inputs, True)
        vae_module.save_mlir("flux/vae_bf16.mlir")
        iree.compiler.compile_file(
            "flux/vae_bf16.mlir",
            output_file="flux/vae_bf16.vmfb",
            extra_args=vae_compile_flags,
        )

    def testBf16FluxPipeline(self):
        iree_devices = get_iree_devices(driver="hip", device_count=1)
         clip_module, clip_vm_context, clip_vm_instance = load_iree_module(
            module_path = "flux/clip_bf16.vmfb",
            devices = iree_devices,
            parameters_path = "/data/shark/fluxtesting/openai__clip_vit_large_patch14_text_model_bf16.irpa",
         )

        t5_module, t5_vm_context, t5_vm_instance = load_iree_module(
            module_path="flux/t5_bf16.vmfb",
            devices=iree_devices,
            parameters_path="/data/shark/fluxtesting/google__t5_v1_1_small_encoder_bf16.irpa",
        )
        flux_module, flux_vm_context, flux_vm_instance = load_iree_module(
            module_path = "flux/flux_bf16.vmfb",
            devices = iree_devices,
            parameters_path = "/data/shark/fluxtesting/black-forest-labs--FLUX.1-schnell--transformer-single-layer-b16.irpa",
        )
        vae_module, vae_vm_context, vae_vm_instance = load_iree_module(
            module_path="flux/vae_bf16.vmfb",
            devices=iree_devices,
            parameters_path="/data/shark/fluxtesting/vae_bf16.irpa",
        )

        tokenizer = CLIPTokenizer.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            subfolder="tokenizer",
        )
        tokenizer_2 = T5TokenizerFast.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            subfolder="tokenizer_2",
        )

        text_inputs = tokenizer(
            sample_prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.ids
        untruncated_ids = tokenizer(
            sample_prompt, padding="longest", return_tensors="pt"
        ).input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = tokenizer.batch_decode(untruncated_ids[:, 76:-1])

        prompt_embeds = run_iree_module_function(
            module=clip_module,
            vm_context=clip_vm_context,
            args=prepare_iree_module_function_args(
                args=flatten_for_iree_signature(text_input_ids), devices=iree_devices
            ),
            drivers="hip",
            function_name="forward",
        )[0]

        text_inputs_2 = tokenizer_2(
            sample_prompt,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids_2 = text_inputs_2.input_ids
        untruncated_ids_2 = tokenizer_2(
            sample_prompt, padding="longest", return_tensors="pt"
        ).input_ids
        if untruncated_ids.shape[-1] >= text_input_ids_2.shape[-1] and not torch.equal(
            text_input_ids_2, untruncated_ids_2
        ):
            removed_text = tokenizer_2.batch_decode(untrucated_ids_2[:, 511:-1])

        prompt_embeds_2 = run_iree_module_function(
            module=t5_module,
            vm_context=t5_vm_context,
            args=prepare_iree_module_function_args(
                args=flatten_for_iree_signature(text_input_ids_2), devices=iree_devices
            ),
            drivers="hip",
            function_name="forward",
        )[0]
