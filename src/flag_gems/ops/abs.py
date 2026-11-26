# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic
from flag_gems.utils.triton_lang_helper import tl_extra_shim

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "COMPLEX_TO_FLOAT")])
@triton.jit
def abs_func(x):
    return tl.abs(x)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def complex_abs_func(real, imag):
    return tl_extra_shim.hypot(real, imag)


def abs(A):
    logger.debug("GEMS ABS")
    if A.is_complex():
        # Conjugation does not change the magnitude.
        parts = torch.view_as_real(torch.conj(A) if A.is_conj() else A)
        return complex_abs_func(parts[..., 0], parts[..., 1])
    return abs_func(A)


def abs_(A):
    logger.debug("GEMS ABS_")
    abs_func(A, out0=A)
    return A
