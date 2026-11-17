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

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def rsqrt_func(x):
    x_fp32 = x.to(tl.float32)
    bits = x_fp32.to(tl.uint32, bitcast=True)
    magnitude = bits & 0x7FFFFFFF
    is_subnormal = (magnitude > 0) & (magnitude < 0x00800000)
    mantissa = bits & 0x007FFFFF
    # Rebuild subnormal magnitude * 2**24 without floating subnormal input.
    scaled = mantissa.to(tl.float32) * 2.350988701644575e-38  # 2**-125
    rsqrt_input = tl.where(is_subnormal, scaled, x_fp32)
    result = tl.rsqrt(rsqrt_input)
    result = tl.where(is_subnormal, result * 4096.0, result)
    negative_subnormal = is_subnormal & ((bits >> 31) != 0)
    return tl.where(negative_subnormal, float("nan"), result)


def rsqrt(A):
    logger.debug("GEMS RSQRT")
    return rsqrt_func(A)


def rsqrt_(A):
    logger.debug("GEMS RSQRT_")
    return rsqrt_func(A, out0=A)
