// Copyright 2026, The FlagOS Contributors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Exact STATE BF16 QKV specialization, gfx936 only. QK/PV use MMAC with
// FP32 accumulators. A Cauchy bound selects a BF16-normal fixed-shift
// softmax; flagged tiles are recomputed by the Triton stable kernel.
// Reuse K/V LDS for the epilogue to coalesce output into 128-bit stores.
#include <cstdint>
#include <hip/hip_runtime.h>

using u2 = uint32_t __attribute__((ext_vector_type(2)));
using u4 = uint32_t __attribute__((ext_vector_type(4)));
using f4 = float __attribute__((ext_vector_type(4)));

__device__ __forceinline__ f4 dot(u2 a, u2 b, f4 c) {
  asm("v_mmac_f32_16x16x16_bf16 %0, %1, %2, %0" : "+v"(c) : "v"(a), "v"(b));
  return c;
}
__device__ __forceinline__ float bf(uint16_t v) {
  return __uint_as_float(uint32_t(v) << 16);
}
__device__ __forceinline__ uint16_t rb(float x) {
  uint32_t z = __float_as_uint(x);
  return (z + 0x7fff + ((z >> 16) & 1)) >> 16;
}
__device__ __forceinline__ u2 ld(const uint16_t *p) {
  return *reinterpret_cast<const u2 *>(p);
}
__device__ __forceinline__ u2 pk(uint16_t a, uint16_t b, uint16_t c,
                                 uint16_t d) {
  return {uint32_t(a) | (uint32_t(b) << 16), uint32_t(c) | (uint32_t(d) << 16)};
}

__device__ __forceinline__ u2 read64(const uint16_t *p) {
  uint32_t addr = uint32_t(uintptr_t(p));
  u2 v;
  asm volatile("ds_read_b64 %0, %1" : "=v"(v) : "v"(addr) : "memory");
  return v;
}
__device__ __forceinline__ void hold_scores(f4 &s, u2 a, u2 b, u2 c, u2 d, u2 e,
                                            u2 f, u2 g, u2 h) {
  asm volatile(""
               : "+v"(s)
               : "v"(a), "v"(b), "v"(c), "v"(d), "v"(e), "v"(f), "v"(g), "v"(h)
               : "memory");
}
__device__ __forceinline__ void wait_values(u2 &a, u2 &b, u2 &c, u2 &d, u2 &e,
                                            u2 &f, u2 &g, u2 &h) {
  asm volatile("s_waitcnt lgkmcnt(0)"
               : "+v"(a), "+v"(b), "+v"(c), "+v"(d), "+v"(e), "+v"(f), "+v"(g),
                 "+v"(h)
               :
               : "memory");
}

__device__ __forceinline__ void wait4(u2 &a, u2 &b, u2 &c, u2 &d) {
  asm volatile("s_waitcnt lgkmcnt(0)"
               : "+v"(a), "+v"(b), "+v"(c), "+v"(d)
               :
               : "memory");
}
__device__ __forceinline__ u4 ld4(const uint16_t *p) {
  return *reinterpret_cast<const u4 *>(p);
}

template <int WARPS, int QT, int BN, int MODE, int KC, int STORE>
__global__ __launch_bounds__(WARPS * 64) void flash_vpackedlds(
    const uint16_t *__restrict__ Q, const uint16_t *__restrict__ K,
    const uint16_t *__restrict__ V, const float *__restrict__ T,
    uint8_t *__restrict__ F, uint16_t *__restrict__ O, float *__restrict__ L) {
  constexpr int M = 2049, D = 128, H = 16, BM = WARPS * QT * 16, KT = BN / 16;
  const int tid = threadIdx.x, lane = tid % 64, wave = tid / 64,
            row = lane % 16, group = lane / 16;
  const int pb = blockIdx.z, ph = blockIdx.y,
            qb = blockIdx.x * BM + wave * (QT * 16);
  const int64_t base = int64_t(pb) * 12589056 + ph * 128;
  __shared__ uint16_t smem[(STORE == 2 ? WARPS * 16 * 136 : BN * (128 + 130))];
  __shared__ float bounds[WARPS];
  auto sk = smem;
  auto sv = smem + BN * D;
  float kmax = 0;
  for (int i = 0; i < 17; ++i)
    kmax = fmaxf(kmax, T[(pb * H + ph) * 17 + i]);
  u2 q[QT][8];
  float shift[QT], denom[QT] = {};
#pragma unroll
  for (int qt = 0; qt < QT; ++qt) {
    float norm = 0;
    int qr = qb + qt * 16 + row;
#pragma unroll
    for (int dt = 0; dt < 8; ++dt) {
      q[qt][dt] = qr < M
                      ? ld(Q + base + int64_t(qr) * 6144 + dt * 16 + group * 4)
                      : u2{0, 0};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        float v = bf(uint16_t(q[qt][dt][j / 2] >> (16 * (j % 2))));
        norm += v * v;
      }
    }
    norm += __shfl_xor(norm, 16, 64);
    norm += __shfl_xor(norm, 32, 64);
    shift[qt] = sqrtf(norm * kmax) * 0.12751743082455375f * 1.00001f;
  }
  float bound = 0;
#pragma unroll
  for (int qi = 0; qi < QT; ++qi)
    bound = fmaxf(bound, shift[qi]);
#pragma unroll
  for (int mask = 32; mask; mask /= 2)
    bound = fmaxf(bound, __shfl_xor(bound, mask, 64));
  if (lane == 0)
    bounds[wave] = bound;
  __syncthreads();
  float block_bound = 0;
#pragma unroll
  for (int wi = 0; wi < WARPS; ++wi)
    block_bound = fmaxf(block_bound, bounds[wi]);
  bool good = block_bound <= 55.f;
  if (tid == 0)
    F[(pb * H + ph) * ((M + BM - 1) / BM) + blockIdx.x] = good;
  if (!good)
    return;
  f4 acc[QT][8];
#pragma unroll
  for (int qt = 0; qt < QT; ++qt) {
#pragma unroll
    for (int dv = 0; dv < 8; ++dv)
      acc[qt][dv] = {0, 0, 0, 0};
  }
#pragma unroll
  for (int j = 0; j < BN * 128 / (WARPS * 64 * 8); ++j) {
    int idx = tid * 8 + j * (WARPS * 64) * 8, kr = idx / 128, dd = idx % 128;
    u4 kv = kr < M ? ld4(K + base + int64_t(kr) * 6144 + dd) : u4{0, 0, 0, 0};
    u4 vv = kr < M ? ld4(V + base + int64_t(kr) * 6144 + dd) : u4{0, 0, 0, 0};
    for (int hp = 0; hp < 2; ++hp)
      *reinterpret_cast<u2 *>(sk + (idx / 128) * 128 +
                              ((dd + hp * 4) ^ ((idx / 128 % 16) * 4))) =
          u2{kv[hp * 2], kv[hp * 2 + 1]};
#pragma unroll
    for (int jp = 0; jp < 8; ++jp) {
      int vr = idx / 128, vc = dd + jp;
      sv[((vr / 16) * 8 * 65 + (vc / 16) * 65 + (vr % 4) * 16 + vc % 16) * 4 +
         (vr % 16) / 4] = uint16_t(vv[jp / 2] >> (16 * (jp % 2)));
    }
  }
  __syncthreads();
  for (int nb = 0; nb < 2048; nb += BN) {
    constexpr int LOADS = BN * 128 / (WARPS * 64 * 8);
    u4 next_k[LOADS], next_v[LOADS];
#pragma unroll
    for (int j = 0; j < LOADS; ++j) {
      int idx = tid * 8 + j * (WARPS * 64) * 8, kr = nb + BN + idx / 128,
          dd = idx % 128;
      if (nb + BN < 2048) {
        next_k[j] = ld4(K + base + int64_t(kr) * 6144 + dd);
        next_v[j] = ld4(V + base + int64_t(kr) * 6144 + dd);
      } else {
        next_k[j] =
            kr < M ? ld4(K + base + int64_t(kr) * 6144 + dd) : u4{0, 0, 0, 0};
        next_v[j] =
            kr < M ? ld4(V + base + int64_t(kr) * 6144 + dd) : u4{0, 0, 0, 0};
      }
    }
    if (qb < M) {
      f4 scores[QT][KT];
#pragma unroll
      for (int qt = 0; qt < QT; ++qt) {
#pragma unroll
        for (int nk = 0; nk < KT; ++nk)
          scores[qt][nk] = {0, 0, 0, 0};
      }
#pragma unroll
      for (int db = 0; db < 8; db += KC) {
        u2 keys[KC][KT];
#pragma unroll
        for (int dk = 0; dk < KC; ++dk) {
#pragma unroll
          for (int nk = 0; nk < KT; ++nk)
            keys[dk][nk] =
                read64(sk + (nk * 16 + row) * 128 +
                       (((db + dk) * 16 + group * 4) ^ ((row % 16) * 4)));
        }
#pragma unroll
        for (int nk = 0; nk < KT; ++nk) {
          if (KC == 8)
            wait_values(keys[0][nk], keys[1][nk], keys[2][nk], keys[3][nk],
                        keys[4][nk], keys[5][nk], keys[6][nk], keys[7][nk]);
          else
            wait4(keys[0][nk], keys[1][nk], keys[2][nk], keys[3][nk]);
        }
#pragma unroll
        for (int dk = 0; dk < KC; ++dk) {
#pragma unroll
          for (int nk = 0; nk < KT; ++nk) {
#pragma unroll
            for (int qt = 0; qt < QT; ++qt)
              scores[qt][nk] =
                  dot(q[qt][db + dk], keys[dk][nk], scores[qt][nk]);
          }
        }
      }
#pragma unroll
      for (int nk = 0; nk < KT; ++nk) {
        u2 values[8];
#pragma unroll
        for (int dv = 0; dv < 8; ++dv)
          values[dv] = read64(sv + (nk * 8 * 65 + dv * 65 + lane) * 4);
        if (MODE) {
#pragma unroll
          for (int qt = 0; qt < QT; ++qt)
            hold_scores(scores[qt][nk], values[0], values[1], values[2],
                        values[3], values[4], values[5], values[6], values[7]);
        }
        u2 pp[QT];
#pragma unroll
        for (int qt = 0; qt < QT; ++qt) {
          uint32_t halves[4];
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            float p = __builtin_amdgcn_exp2f(scores[qt][nk][j] *
                                                 0.12751743082455375f -
                                             (MODE >= 3 ? 0.f : shift[qt]));
            denom[qt] += p;
            uint32_t z = __float_as_uint(p);
            halves[j] = (MODE & 1) ? z + 0x7fff + ((z >> 16) & 1) : z + 0x8000;
          }
          pp[qt] = {__builtin_amdgcn_perm(halves[1], halves[0], 0x07060302),
                    __builtin_amdgcn_perm(halves[3], halves[2], 0x07060302)};
        }
        wait_values(values[0], values[1], values[2], values[3], values[4],
                    values[5], values[6], values[7]);
#pragma unroll
        for (int dv = 0; dv < 8; ++dv) {
          u2 vv = values[dv];
#pragma unroll
          for (int qt = 0; qt < QT; ++qt)
            acc[qt][dv] = dot(pp[qt], vv, acc[qt][dv]);
        }
      }
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < LOADS; ++j) {
      int idx = tid * 8 + j * (WARPS * 64) * 8, dd = idx % 128;
      for (int hp = 0; hp < 2; ++hp)
        *reinterpret_cast<u2 *>(sk + (idx / 128) * 128 +
                                ((dd + hp * 4) ^ ((idx / 128 % 16) * 4))) =
            u2{next_k[j][hp * 2], next_k[j][hp * 2 + 1]};
#pragma unroll
      for (int jp = 0; jp < 8; ++jp) {
        int vr = idx / 128, vc = dd + jp;
        sv[((vr / 16) * 8 * 65 + (vc / 16) * 65 + (vr % 4) * 16 + vc % 16) * 4 +
           (vr % 16) / 4] = uint16_t(next_v[j][jp / 2] >> (16 * (jp % 2)));
      }
    }
    __syncthreads();
  }
  {
    constexpr int nb = 2048;
    constexpr int LOADS = BN * 128 / (WARPS * 64 * 8);
    u4 next_k[LOADS], next_v[LOADS];
#pragma unroll
    for (int j = 0; j < LOADS; ++j) {
      int idx = tid * 8 + j * (WARPS * 64) * 8, kr = nb + BN + idx / 128,
          dd = idx % 128;
      next_k[j] = u4{0, 0, 0, 0};
      next_v[j] = u4{0, 0, 0, 0};
    }
    if (qb < M) {
      f4 scores[QT][KT];
#pragma unroll
      for (int qt = 0; qt < QT; ++qt) {
#pragma unroll
        for (int nk = 0; nk < KT; ++nk)
          scores[qt][nk] = {0, 0, 0, 0};
      }
#pragma unroll
      for (int db = 0; db < 8; db += KC) {
        u2 keys[KC][KT];
#pragma unroll
        for (int dk = 0; dk < KC; ++dk) {
#pragma unroll
          for (int nk = 0; nk < KT; ++nk)
            keys[dk][nk] =
                read64(sk + (nk * 16 + row) * 128 +
                       (((db + dk) * 16 + group * 4) ^ ((row % 16) * 4)));
        }
#pragma unroll
        for (int nk = 0; nk < KT; ++nk) {
          if (KC == 8)
            wait_values(keys[0][nk], keys[1][nk], keys[2][nk], keys[3][nk],
                        keys[4][nk], keys[5][nk], keys[6][nk], keys[7][nk]);
          else
            wait4(keys[0][nk], keys[1][nk], keys[2][nk], keys[3][nk]);
        }
#pragma unroll
        for (int dk = 0; dk < KC; ++dk) {
#pragma unroll
          for (int nk = 0; nk < KT; ++nk) {
#pragma unroll
            for (int qt = 0; qt < QT; ++qt)
              scores[qt][nk] =
                  dot(q[qt][db + dk], keys[dk][nk], scores[qt][nk]);
          }
        }
      }
#pragma unroll
      for (int nk = 0; nk < KT; ++nk) {
        u2 values[8];
#pragma unroll
        for (int dv = 0; dv < 8; ++dv)
          values[dv] = read64(sv + (nk * 8 * 65 + dv * 65 + lane) * 4);
        if (MODE) {
#pragma unroll
          for (int qt = 0; qt < QT; ++qt)
            hold_scores(scores[qt][nk], values[0], values[1], values[2],
                        values[3], values[4], values[5], values[6], values[7]);
        }
        u2 pp[QT];
#pragma unroll
        for (int qt = 0; qt < QT; ++qt) {
          uint32_t halves[4];
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            float p = nb + nk * 16 + group + j * 4 < M
                          ? __builtin_amdgcn_exp2f(
                                scores[qt][nk][j] * 0.12751743082455375f -
                                (MODE >= 3 ? 0.f : shift[qt]))
                          : 0.f;
            denom[qt] += p;
            uint32_t z = __float_as_uint(p);
            halves[j] = (MODE & 1) ? z + 0x7fff + ((z >> 16) & 1) : z + 0x8000;
          }
          pp[qt] = {__builtin_amdgcn_perm(halves[1], halves[0], 0x07060302),
                    __builtin_amdgcn_perm(halves[3], halves[2], 0x07060302)};
        }
        wait_values(values[0], values[1], values[2], values[3], values[4],
                    values[5], values[6], values[7]);
#pragma unroll
        for (int dv = 0; dv < 8; ++dv) {
          u2 vv = values[dv];
#pragma unroll
          for (int qt = 0; qt < QT; ++qt)
            acc[qt][dv] = dot(pp[qt], vv, acc[qt][dv]);
        }
      }
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < LOADS; ++j) {
      int idx = tid * 8 + j * (WARPS * 64) * 8, dd = idx % 128;
      for (int hp = 0; hp < 2; ++hp)
        *reinterpret_cast<u2 *>(sk + (idx / 128) * 128 +
                                ((dd + hp * 4) ^ ((idx / 128 % 16) * 4))) =
            u2{next_k[j][hp * 2], next_k[j][hp * 2 + 1]};
#pragma unroll
      for (int jp = 0; jp < 8; ++jp) {
        int vr = idx / 128, vc = dd + jp;
        sv[((vr / 16) * 8 * 65 + (vc / 16) * 65 + (vr % 4) * 16 + vc % 16) * 4 +
           (vr % 16) / 4] = uint16_t(next_v[j][jp / 2] >> (16 * (jp % 2)));
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int qt = 0; qt < QT; ++qt) {
    float den = denom[qt];
    den += __shfl_xor(den, 16, 64);
    den += __shfl_xor(den, 32, 64);
    int qr = qb + qt * 16 + row;
    if (qr < M) {
#pragma unroll
      for (int dv = 0; dv < 8; ++dv) {
        if (STORE == 1) {
          uint32_t p0 = uint32_t(rb(acc[qt][dv][0] / den)) |
                        (uint32_t(rb(acc[qt][dv][1] / den)) << 16);
          uint32_t p1 = uint32_t(rb(acc[qt][dv][2] / den)) |
                        (uint32_t(rb(acc[qt][dv][3] / den)) << 16);
          uint32_t o0 = __shfl_xor(p0, 16, 64), o1 = __shfl_xor(p1, 16, 64);
          uint32_t r0 = (group & 1) ? ((o0 >> 16) | (p0 & 0xffff0000u))
                                    : ((p0 & 0xffffu) | (o0 << 16));
          uint32_t r1 = (group & 1) ? ((o1 >> 16) | (p1 & 0xffff0000u))
                                    : ((p1 & 0xffffu) | (o1 << 16));
          uint32_t s0 = __shfl_xor(r0, 32, 64), s1 = __shfl_xor(r1, 32, 64);
          u2 vec = (group & 2) ? u2{s1, r1} : u2{r0, s0};
          *reinterpret_cast<u2 *>(O + ((pb * H + ph) * M + qr) * D + dv * 16 +
                                  group * 4) = vec;
        } else {
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            uint16_t val = rb(acc[qt][dv][j] / den);
            if (STORE == 2)
              smem[(wave * 16 + row) * 136 + dv * 16 + group + j * 4] = val;
            else
              O[((pb * H + ph) * M + qr) * D + dv * 16 + group + j * 4] = val;
          }
        }
      }
      if (group == 0)
        L[(pb * H + ph) * M + qr] =
            ((MODE >= 3 ? 0.f : shift[qt]) + log2f(den)) * 0.6931471805599453f;
    }
    if (STORE == 2) {
      __syncthreads();
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        int idx = tid * 8 + j * WARPS * 64 * 8;
        if (idx < WARPS * 16 * 128) {
          int rr = idx / 128, cc = idx % 128;
          int oq = blockIdx.x * BM + (rr / 16) * (QT * 16) + qt * 16 + rr % 16;
          if (oq < M)
            *reinterpret_cast<u4 *>(O + ((pb * H + ph) * M + oq) * 128 + cc) =
                ld4(smem + rr * 136 + cc);
        }
      }
      __syncthreads();
    }
  }
}

extern "C" int launch_state_flash(const void *q, const void *k, const void *v,
                                  const float *t, uint8_t *f, void *o, float *l,
                                  hipStream_t stream) {
  hipLaunchKernelGGL((flash_vpackedlds<4, 3, 16, 1, 8, 2>), dim3(11, 16, 8),
                     dim3(256), 0, stream, static_cast<const uint16_t *>(q),
                     static_cast<const uint16_t *>(k),
                     static_cast<const uint16_t *>(v), t, f,
                     static_cast<uint16_t *>(o), l);
  return hipGetLastError();
}
