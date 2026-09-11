
#include "gpu_hierarchy.h"
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdint>
#include <cinttypes>
#include <cstring>
#include <algorithm>
#include <vector>

struct U256 { uint64_t v[4]; };

__device__ __constant__ uint64_t P_LIMBS[4] = {
    0xFFFFFFFEFFFFFC2FULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL
};

#define SECP_C 0x1000003D1ULL

__device__ __forceinline__ int u256_is_zero(const U256 &a){
    return (a.v[0]|a.v[1]|a.v[2]|a.v[3])==0;
}

__device__ __forceinline__ int u256_eq(const U256 &a, const U256 &b){
    return a.v[0]==b.v[0] && a.v[1]==b.v[1] && a.v[2]==b.v[2] && a.v[3]==b.v[3];
}

__device__ __forceinline__ int u256_geq(const U256 &a, const U256 &b){
    for(int i=3;i>=0;--i){
        if(a.v[i]!=b.v[i]) return a.v[i]>b.v[i];
    }
    return 1;
}

__device__ __forceinline__ uint64_t u256_add(U256 &a, const U256 &b){
    unsigned __int128 carry = 0;
    for(int i=0;i<4;++i){
        unsigned __int128 s = (unsigned __int128)a.v[i] + b.v[i] + carry;
        a.v[i] = (uint64_t)s;
        carry = s >> 64;
    }
    return (uint64_t)carry;
}

__device__ __forceinline__ uint64_t u256_sub(U256 &a, const U256 &b){
    unsigned __int128 borrow = 0;
    for(int i=0;i<4;++i){
        unsigned __int128 d = (unsigned __int128)a.v[i] - b.v[i] - borrow;
        a.v[i] = (uint64_t)d;
        borrow = (d >> 64) & 1;
    }
    return (uint64_t)borrow;
}

__device__ __forceinline__ U256 make_p(){
    U256 p; p.v[0]=P_LIMBS[0]; p.v[1]=P_LIMBS[1]; p.v[2]=P_LIMBS[2]; p.v[3]=P_LIMBS[3];
    return p;
}

__device__ __forceinline__ void reduce_once(U256 &a){
    U256 p = make_p();
    if(u256_geq(a,p)) u256_sub(a,p);
}

__device__ void field_add(U256 &r, const U256 &a, const U256 &b){
    r = a;
    uint64_t carry = u256_add(r,b);

    if(carry){ U256 p=make_p(); u256_sub(r,p); }
    reduce_once(r);
}

__device__ void field_sub(U256 &r, const U256 &a, const U256 &b){
    r = a;
    uint64_t borrow = u256_sub(r,b);
    if(borrow){ U256 p=make_p(); u256_add(r,p); }
}

__device__ void u256_mul_512(uint64_t *out , const U256 &a, const U256 &b){
    for(int i=0;i<8;++i) out[i]=0;
    for(int i=0;i<4;++i){
        unsigned __int128 carry = 0;
        for(int j=0;j<4;++j){
            unsigned __int128 prod = (unsigned __int128)a.v[i]*b.v[j];
            unsigned __int128 sum = (unsigned __int128)out[i+j] + (uint64_t)prod + carry;
            out[i+j] = (uint64_t)sum;
            carry = (prod>>64) + (sum>>64);
        }

        int k = i+4;
        while(carry){
            unsigned __int128 sum = (unsigned __int128)out[k] + (uint64_t)carry;
            out[k] = (uint64_t)sum;
            carry = (carry>>64) + (sum>>64);
            ++k;
        }
    }
}

__device__ void field_mul(U256 &r, const U256 &a, const U256 &b){
    uint64_t m[8];
    u256_mul_512(m, a, b);

    U256 lo; lo.v[0]=m[0]; lo.v[1]=m[1]; lo.v[2]=m[2]; lo.v[3]=m[3];
    U256 hi; hi.v[0]=m[4]; hi.v[1]=m[5]; hi.v[2]=m[6]; hi.v[3]=m[7];

    uint64_t hc[5] = {0,0,0,0,0};
    unsigned __int128 carry = 0;
    for(int i=0;i<4;++i){
        unsigned __int128 prod = (unsigned __int128)hi.v[i]*SECP_C + carry;
        hc[i] = (uint64_t)prod;
        carry = prod>>64;
    }
    hc[4] = (uint64_t)carry;

    U256 sum; sum.v[0]=lo.v[0]; sum.v[1]=lo.v[1]; sum.v[2]=lo.v[2]; sum.v[3]=lo.v[3];
    U256 hc_low; hc_low.v[0]=hc[0]; hc_low.v[1]=hc[1]; hc_low.v[2]=hc[2]; hc_low.v[3]=hc[3];
    uint64_t add_carry = u256_add(sum, hc_low);
    uint64_t overflow_limb = hc[4] + add_carry;

    unsigned __int128 fold = (unsigned __int128)overflow_limb * SECP_C;
    U256 fold_u; fold_u.v[0]=(uint64_t)fold; fold_u.v[1]=(uint64_t)(fold>>64); fold_u.v[2]=0; fold_u.v[3]=0;
    u256_add(sum, fold_u);

    U256 p = make_p();
    while(u256_geq(sum,p)) u256_sub(sum,p);
    r = sum;
}

__device__ void field_sqr(U256 &r, const U256 &a){ field_mul(r,a,a); }

__device__ __constant__ uint64_t P_MINUS_2[4] = {
    0xFFFFFFFEFFFFFC2DULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL
};
__device__ void field_inv(U256 &r, const U256 &a){
    U256 result; result.v[0]=1; result.v[1]=0; result.v[2]=0; result.v[3]=0;
    U256 base = a;
    for(int limb=0; limb<4; ++limb){
        uint64_t e = P_MINUS_2[limb];
        for(int bit=0; bit<64; ++bit){
            if((e>>bit)&1){
                U256 tmp; field_mul(tmp, result, base);
                result = tmp;
            }
            U256 sq; field_sqr(sq, base);
            base = sq;
        }
    }
    r = result;
}

struct ECPoint { U256 x, y; bool infinity; };

__device__ void point_double(ECPoint &r, const ECPoint &p){
    if(p.infinity || u256_is_zero(p.y)){ r.infinity = true; return; }

    U256 x2; field_sqr(x2, p.x);
    U256 three_x2; { U256 two_x2; field_add(two_x2, x2, x2); field_add(three_x2, two_x2, x2); }
    U256 two_y; field_add(two_y, p.y, p.y);
    U256 inv_2y; field_inv(inv_2y, two_y);
    U256 lambda; field_mul(lambda, three_x2, inv_2y);

    U256 lambda2; field_sqr(lambda2, lambda);
    U256 two_x; field_add(two_x, p.x, p.x);
    U256 x3; field_sub(x3, lambda2, two_x);

    U256 x1_minus_x3; field_sub(x1_minus_x3, p.x, x3);
    U256 lam_term; field_mul(lam_term, lambda, x1_minus_x3);
    U256 y3; field_sub(y3, lam_term, p.y);

    r.x = x3; r.y = y3; r.infinity = false;
}

__device__ void point_add(ECPoint &r, const ECPoint &a, const ECPoint &b){
    if(a.infinity){ r = b; return; }
    if(b.infinity){ r = a; return; }
    if(u256_eq(a.x, b.x)){

        U256 y_sum; field_add(y_sum, a.y, b.y);
        if(u256_is_zero(y_sum)){ r.infinity = true; return; }
        point_double(r, a);
        return;
    }

    U256 dy; field_sub(dy, b.y, a.y);
    U256 dx; field_sub(dx, b.x, a.x);
    U256 inv_dx; field_inv(inv_dx, dx);
    U256 lambda; field_mul(lambda, dy, inv_dx);

    U256 lambda2; field_sqr(lambda2, lambda);
    U256 x1x2; field_add(x1x2, a.x, b.x);
    U256 x3; field_sub(x3, lambda2, x1x2);

    U256 x1_minus_x3; field_sub(x1_minus_x3, a.x, x3);
    U256 lam_term; field_mul(lam_term, lambda, x1_minus_x3);
    U256 y3; field_sub(y3, lam_term, a.y);

    r.x = x3; r.y = y3; r.infinity = false;
}

__device__ void scalar_mult(ECPoint &r, const uint64_t k[4], const ECPoint &p){
    ECPoint result; result.infinity = true;
    ECPoint addend = p;
    for(int limb=0; limb<4; ++limb){
        uint64_t e = k[limb];
        for(int bit=0; bit<64; ++bit){
            if((e>>bit)&1){
                ECPoint tmp; point_add(tmp, result, addend);
                result = tmp;
            }
            ECPoint dbl; point_double(dbl, addend);
            addend = dbl;
        }
    }
    r = result;
}

struct ProjPoint { U256 X, Y, Z; };

__device__ void mixed_add_projective(ProjPoint &r, const ProjPoint &p1, const ECPoint &p2){
    U256 u; { U256 t; field_mul(t, p2.y, p1.Z); field_sub(u, t, p1.Y); }
    U256 v; { U256 t; field_mul(t, p2.x, p1.Z); field_sub(v, t, p1.X); }
    U256 v2; field_sqr(v2, v);
    U256 v3; field_mul(v3, v2, v);
    U256 A;
    {
        U256 u2; field_sqr(u2, u);
        U256 u2z1; field_mul(u2z1, u2, p1.Z);
        U256 two_v2x1; { U256 v2x1; field_mul(v2x1, v2, p1.X); field_add(two_v2x1, v2x1, v2x1); }
        U256 tmp; field_sub(tmp, u2z1, v3);
        field_sub(A, tmp, two_v2x1);
    }
    U256 X3; field_mul(X3, v, A);
    U256 Y3;
    {
        U256 v2x1; field_mul(v2x1, v2, p1.X);
        U256 inner; field_sub(inner, v2x1, A);
        U256 u_inner; field_mul(u_inner, u, inner);
        U256 v3y1; field_mul(v3y1, v3, p1.Y);
        field_sub(Y3, u_inner, v3y1);
    }
    U256 Z3; field_mul(Z3, v3, p1.Z);
    r.X = X3; r.Y = Y3; r.Z = Z3;
}

struct GpuBloom {
    uint64_t *d_bits;
    uint64_t bit_count, nwords, block_bits;
    uint32_t k_hashes;
};

__device__ __forceinline__ void bloom_positions_real(uint64_t bit_count, uint32_t k_hashes,
                                                       uint64_t block_bits, uint64_t fp, uint64_t *out){
    uint64_t h1 = fp;
    uint64_t h2 = fp + 0x9e3779b97f4a7c15ULL;
    h2 = (h2^(h2>>30)) * 0xbf58476d1ce4e5b9ULL;
    h2 = (h2^(h2>>27)) * 0x94d049bb133111ebULL;
    h2 ^= h2>>31; if(h2==0) h2=1;
    if(block_bits>0){
        uint64_t num_blocks = bit_count / block_bits;
        if(num_blocks<1) num_blocks=1;
        uint64_t block = h1 % num_blocks;
        uint64_t block_base_bit = block * block_bits;
        for(uint32_t i=0;i<k_hashes;++i){
            uint64_t sub = (h1 + (uint64_t)i*h2) % block_bits;
            out[i] = block_base_bit + sub;
        }
    } else {
        for(uint32_t i=0;i<k_hashes;++i) out[i] = (h1 + (uint64_t)i*h2) % bit_count;
    }
}

__device__ int bloom_check_real(const uint64_t *bits, uint64_t bit_count, uint32_t k_hashes,
                                 uint64_t block_bits, uint64_t fp){
    uint64_t positions[32];
    bloom_positions_real(bit_count, k_hashes, block_bits, fp, positions);
    for(uint32_t i=0;i<k_hashes;++i){
        uint64_t pos = positions[i];
        if(!(bits[pos>>6] & (1ULL<<(pos&63)))) return 0;
    }
    return 1;
}

__global__ void single_check_kernel(const uint64_t *bits, uint64_t bit_count, uint32_t k_hashes,
                                     uint64_t block_bits, uint64_t fp, int *out_result){
    *out_result = bloom_check_real(bits, bit_count, k_hashes, block_bits, fp);
}

extern "C" {

GpuBloom* gpu_bloom_upload(const uint64_t *host_bits, uint64_t bit_count,
                           uint64_t nwords, uint32_t k_hashes, uint64_t block_bits){
    GpuBloom *b = new GpuBloom();
    b->bit_count = bit_count; b->nwords = nwords; b->k_hashes = k_hashes; b->block_bits = block_bits;
    cudaError_t err = cudaMalloc(&b->d_bits, nwords*sizeof(uint64_t));
    if(err != cudaSuccess){
        fprintf(stderr, "[gpu_hierarchy] gpu_bloom_upload: cudaMalloc failed for %.2f GB (%s)\n",
                nwords*8.0/1e9, cudaGetErrorString(err));
        delete b;
        return nullptr;
    }
    err = cudaMemcpy(b->d_bits, host_bits, nwords*sizeof(uint64_t), cudaMemcpyHostToDevice);
    if(err != cudaSuccess){
        fprintf(stderr, "[gpu_hierarchy] gpu_bloom_upload: cudaMemcpy failed (%s)\n", cudaGetErrorString(err));
        cudaFree(b->d_bits);
        delete b;
        return nullptr;
    }
    return b;
}

void gpu_bloom_free(GpuBloom *b){
    if(!b) return;
    if(b->d_bits) cudaFree(b->d_bits);
    delete b;
}

int gpu_bloom_check_single(GpuBloom *b, uint64_t fingerprint){
    if(!b) return -1;
    int *d_result;
    cudaMalloc(&d_result, sizeof(int));
    single_check_kernel<<<1,1>>>(b->d_bits, b->bit_count, b->k_hashes, b->block_bits, fingerprint, d_result);
    cudaError_t err = cudaDeviceSynchronize();
    if(err != cudaSuccess){
        fprintf(stderr, "[gpu_hierarchy] gpu_bloom_check_single: kernel failed (%s)\n", cudaGetErrorString(err));
        cudaFree(d_result);
        return -1;
    }
    int h_result;
    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    return h_result;
}

void gpu_query_device_info(char *out_buf, int buflen){
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if(err != cudaSuccess || device_count==0){
        snprintf(out_buf, buflen, "no usable CUDA GPU found (%s)",
                 err!=cudaSuccess ? cudaGetErrorString(err) : "device count is 0");
        return;
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    snprintf(out_buf, buflen, "%s (%.1f GB VRAM, compute capability %d.%d)",
             prop.name, prop.totalGlobalMem/1e9, prop.major, prop.minor);
}

int gpu_query_device_count(){
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if(err != cudaSuccess) return 0;
    return device_count;
}

int gpu_set_device(int device_id){
    cudaError_t err = cudaSetDevice(device_id);
    if(err != cudaSuccess){
        fprintf(stderr, "[gpu_hierarchy] gpu_set_device(%d) failed: %s\n", device_id, cudaGetErrorString(err));
        return 0;
    }
    return 1;
}

void gpu_query_device_info_n(int device_id, char *out_buf, int buflen){
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
    if(err != cudaSuccess){
        snprintf(out_buf, buflen, "device %d: error (%s)", device_id, cudaGetErrorString(err));
        return;
    }
    snprintf(out_buf, buflen, "%s (%.1f GB VRAM, compute capability %d.%d)",
             prop.name, prop.totalGlobalMem/1e9, prop.major, prop.minor);
}

}

#define I2_N_JUMPS 32
#define I2_LANES_PER_BLOCK 256
#define I2_POINTS_PER_BLOCK (I2_LANES_PER_BLOCK/I2_N_JUMPS)

struct GpuShiftTable { ECPoint *d_points; int count; };

__device__ __forceinline__ void bloom_positions_i2loop(uint64_t bit_count, uint32_t k_hashes,
                                                         uint64_t block_bits, uint64_t fp, uint64_t *out){
    bloom_positions_real(bit_count, k_hashes, block_bits, fp, out);
}
__device__ int bloom_check_i2loop(const uint64_t *bits, uint64_t bit_count, uint32_t k_hashes,
                                   uint64_t block_bits, uint64_t fp){
    return bloom_check_real(bits, bit_count, k_hashes, block_bits, fp);
}
__device__ uint64_t fingerprint_gpu_h(const U256 &x){

    uint64_t v = x.v[3];
    v += 0x9e3779b97f4a7c15ULL;
    v = (v^(v>>30)) * 0xbf58476d1ce4e5b9ULL;
    v = (v^(v>>27)) * 0x94d049bb133111ebULL;
    v ^= v>>31;
    return v;
}

__global__ void i2loop_kernel_real(ECPoint *shift2, ECPoint *R_points,
                                    uint64_t *bloom2_bits, uint64_t bloom2_bitcount, uint32_t bloom2_k,
                                    uint64_t bloom2_block_bits, int *out_found_i2){
    __shared__ U256 prefix[I2_LANES_PER_BLOCK];
    __shared__ U256 suffix[I2_LANES_PER_BLOCK];
    __shared__ U256 total_inv;
    int tid = threadIdx.x;
    int local_point_idx = tid / I2_N_JUMPS;
    int point_idx = blockIdx.x * I2_POINTS_PER_BLOCK + local_point_idx;
    int i2 = tid % I2_N_JUMPS;

    ECPoint R = R_points[point_idx];
    ProjPoint shifted;
    U256 one; one.v[0]=1; one.v[1]=0; one.v[2]=0; one.v[3]=0;

    if(i2==0){
        shifted.X = R.x; shifted.Y = R.y; shifted.Z = one;
    } else {
        ProjPoint Rproj; Rproj.X=R.x; Rproj.Y=R.y; Rproj.Z = one;
        mixed_add_projective(shifted, Rproj, shift2[i2]);
    }

    prefix[tid] = shifted.Z; suffix[tid] = shifted.Z;
    __syncthreads();
    for(int d=1; d<I2_LANES_PER_BLOCK; d*=2){
        U256 val; bool active=(tid>=d);
        if(active) field_mul(val, prefix[tid], prefix[tid-d]);
        __syncthreads();
        if(active) prefix[tid]=val;
        __syncthreads();
    }
    for(int d=1; d<I2_LANES_PER_BLOCK; d*=2){
        U256 val; bool active=(tid+d<I2_LANES_PER_BLOCK);
        if(active) field_mul(val, suffix[tid], suffix[tid+d]);
        __syncthreads();
        if(active) suffix[tid]=val;
        __syncthreads();
    }
    if(tid==0) field_inv(total_inv, prefix[I2_LANES_PER_BLOCK-1]);
    __syncthreads();
    U256 left = (tid>0) ? prefix[tid-1] : one;
    U256 right = (tid<I2_LANES_PER_BLOCK-1) ? suffix[tid+1] : one;
    U256 lr; field_mul(lr, left, right);
    U256 zinv; field_mul(zinv, total_inv, lr);
    U256 result_x, result_y;
    field_mul(result_x, shifted.X, zinv);
    field_mul(result_y, shifted.Y, zinv);

    uint64_t fp2 = fingerprint_gpu_h(result_x);
    if(bloom_check_i2loop(bloom2_bits, bloom2_bitcount, bloom2_k, bloom2_block_bits, fp2)){
        out_found_i2[point_idx] = i2;
    }
}

__global__ void i2loop_kernel_debug(ECPoint *shift2, ECPoint *R_points, U256 *out_result_x){
    __shared__ U256 prefix[I2_LANES_PER_BLOCK];
    __shared__ U256 suffix[I2_LANES_PER_BLOCK];
    __shared__ U256 total_inv;
    int tid = threadIdx.x;
    int local_point_idx = tid / I2_N_JUMPS;
    int point_idx = blockIdx.x * I2_POINTS_PER_BLOCK + local_point_idx;
    int i2 = tid % I2_N_JUMPS;

    ECPoint R = R_points[point_idx];
    ProjPoint shifted;
    U256 one; one.v[0]=1; one.v[1]=0; one.v[2]=0; one.v[3]=0;

    if(i2==0){
        shifted.X = R.x; shifted.Y = R.y; shifted.Z = one;
    } else {
        ProjPoint Rproj; Rproj.X=R.x; Rproj.Y=R.y; Rproj.Z = one;
        mixed_add_projective(shifted, Rproj, shift2[i2]);
    }

    prefix[tid] = shifted.Z; suffix[tid] = shifted.Z;
    __syncthreads();
    for(int d=1; d<I2_LANES_PER_BLOCK; d*=2){
        U256 val; bool active=(tid>=d);
        if(active) field_mul(val, prefix[tid], prefix[tid-d]);
        __syncthreads();
        if(active) prefix[tid]=val;
        __syncthreads();
    }
    for(int d=1; d<I2_LANES_PER_BLOCK; d*=2){
        U256 val; bool active=(tid+d<I2_LANES_PER_BLOCK);
        if(active) field_mul(val, suffix[tid], suffix[tid+d]);
        __syncthreads();
        if(active) suffix[tid]=val;
        __syncthreads();
    }
    if(tid==0) field_inv(total_inv, prefix[I2_LANES_PER_BLOCK-1]);
    __syncthreads();
    U256 left = (tid>0) ? prefix[tid-1] : one;
    U256 right = (tid<I2_LANES_PER_BLOCK-1) ? suffix[tid+1] : one;
    U256 lr; field_mul(lr, left, right);
    U256 zinv; field_mul(zinv, total_inv, lr);
    U256 result_x;
    field_mul(result_x, shifted.X, zinv);
    out_result_x[tid] = result_x;
}

extern "C" {

GpuShiftTable* gpu_shift_table_upload(const GpuECPoint *host_points, int count){
    if(count != I2_N_JUMPS){
        fprintf(stderr, "[gpu_hierarchy] gpu_shift_table_upload: count must be %d, got %d\n", I2_N_JUMPS, count);
        return nullptr;
    }
    ECPoint *h_conv = new ECPoint[count];
    for(int i=0;i<count;++i){
        for(int j=0;j<4;++j){ h_conv[i].x.v[j]=host_points[i].x[j]; h_conv[i].y.v[j]=host_points[i].y[j]; }
        h_conv[i].infinity=false;
    }
    GpuShiftTable *t = new GpuShiftTable();
    t->count = count;
    cudaError_t err = cudaMalloc(&t->d_points, count*sizeof(ECPoint));
    if(err != cudaSuccess){
        fprintf(stderr, "[gpu_hierarchy] gpu_shift_table_upload: cudaMalloc failed (%s)\n", cudaGetErrorString(err));
        delete[] h_conv; delete t;
        return nullptr;
    }
    cudaMemcpy(t->d_points, h_conv, count*sizeof(ECPoint), cudaMemcpyHostToDevice);
    delete[] h_conv;
    return t;
}

void gpu_shift_table_free(GpuShiftTable *t){
    if(!t) return;
    if(t->d_points) cudaFree(t->d_points);
    delete t;
}

int gpu_i2loop_check_single(GpuShiftTable *shift_table, GpuBloom *bloom2, const GpuECPoint *R){
    if(!shift_table || !bloom2) return -2;
    ECPoint h_R;
    for(int j=0;j<4;++j){ h_R.x.v[j]=R->x[j]; h_R.y.v[j]=R->y[j]; }
    h_R.infinity=false;

    ECPoint *d_R;
    cudaMalloc(&d_R, I2_POINTS_PER_BLOCK*sizeof(ECPoint));
    ECPoint h_R_padded[I2_POINTS_PER_BLOCK];
    for(int i=0;i<I2_POINTS_PER_BLOCK;++i) h_R_padded[i] = h_R;
    cudaMemcpy(d_R, h_R_padded, I2_POINTS_PER_BLOCK*sizeof(ECPoint), cudaMemcpyHostToDevice);

    int *d_found;
    cudaMalloc(&d_found, I2_POINTS_PER_BLOCK*sizeof(int));
    int neg1[I2_POINTS_PER_BLOCK]; for(int i=0;i<I2_POINTS_PER_BLOCK;++i) neg1[i]=-1;
    cudaMemcpy(d_found, neg1, I2_POINTS_PER_BLOCK*sizeof(int), cudaMemcpyHostToDevice);

    i2loop_kernel_real<<<1, I2_LANES_PER_BLOCK>>>(shift_table->d_points, d_R,
                                                    bloom2->d_bits, bloom2->bit_count, bloom2->k_hashes,
                                                    bloom2->block_bits, d_found);
    cudaError_t err = cudaDeviceSynchronize();
    if(err != cudaSuccess){
        fprintf(stderr, "[gpu_hierarchy] gpu_i2loop_check_single: kernel failed (%s)\n", cudaGetErrorString(err));
        cudaFree(d_R); cudaFree(d_found);
        return -2;
    }
    int h_found;
    cudaMemcpy(&h_found, d_found, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_R); cudaFree(d_found);
    return h_found;
}

void gpu_i2loop_debug(GpuShiftTable *shift_table, const GpuECPoint *R, uint64_t *out_results){
    ECPoint h_R;
    for(int j=0;j<4;++j){ h_R.x.v[j]=R->x[j]; h_R.y.v[j]=R->y[j]; }
    h_R.infinity=false;
    ECPoint *d_R;
    cudaMalloc(&d_R, I2_POINTS_PER_BLOCK*sizeof(ECPoint));
    ECPoint h_R_padded[I2_POINTS_PER_BLOCK];
    for(int i=0;i<I2_POINTS_PER_BLOCK;++i) h_R_padded[i] = h_R;
    cudaMemcpy(d_R, h_R_padded, I2_POINTS_PER_BLOCK*sizeof(ECPoint), cudaMemcpyHostToDevice);

    U256 *d_results;
    cudaMalloc(&d_results, I2_LANES_PER_BLOCK*sizeof(U256));

    i2loop_kernel_debug<<<1, I2_LANES_PER_BLOCK>>>(shift_table->d_points, d_R, d_results);
    cudaDeviceSynchronize();

    U256 h_results[I2_LANES_PER_BLOCK];
    cudaMemcpy(h_results, d_results, I2_LANES_PER_BLOCK*sizeof(U256), cudaMemcpyDeviceToHost);

    for(int i2=0;i2<32;++i2){
        for(int j=0;j<4;++j) out_results[i2*4+j] = h_results[i2].v[j];
    }
    cudaFree(d_R); cudaFree(d_results);
}

}

struct GpuBabyTable { GpuBabyEntry *d_table; uint64_t array_size; int K; };

__device__ __forceinline__ uint32_t baby_index_gpu(uint64_t fp, int K){
    return (uint32_t)(fp >> (64-K));
}
__device__ int32_t baby_find_gpu(const GpuBabyEntry *table, uint64_t array_size, int K, uint64_t fp){
    uint32_t pos = baby_index_gpu(fp, K);
    for(uint64_t probes=0; probes<array_size; ++probes){
        if(table[pos].idx==UINT32_MAX) return -1;
        if(table[pos].fp==fp) return (int32_t)pos;
        pos = (uint32_t)((pos+1) & (array_size-1));
    }
    return -1;
}

__global__ void i3loop_and_exact_kernel(ECPoint *shift3, ECPoint *R_points,
                                         uint64_t *bloom3_bits, uint64_t bloom3_bitcount, uint32_t bloom3_k,
                                         uint64_t bloom3_block_bits,
                                         const GpuBabyEntry *baby_table, uint64_t baby_array_size, int baby_K,
                                         int *out_found_i3, int *out_found_idx){
    __shared__ U256 prefix[I2_LANES_PER_BLOCK];
    __shared__ U256 suffix[I2_LANES_PER_BLOCK];
    __shared__ U256 total_inv;
    int tid = threadIdx.x;
    int local_point_idx = tid / I2_N_JUMPS;
    int point_idx = blockIdx.x * I2_POINTS_PER_BLOCK + local_point_idx;
    int i3 = tid % I2_N_JUMPS;

    ECPoint R = R_points[point_idx];
    ProjPoint shifted;
    U256 one; one.v[0]=1; one.v[1]=0; one.v[2]=0; one.v[3]=0;

    if(i3==0){
        shifted.X = R.x; shifted.Y = R.y; shifted.Z = one;
    } else {
        ProjPoint Rproj; Rproj.X=R.x; Rproj.Y=R.y; Rproj.Z = one;
        mixed_add_projective(shifted, Rproj, shift3[i3]);
    }

    prefix[tid] = shifted.Z; suffix[tid] = shifted.Z;
    __syncthreads();
    for(int d=1; d<I2_LANES_PER_BLOCK; d*=2){
        U256 val; bool active=(tid>=d);
        if(active) field_mul(val, prefix[tid], prefix[tid-d]);
        __syncthreads();
        if(active) prefix[tid]=val;
        __syncthreads();
    }
    for(int d=1; d<I2_LANES_PER_BLOCK; d*=2){
        U256 val; bool active=(tid+d<I2_LANES_PER_BLOCK);
        if(active) field_mul(val, suffix[tid], suffix[tid+d]);
        __syncthreads();
        if(active) suffix[tid]=val;
        __syncthreads();
    }
    if(tid==0) field_inv(total_inv, prefix[I2_LANES_PER_BLOCK-1]);
    __syncthreads();
    U256 left = (tid>0) ? prefix[tid-1] : one;
    U256 right = (tid<I2_LANES_PER_BLOCK-1) ? suffix[tid+1] : one;
    U256 lr; field_mul(lr, left, right);
    U256 zinv; field_mul(zinv, total_inv, lr);
    U256 result_x, result_y;
    field_mul(result_x, shifted.X, zinv);
    field_mul(result_y, shifted.Y, zinv);

    uint64_t fp3 = fingerprint_gpu_h(result_x);
    if(bloom_check_real(bloom3_bits, bloom3_bitcount, bloom3_k, bloom3_block_bits, fp3)){
        int32_t pos = baby_find_gpu(baby_table, baby_array_size, baby_K, fp3);
        if(pos>=0){
            out_found_i3[point_idx] = i3;
            out_found_idx[point_idx] = (int)baby_table[pos].idx;
        }
    }
}

extern "C" {

GpuBabyTable* gpu_baby_table_upload(const GpuBabyEntry *host_table, uint64_t array_size, int K){
    GpuBabyTable *t = new GpuBabyTable();
    t->array_size = array_size; t->K = K;
    cudaError_t err = cudaMalloc(&t->d_table, array_size*sizeof(GpuBabyEntry));
    if(err != cudaSuccess){
        fprintf(stderr, "[gpu_hierarchy] gpu_baby_table_upload: cudaMalloc failed for %.2f GB (%s)\n",
                array_size*sizeof(GpuBabyEntry)/1e9, cudaGetErrorString(err));
        delete t;
        return nullptr;
    }
    err = cudaMemcpy(t->d_table, host_table, array_size*sizeof(GpuBabyEntry), cudaMemcpyHostToDevice);
    if(err != cudaSuccess){
        fprintf(stderr, "[gpu_hierarchy] gpu_baby_table_upload: cudaMemcpy failed (%s)\n", cudaGetErrorString(err));
        cudaFree(t->d_table); delete t;
        return nullptr;
    }
    return t;
}

void gpu_baby_table_free(GpuBabyTable *t){
    if(!t) return;
    if(t->d_table) cudaFree(t->d_table);
    delete t;
}

int gpu_i3loop_and_exact_check_single(GpuShiftTable *shift_table, GpuBloom *bloom3,
                                       GpuBabyTable *baby_table, const GpuECPoint *R,
                                       int *out_found_i3){
    if(!shift_table || !bloom3 || !baby_table) return -2;
    ECPoint h_R;
    for(int j=0;j<4;++j){ h_R.x.v[j]=R->x[j]; h_R.y.v[j]=R->y[j]; }
    h_R.infinity=false;

    ECPoint *d_R;
    cudaMalloc(&d_R, I2_POINTS_PER_BLOCK*sizeof(ECPoint));
    ECPoint h_R_padded[I2_POINTS_PER_BLOCK];
    for(int i=0;i<I2_POINTS_PER_BLOCK;++i) h_R_padded[i] = h_R;
    cudaMemcpy(d_R, h_R_padded, I2_POINTS_PER_BLOCK*sizeof(ECPoint), cudaMemcpyHostToDevice);

    int *d_found_i3, *d_found_idx;
    cudaMalloc(&d_found_i3, I2_POINTS_PER_BLOCK*sizeof(int));
    cudaMalloc(&d_found_idx, I2_POINTS_PER_BLOCK*sizeof(int));
    int neg1[I2_POINTS_PER_BLOCK]; for(int i=0;i<I2_POINTS_PER_BLOCK;++i) neg1[i]=-1;
    cudaMemcpy(d_found_i3, neg1, I2_POINTS_PER_BLOCK*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_found_idx, neg1, I2_POINTS_PER_BLOCK*sizeof(int), cudaMemcpyHostToDevice);

    i3loop_and_exact_kernel<<<1, I2_LANES_PER_BLOCK>>>(shift_table->d_points, d_R,
                                                         bloom3->d_bits, bloom3->bit_count, bloom3->k_hashes, bloom3->block_bits,
                                                         baby_table->d_table, baby_table->array_size, baby_table->K,
                                                         d_found_i3, d_found_idx);
    cudaError_t err = cudaDeviceSynchronize();
    if(err != cudaSuccess){
        fprintf(stderr, "[gpu_hierarchy] gpu_i3loop_and_exact_check_single: kernel failed (%s)\n", cudaGetErrorString(err));
        cudaFree(d_R); cudaFree(d_found_i3); cudaFree(d_found_idx);
        return -2;
    }
    int h_found_i3, h_found_idx;
    cudaMemcpy(&h_found_i3, d_found_i3, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_found_idx, d_found_idx, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_R); cudaFree(d_found_i3); cudaFree(d_found_idx);

    if(out_found_i3) *out_found_i3 = h_found_i3;
    return h_found_idx;
}

}

__global__ void bloom1_compact_kernel_real(const uint64_t *bloom1_bits, uint64_t bloom1_bitcount,
                                            uint32_t bloom1_k, uint64_t bloom1_block_bits,
                                            const ECPoint *candidates, int n_candidates,
                                            ECPoint *out_survivors, int *out_survivor_orig_idx,
                                            unsigned long long *out_survivor_count){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx>=n_candidates) return;
    uint64_t fp = fingerprint_gpu_h(candidates[idx].x);
    if(bloom_check_real(bloom1_bits, bloom1_bitcount, bloom1_k, bloom1_block_bits, fp)){
        unsigned long long pos = atomicAdd(out_survivor_count, 1ULL);
        out_survivors[pos] = candidates[idx];
        out_survivor_orig_idx[pos] = idx;
    }
}

__global__ void i2loop_kernel_batch(ECPoint *shift2, ECPoint *R_points,
                                     uint64_t *bloom2_bits, uint64_t bloom2_bitcount, uint32_t bloom2_k,
                                     uint64_t bloom2_block_bits,
                                     int *out_found_i2, ECPoint *out_found_point){
    __shared__ U256 prefix[I2_LANES_PER_BLOCK];
    __shared__ U256 suffix[I2_LANES_PER_BLOCK];
    __shared__ U256 total_inv;
    int tid = threadIdx.x;
    int local_point_idx = tid / I2_N_JUMPS;
    int point_idx = blockIdx.x * I2_POINTS_PER_BLOCK + local_point_idx;
    int i2 = tid % I2_N_JUMPS;

    ECPoint R = R_points[point_idx];
    ProjPoint shifted;
    U256 one; one.v[0]=1; one.v[1]=0; one.v[2]=0; one.v[3]=0;

    if(i2==0){
        shifted.X = R.x; shifted.Y = R.y; shifted.Z = one;
    } else {
        ProjPoint Rproj; Rproj.X=R.x; Rproj.Y=R.y; Rproj.Z = one;
        mixed_add_projective(shifted, Rproj, shift2[i2]);
    }

    prefix[tid] = shifted.Z; suffix[tid] = shifted.Z;
    __syncthreads();
    for(int d=1; d<I2_LANES_PER_BLOCK; d*=2){
        U256 val; bool active=(tid>=d);
        if(active) field_mul(val, prefix[tid], prefix[tid-d]);
        __syncthreads();
        if(active) prefix[tid]=val;
        __syncthreads();
    }
    for(int d=1; d<I2_LANES_PER_BLOCK; d*=2){
        U256 val; bool active=(tid+d<I2_LANES_PER_BLOCK);
        if(active) field_mul(val, suffix[tid], suffix[tid+d]);
        __syncthreads();
        if(active) suffix[tid]=val;
        __syncthreads();
    }
    if(tid==0) field_inv(total_inv, prefix[I2_LANES_PER_BLOCK-1]);
    __syncthreads();
    U256 left = (tid>0) ? prefix[tid-1] : one;
    U256 right = (tid<I2_LANES_PER_BLOCK-1) ? suffix[tid+1] : one;
    U256 lr; field_mul(lr, left, right);
    U256 zinv; field_mul(zinv, total_inv, lr);
    U256 result_x, result_y;
    field_mul(result_x, shifted.X, zinv);
    field_mul(result_y, shifted.Y, zinv);

    uint64_t fp2 = fingerprint_gpu_h(result_x);
    if(bloom_check_real(bloom2_bits, bloom2_bitcount, bloom2_k, bloom2_block_bits, fp2)){
        out_found_i2[point_idx] = i2;
        out_found_point[point_idx].x = result_x;
        out_found_point[point_idx].y = result_y;
        out_found_point[point_idx].infinity = false;
    }
}

__global__ void pad_points_kernel(ECPoint *arr, int real_count, int padded_count){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx < real_count || idx >= padded_count) return;
    arr[idx].x.v[0]=0; arr[idx].x.v[1]=0; arr[idx].x.v[2]=0; arr[idx].x.v[3]=0;
    arr[idx].y.v[0]=0; arr[idx].y.v[1]=0; arr[idx].y.v[2]=0; arr[idx].y.v[3]=0;
    arr[idx].infinity=false;
}

extern "C" {

void gpu_batch_hierarchical_search(GpuBloom *bloom1, GpuShiftTable *shift2, GpuBloom *bloom2,
                                    GpuShiftTable *shift3, GpuBloom *bloom3, GpuBabyTable *baby_table,
                                    const GpuECPoint *R_points, int n_points,
                                    int *out_found_i2, int *out_found_i3, int *out_found_idx){
    if(n_points<=0) return;

    ECPoint *h_candidates = new ECPoint[n_points];
    for(int i=0;i<n_points;++i){
        for(int j=0;j<4;++j){ h_candidates[i].x.v[j]=R_points[i].x[j]; h_candidates[i].y.v[j]=R_points[i].y[j]; }
        h_candidates[i].infinity=false;
    }
    ECPoint *d_candidates;
    cudaMalloc(&d_candidates, n_points*sizeof(ECPoint));
    cudaMemcpy(d_candidates, h_candidates, n_points*sizeof(ECPoint), cudaMemcpyHostToDevice);
    delete[] h_candidates;

    ECPoint *d_surv1; int *d_surv1_idx; unsigned long long *d_surv1_count;
    cudaMalloc(&d_surv1, n_points*sizeof(ECPoint));
    cudaMalloc(&d_surv1_idx, n_points*sizeof(int));
    cudaMalloc(&d_surv1_count, sizeof(unsigned long long));
    cudaMemset(d_surv1_count, 0, sizeof(unsigned long long));
    int tpb=256; int blocks1=(n_points+tpb-1)/tpb;
    bloom1_compact_kernel_real<<<blocks1,tpb>>>(bloom1->d_bits, bloom1->bit_count, bloom1->k_hashes, bloom1->block_bits,
                                                 d_candidates, n_points, d_surv1, d_surv1_idx, d_surv1_count);
    cudaDeviceSynchronize();
    unsigned long long h_surv1_count;
    cudaMemcpy(&h_surv1_count, d_surv1_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaFree(d_candidates); cudaFree(d_surv1_count);
    if(h_surv1_count==0){ cudaFree(d_surv1); cudaFree(d_surv1_idx); return; }

    int *h_surv1_idx = new int[h_surv1_count];
    cudaMemcpy(h_surv1_idx, d_surv1_idx, h_surv1_count*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_surv1_idx);

    int padded1 = (int)((h_surv1_count + I2_POINTS_PER_BLOCK - 1)/I2_POINTS_PER_BLOCK)*I2_POINTS_PER_BLOCK;
    if(padded1==0) padded1 = I2_POINTS_PER_BLOCK;
    pad_points_kernel<<<(padded1+255)/256,256>>>(d_surv1, (int)h_surv1_count, padded1);
    cudaDeviceSynchronize();

    int *d_i2_found; ECPoint *d_i2_pt;
    cudaMalloc(&d_i2_found, padded1*sizeof(int));
    cudaMalloc(&d_i2_pt, padded1*sizeof(ECPoint));
    std::vector<int> neg1fill(padded1, -1);
    cudaMemcpy(d_i2_found, neg1fill.data(), padded1*sizeof(int), cudaMemcpyHostToDevice);

    int blocks2 = padded1/I2_POINTS_PER_BLOCK;
    i2loop_kernel_batch<<<blocks2, I2_LANES_PER_BLOCK>>>(shift2->d_points, d_surv1,
                                                          bloom2->d_bits, bloom2->bit_count, bloom2->k_hashes, bloom2->block_bits,
                                                          d_i2_found, d_i2_pt);
    cudaDeviceSynchronize();
    cudaFree(d_surv1);

    int *h_i2_found = new int[padded1];
    ECPoint *h_i2_pt = new ECPoint[padded1];
    cudaMemcpy(h_i2_found, d_i2_found, padded1*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_i2_pt, d_i2_pt, padded1*sizeof(ECPoint), cudaMemcpyDeviceToHost);
    cudaFree(d_i2_found); cudaFree(d_i2_pt);

    std::vector<ECPoint> i2_survivors;
    std::vector<int> i2_orig_idx, i2_found_i2;
    for(int i=0;i<padded1;++i){
        if(h_i2_found[i]>=0 && i<(int)h_surv1_count){
            i2_survivors.push_back(h_i2_pt[i]);
            i2_orig_idx.push_back(h_surv1_idx[i]);
            i2_found_i2.push_back(h_i2_found[i]);
        }
    }
    delete[] h_i2_found; delete[] h_i2_pt; delete[] h_surv1_idx;
    if(i2_survivors.empty()) return;

    int padded2 = (int)((i2_survivors.size()+I2_POINTS_PER_BLOCK-1)/I2_POINTS_PER_BLOCK)*I2_POINTS_PER_BLOCK;
    if(padded2==0) padded2 = I2_POINTS_PER_BLOCK;
    ECPoint *d_surv2; cudaMalloc(&d_surv2, padded2*sizeof(ECPoint));
    cudaMemcpy(d_surv2, i2_survivors.data(), i2_survivors.size()*sizeof(ECPoint), cudaMemcpyHostToDevice);
    pad_points_kernel<<<(padded2+255)/256,256>>>(d_surv2, (int)i2_survivors.size(), padded2);
    cudaDeviceSynchronize();

    int *d_i3_found, *d_idx_found;
    cudaMalloc(&d_i3_found, padded2*sizeof(int));
    cudaMalloc(&d_idx_found, padded2*sizeof(int));
    std::vector<int> neg1fill2(padded2, -1);
    cudaMemcpy(d_i3_found, neg1fill2.data(), padded2*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_idx_found, neg1fill2.data(), padded2*sizeof(int), cudaMemcpyHostToDevice);

    int blocks3 = padded2/I2_POINTS_PER_BLOCK;
    i3loop_and_exact_kernel<<<blocks3, I2_LANES_PER_BLOCK>>>(shift3->d_points, d_surv2,
                                                              bloom3->d_bits, bloom3->bit_count, bloom3->k_hashes, bloom3->block_bits,
                                                              baby_table->d_table, baby_table->array_size, baby_table->K,
                                                              d_i3_found, d_idx_found);
    cudaDeviceSynchronize();
    cudaFree(d_surv2);

    int *h_i3_found = new int[padded2];
    int *h_idx_found = new int[padded2];
    cudaMemcpy(h_i3_found, d_i3_found, padded2*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_idx_found, d_idx_found, padded2*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_i3_found); cudaFree(d_idx_found);

    for(int i=0;i<(int)i2_survivors.size();++i){
        if(h_i3_found[i]>=0 && h_idx_found[i]>=0){
            int orig = i2_orig_idx[i];
            out_found_i2[orig] = i2_found_i2[i];
            out_found_i3[orig] = h_i3_found[i];
            out_found_idx[orig] = h_idx_found[i];
        }
    }
    delete[] h_i3_found; delete[] h_idx_found;
}

}

#define ADVANCE_BLOCK 256

__global__ void advance_batch_kernel(ECPoint *points, int n_points, ECPoint shift){
    __shared__ U256 prefix[ADVANCE_BLOCK];
    __shared__ U256 suffix[ADVANCE_BLOCK];
    __shared__ U256 total_inv;
    int tid = threadIdx.x;
    int idx = blockIdx.x*ADVANCE_BLOCK + tid;
    U256 one; one.v[0]=1; one.v[1]=0; one.v[2]=0; one.v[3]=0;

    bool valid = (idx < n_points);
    ProjPoint shifted;
    if(valid){
        ProjPoint Pproj; Pproj.X=points[idx].x; Pproj.Y=points[idx].y; Pproj.Z=one;
        mixed_add_projective(shifted, Pproj, shift);
    } else {

        shifted.X=one; shifted.Y=one; shifted.Z=one;
    }

    prefix[tid] = shifted.Z; suffix[tid] = shifted.Z;
    __syncthreads();
    for(int d=1; d<ADVANCE_BLOCK; d*=2){
        U256 val; bool active=(tid>=d);
        if(active) field_mul(val, prefix[tid], prefix[tid-d]);
        __syncthreads();
        if(active) prefix[tid]=val;
        __syncthreads();
    }
    for(int d=1; d<ADVANCE_BLOCK; d*=2){
        U256 val; bool active=(tid+d<ADVANCE_BLOCK);
        if(active) field_mul(val, suffix[tid], suffix[tid+d]);
        __syncthreads();
        if(active) suffix[tid]=val;
        __syncthreads();
    }
    if(tid==0) field_inv(total_inv, prefix[ADVANCE_BLOCK-1]);
    __syncthreads();
    if(!valid) return;
    U256 left = (tid>0) ? prefix[tid-1] : one;
    U256 right = (tid<ADVANCE_BLOCK-1) ? suffix[tid+1] : one;
    U256 lr; field_mul(lr, left, right);
    U256 zinv; field_mul(zinv, total_inv, lr);
    U256 result_x, result_y;
    field_mul(result_x, shifted.X, zinv);
    field_mul(result_y, shifted.Y, zinv);
    points[idx].x = result_x;
    points[idx].y = result_y;
}

extern "C" {

void gpu_batch_advance(GpuECPoint *points, int n_points, const GpuECPoint *shift){
    if(n_points<=0) return;
    ECPoint *h_pts = new ECPoint[n_points];
    for(int i=0;i<n_points;++i){
        for(int j=0;j<4;++j){ h_pts[i].x.v[j]=points[i].x[j]; h_pts[i].y.v[j]=points[i].y[j]; }
        h_pts[i].infinity=false;
    }
    ECPoint *d_pts;
    cudaMalloc(&d_pts, n_points*sizeof(ECPoint));
    cudaMemcpy(d_pts, h_pts, n_points*sizeof(ECPoint), cudaMemcpyHostToDevice);
    delete[] h_pts;

    ECPoint h_shift;
    for(int j=0;j<4;++j){ h_shift.x.v[j]=shift->x[j]; h_shift.y.v[j]=shift->y[j]; }
    h_shift.infinity=false;

    int blocks = (n_points + ADVANCE_BLOCK - 1)/ADVANCE_BLOCK;
    advance_batch_kernel<<<blocks, ADVANCE_BLOCK>>>(d_pts, n_points, h_shift);
    cudaError_t err = cudaDeviceSynchronize();
    if(err != cudaSuccess){
        fprintf(stderr, "[gpu_hierarchy] gpu_batch_advance: kernel failed (%s)\n", cudaGetErrorString(err));
        cudaFree(d_pts);
        return;
    }

    ECPoint *h_result = new ECPoint[n_points];
    cudaMemcpy(h_result, d_pts, n_points*sizeof(ECPoint), cudaMemcpyDeviceToHost);
    for(int i=0;i<n_points;++i){
        for(int j=0;j<4;++j){ points[i].x[j]=h_result[i].x.v[j]; points[i].y[j]=h_result[i].y.v[j]; }
    }
    delete[] h_result;
    cudaFree(d_pts);
}

}

struct GpuHierarchy {
    GpuBloom *bloom1, *bloom2, *bloom3;
    GpuShiftTable *shift2, *shift3;
    GpuBabyTable *baby_table;
    GpuECPoint giant_step;
    GpuECPoint neg_giant_step;
};

extern "C" {

GpuHierarchy* gpu_hierarchy_build(
    const uint64_t *bloom1_bits, uint64_t bloom1_bitcount, uint64_t bloom1_nwords, uint32_t bloom1_k, uint64_t bloom1_block_bits,
    const uint64_t *bloom2_bits, uint64_t bloom2_bitcount, uint64_t bloom2_nwords, uint32_t bloom2_k, uint64_t bloom2_block_bits,
    const uint64_t *bloom3_bits, uint64_t bloom3_bitcount, uint64_t bloom3_nwords, uint32_t bloom3_k, uint64_t bloom3_block_bits,
    const GpuECPoint *shift2_32, const GpuECPoint *shift3_32,
    const void *baby_table_raw, uint64_t baby_array_size, int baby_K,
    const GpuECPoint *giant_step_point){

    GpuHierarchy *h = new GpuHierarchy();
    h->bloom1 = gpu_bloom_upload(bloom1_bits, bloom1_bitcount, bloom1_nwords, bloom1_k, bloom1_block_bits);
    h->bloom2 = gpu_bloom_upload(bloom2_bits, bloom2_bitcount, bloom2_nwords, bloom2_k, bloom2_block_bits);
    h->bloom3 = gpu_bloom_upload(bloom3_bits, bloom3_bitcount, bloom3_nwords, bloom3_k, bloom3_block_bits);
    h->shift2 = gpu_shift_table_upload(shift2_32, 32);
    h->shift3 = gpu_shift_table_upload(shift3_32, 32);
    h->baby_table = gpu_baby_table_upload((const GpuBabyEntry*)baby_table_raw, baby_array_size, baby_K);

    if(!h->bloom1 || !h->bloom2 || !h->bloom3 || !h->shift2 || !h->shift3 || !h->baby_table){
        fprintf(stderr, "[gpu_hierarchy] gpu_hierarchy_build: one or more resource uploads failed\n");
        gpu_hierarchy_free(h);
        return nullptr;
    }

    h->giant_step = *giant_step_point;

    extern __global__ void negate_point_kernel(ECPoint*, ECPoint*);
    ECPoint h_gs, h_ngs;
    for(int j=0;j<4;++j){ h_gs.x.v[j]=giant_step_point->x[j]; h_gs.y.v[j]=giant_step_point->y[j]; }
    h_gs.infinity=false;
    ECPoint *d_gs, *d_ngs;
    cudaMalloc(&d_gs, sizeof(ECPoint));
    cudaMalloc(&d_ngs, sizeof(ECPoint));
    cudaMemcpy(d_gs, &h_gs, sizeof(ECPoint), cudaMemcpyHostToDevice);
    negate_point_kernel<<<1,1>>>(d_gs, d_ngs);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_ngs, d_ngs, sizeof(ECPoint), cudaMemcpyDeviceToHost);
    cudaFree(d_gs); cudaFree(d_ngs);
    for(int j=0;j<4;++j){ h->neg_giant_step.x[j]=h_ngs.x.v[j]; h->neg_giant_step.y[j]=h_ngs.y.v[j]; }

    return h;
}

void gpu_hierarchy_free(GpuHierarchy *h){
    if(!h) return;
    gpu_bloom_free(h->bloom1); gpu_bloom_free(h->bloom2); gpu_bloom_free(h->bloom3);
    gpu_shift_table_free(h->shift2); gpu_shift_table_free(h->shift3);
    gpu_baby_table_free(h->baby_table);
    delete h;
}

}

#define ROUNDS_PER_LAUNCH 32
#define MAX_SURVIVORS_PER_LAUNCH 65536

#define BLOOM1_SCAN_BLOCK 256

__global__ void bloom1_multiround_kernel(ECPoint *points, int n_points, ECPoint shift, int K,
                                          const uint64_t *bloom1_bits, uint64_t bloom1_bitcount,
                                          uint32_t bloom1_k, uint64_t bloom1_block_bits,
                                          ECPoint *out_pt, int *out_leaf, int *out_round_offset,
                                          unsigned long long *out_count, int max_out){
    __shared__ U256 prefix[BLOOM1_SCAN_BLOCK];
    __shared__ U256 suffix[BLOOM1_SCAN_BLOCK];
    __shared__ U256 total_inv;

    int tid = threadIdx.x;
    int idx = blockIdx.x*BLOOM1_SCAN_BLOCK + tid;
    bool valid = (idx < n_points);
    U256 one; one.v[0]=1; one.v[1]=0; one.v[2]=0; one.v[3]=0;

    ECPoint cur;
    if(valid) cur = points[idx];

    for(int r=0;r<K;++r){

        if(valid){
            uint64_t fp = fingerprint_gpu_h(cur.x);
            if(bloom_check_real(bloom1_bits, bloom1_bitcount, bloom1_k, bloom1_block_bits, fp)){
                unsigned long long pos = atomicAdd(out_count, 1ULL);
                if(pos < (unsigned long long)max_out){
                    out_pt[pos] = cur;
                    out_leaf[pos] = idx;
                    out_round_offset[pos] = r;
                }
            }
        }

        ProjPoint shifted;
        if(valid){
            ProjPoint Pproj; Pproj.X=cur.x; Pproj.Y=cur.y; Pproj.Z=one;
            mixed_add_projective(shifted, Pproj, shift);
        } else {
            shifted.X=one; shifted.Y=one; shifted.Z=one;
        }

        prefix[tid] = shifted.Z; suffix[tid] = shifted.Z;
        __syncthreads();
        for(int d=1; d<BLOOM1_SCAN_BLOCK; d*=2){
            U256 val; bool active=(tid>=d);
            if(active) field_mul(val, prefix[tid], prefix[tid-d]);
            __syncthreads();
            if(active) prefix[tid]=val;
            __syncthreads();
        }
        for(int d=1; d<BLOOM1_SCAN_BLOCK; d*=2){
            U256 val; bool active=(tid+d<BLOOM1_SCAN_BLOCK);
            if(active) field_mul(val, suffix[tid], suffix[tid+d]);
            __syncthreads();
            if(active) suffix[tid]=val;
            __syncthreads();
        }
        if(tid==0) field_inv(total_inv, prefix[BLOOM1_SCAN_BLOCK-1]);
        __syncthreads();
        U256 left = (tid>0) ? prefix[tid-1] : one;
        U256 right = (tid<BLOOM1_SCAN_BLOCK-1) ? suffix[tid+1] : one;
        U256 lr; field_mul(lr, left, right);
        U256 zinv; field_mul(zinv, total_inv, lr);

        if(valid){
            field_mul(cur.x, shifted.X, zinv);
            field_mul(cur.y, shifted.Y, zinv);
        }
        __syncthreads();
    }
    if(valid) points[idx] = cur;
}

extern "C" {

static int resolve_survivors(GpuHierarchy *h, ECPoint *d_surv_pts, int n_surv,
                              int *out_i2, int *out_i3, int *out_idx){
    if(n_surv<=0) return -1;
    int padded1 = (int)((n_surv+I2_POINTS_PER_BLOCK-1)/I2_POINTS_PER_BLOCK)*I2_POINTS_PER_BLOCK;
    ECPoint *d_p1; cudaMalloc(&d_p1, padded1*sizeof(ECPoint));
    cudaMemcpy(d_p1, d_surv_pts, n_surv*sizeof(ECPoint), cudaMemcpyDeviceToDevice);
    pad_points_kernel<<<(padded1+255)/256,256>>>(d_p1, n_surv, padded1);

    int *d_i2; ECPoint *d_i2pt;
    cudaMalloc(&d_i2, padded1*sizeof(int));
    cudaMalloc(&d_i2pt, padded1*sizeof(ECPoint));
    cudaMemset(d_i2, 0xFF, padded1*sizeof(int));
    int blocks2 = padded1/I2_POINTS_PER_BLOCK;
    i2loop_kernel_batch<<<blocks2, I2_LANES_PER_BLOCK>>>(h->shift2->d_points, d_p1,
                                                          h->bloom2->d_bits, h->bloom2->bit_count, h->bloom2->k_hashes, h->bloom2->block_bits,
                                                          d_i2, d_i2pt);
    std::vector<int> h_i2(padded1);
    cudaMemcpy(h_i2.data(), d_i2, padded1*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_p1); cudaFree(d_i2);

    std::vector<int> surv_idx, surv_i2;
    for(int i=0;i<n_surv;++i) if(h_i2[i]>=0){ surv_idx.push_back(i); surv_i2.push_back(h_i2[i]); }
    if(surv_idx.empty()){ cudaFree(d_i2pt); return -1; }

    int n2 = (int)surv_idx.size();
    int padded2 = (int)((n2+I2_POINTS_PER_BLOCK-1)/I2_POINTS_PER_BLOCK)*I2_POINTS_PER_BLOCK;
    ECPoint *d_p2; cudaMalloc(&d_p2, padded2*sizeof(ECPoint));
    for(int i=0;i<n2;++i) cudaMemcpy(d_p2+i, d_i2pt+surv_idx[i], sizeof(ECPoint), cudaMemcpyDeviceToDevice);
    pad_points_kernel<<<(padded2+255)/256,256>>>(d_p2, n2, padded2);
    cudaFree(d_i2pt);

    int *d_i3, *d_idx;
    cudaMalloc(&d_i3, padded2*sizeof(int));
    cudaMalloc(&d_idx, padded2*sizeof(int));
    cudaMemset(d_i3, 0xFF, padded2*sizeof(int));
    cudaMemset(d_idx, 0xFF, padded2*sizeof(int));
    int blocks3 = padded2/I2_POINTS_PER_BLOCK;
    i3loop_and_exact_kernel<<<blocks3, I2_LANES_PER_BLOCK>>>(h->shift3->d_points, d_p2,
                                                              h->bloom3->d_bits, h->bloom3->bit_count, h->bloom3->k_hashes, h->bloom3->block_bits,
                                                              h->baby_table->d_table, h->baby_table->array_size, h->baby_table->K,
                                                              d_i3, d_idx);
    std::vector<int> h_i3(padded2), h_idx(padded2);
    cudaMemcpy(h_i3.data(), d_i3, padded2*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_idx.data(), d_idx, padded2*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_p2); cudaFree(d_i3); cudaFree(d_idx);

    for(int i=0;i<n2;++i){
        if(h_i3[i]>=0 && h_idx[i]>=0){
            *out_i2 = surv_i2[i]; *out_i3 = h_i3[i]; *out_idx = h_idx[i];
            return surv_idx[i];
        }
    }
    return -1;
}

int gpu_hierarchy_search_block(GpuHierarchy *h, const GpuECPoint *lower_start, const GpuECPoint *upper_start,
                                int n_leaves, uint64_t half_giant,
                                int *out_is_upper_front, uint64_t *out_round,
                                int *out_i2, int *out_i3, int *out_idx){
    if(!h || n_leaves<=0) return -1;

    ECPoint *h_lower = new ECPoint[n_leaves];
    ECPoint *h_upper = new ECPoint[n_leaves];
    for(int i=0;i<n_leaves;++i){
        for(int j=0;j<4;++j){
            h_lower[i].x.v[j]=lower_start[i].x[j]; h_lower[i].y.v[j]=lower_start[i].y[j];
            h_upper[i].x.v[j]=upper_start[i].x[j]; h_upper[i].y.v[j]=upper_start[i].y[j];
        }
        h_lower[i].infinity=false; h_upper[i].infinity=false;
    }
    ECPoint *d_lower, *d_upper;
    cudaMalloc(&d_lower, n_leaves*sizeof(ECPoint));
    cudaMalloc(&d_upper, n_leaves*sizeof(ECPoint));
    cudaMemcpy(d_lower, h_lower, n_leaves*sizeof(ECPoint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_upper, h_upper, n_leaves*sizeof(ECPoint), cudaMemcpyHostToDevice);
    delete[] h_lower; delete[] h_upper;

    ECPoint h_gs, h_ngs;
    for(int j=0;j<4;++j){ h_gs.x.v[j]=h->giant_step.x[j]; h_gs.y.v[j]=h->giant_step.y[j];
                           h_ngs.x.v[j]=h->neg_giant_step.x[j]; h_ngs.y.v[j]=h->neg_giant_step.y[j]; }
    h_gs.infinity=false; h_ngs.infinity=false;

    ECPoint *d_surv_pt; int *d_surv_leaf, *d_surv_round; unsigned long long *d_surv_count;
    cudaMalloc(&d_surv_pt, MAX_SURVIVORS_PER_LAUNCH*sizeof(ECPoint));
    cudaMalloc(&d_surv_leaf, MAX_SURVIVORS_PER_LAUNCH*sizeof(int));
    cudaMalloc(&d_surv_round, MAX_SURVIVORS_PER_LAUNCH*sizeof(int));
    cudaMalloc(&d_surv_count, sizeof(unsigned long long));

    bool found=false;
    int leaf_idx=-1, fi2=-1, fi3=-1, fidx=-1;
    int found_is_upper=0; uint64_t found_abs_round=0;
    int tpb=256; int blocks=(n_leaves+tpb-1)/tpb;

    for(uint64_t chunk_start=0; chunk_start<half_giant && !found; chunk_start+=ROUNDS_PER_LAUNCH){
        int K = (int)((half_giant-chunk_start) < ROUNDS_PER_LAUNCH ? (half_giant-chunk_start) : ROUNDS_PER_LAUNCH);

        for(int front=0; front<2 && !found; ++front){
            ECPoint *d_pts = (front==0) ? d_lower : d_upper;
            ECPoint shift = (front==0) ? h_ngs : h_gs;

            cudaMemset(d_surv_count, 0, sizeof(unsigned long long));
            bloom1_multiround_kernel<<<blocks,tpb>>>(d_pts, n_leaves, shift, K,
                                                      h->bloom1->d_bits, h->bloom1->bit_count, h->bloom1->k_hashes, h->bloom1->block_bits,
                                                      d_surv_pt, d_surv_leaf, d_surv_round, d_surv_count, MAX_SURVIVORS_PER_LAUNCH);
            unsigned long long h_count;
            cudaMemcpy(&h_count, d_surv_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
            if(h_count==0) continue;
            int n_surv = (int)(h_count > MAX_SURVIVORS_PER_LAUNCH ? MAX_SURVIVORS_PER_LAUNCH : h_count);

            int rslv_i2=-1, rslv_i3=-1, rslv_idx=-1;
            int surv_pos = resolve_survivors(h, d_surv_pt, n_surv, &rslv_i2, &rslv_i3, &rslv_idx);
            if(surv_pos>=0){
                std::vector<int> h_leaf(n_surv), h_round(n_surv);
                cudaMemcpy(h_leaf.data(), d_surv_leaf, n_surv*sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_round.data(), d_surv_round, n_surv*sizeof(int), cudaMemcpyDeviceToHost);
                leaf_idx = h_leaf[surv_pos];
                found_abs_round = chunk_start + (uint64_t)h_round[surv_pos];
                found_is_upper = front;
                fi2=rslv_i2; fi3=rslv_i3; fidx=rslv_idx;
                found = true;
            }
        }
    }

    cudaFree(d_surv_pt); cudaFree(d_surv_leaf); cudaFree(d_surv_round); cudaFree(d_surv_count);
    cudaFree(d_lower); cudaFree(d_upper);

    if(found){
        *out_is_upper_front = found_is_upper; *out_round = found_abs_round;
        *out_i2 = fi2; *out_i3 = fi3; *out_idx = fidx;
        return leaf_idx;
    }
    return -1;
}

int gpu_range_scan_chunk(GpuHierarchy *h, GpuECPoint *lane_points, int n_lanes,
                          const GpuECPoint *stride_shift, int K,
                          int *out_round, int *out_i2, int *out_i3, int *out_idx){
    if(!h || n_lanes<=0) return -1;

    ECPoint *h_pts = new ECPoint[n_lanes];
    for(int i=0;i<n_lanes;++i){
        for(int j=0;j<4;++j){ h_pts[i].x.v[j]=lane_points[i].x[j]; h_pts[i].y.v[j]=lane_points[i].y[j]; }
        h_pts[i].infinity=false;
    }
    ECPoint *d_pts;
    cudaMalloc(&d_pts, n_lanes*sizeof(ECPoint));
    cudaMemcpy(d_pts, h_pts, n_lanes*sizeof(ECPoint), cudaMemcpyHostToDevice);
    delete[] h_pts;

    ECPoint h_shift;
    for(int j=0;j<4;++j){ h_shift.x.v[j]=stride_shift->x[j]; h_shift.y.v[j]=stride_shift->y[j]; }
    h_shift.infinity=false;

    ECPoint *d_surv_pt; int *d_surv_leaf, *d_surv_round; unsigned long long *d_surv_count;
    cudaMalloc(&d_surv_pt, MAX_SURVIVORS_PER_LAUNCH*sizeof(ECPoint));
    cudaMalloc(&d_surv_leaf, MAX_SURVIVORS_PER_LAUNCH*sizeof(int));
    cudaMalloc(&d_surv_round, MAX_SURVIVORS_PER_LAUNCH*sizeof(int));
    cudaMalloc(&d_surv_count, sizeof(unsigned long long));
    cudaMemset(d_surv_count, 0, sizeof(unsigned long long));

    int tpb=256; int blocks=(n_lanes+tpb-1)/tpb;
    bloom1_multiround_kernel<<<blocks,tpb>>>(d_pts, n_lanes, h_shift, K,
                                              h->bloom1->d_bits, h->bloom1->bit_count, h->bloom1->k_hashes, h->bloom1->block_bits,
                                              d_surv_pt, d_surv_leaf, d_surv_round, d_surv_count, MAX_SURVIVORS_PER_LAUNCH);

    ECPoint *h_result = new ECPoint[n_lanes];
    cudaMemcpy(h_result, d_pts, n_lanes*sizeof(ECPoint), cudaMemcpyDeviceToHost);
    for(int i=0;i<n_lanes;++i){
        for(int j=0;j<4;++j){ lane_points[i].x[j]=h_result[i].x.v[j]; lane_points[i].y[j]=h_result[i].y.v[j]; }
    }
    delete[] h_result;
    cudaFree(d_pts);

    unsigned long long h_count;
    cudaMemcpy(&h_count, d_surv_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    int result_lane = -1;
    if(h_count>0){
        int n_surv = (int)(h_count > MAX_SURVIVORS_PER_LAUNCH ? MAX_SURVIVORS_PER_LAUNCH : h_count);
        int rslv_i2=-1, rslv_i3=-1, rslv_idx=-1;
        int surv_pos = resolve_survivors(h, d_surv_pt, n_surv, &rslv_i2, &rslv_i3, &rslv_idx);
        if(surv_pos>=0){
            std::vector<int> h_leaf(n_surv), h_round(n_surv);
            cudaMemcpy(h_leaf.data(), d_surv_leaf, n_surv*sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_round.data(), d_surv_round, n_surv*sizeof(int), cudaMemcpyDeviceToHost);
            result_lane = h_leaf[surv_pos];
            *out_round = h_round[surv_pos];
            *out_i2 = rslv_i2; *out_i3 = rslv_i3; *out_idx = rslv_idx;
        }
    }

    cudaFree(d_surv_pt); cudaFree(d_surv_leaf); cudaFree(d_surv_round); cudaFree(d_surv_count);
    return result_lane;
}

}

__global__ void negate_point_kernel(ECPoint *in, ECPoint *out){
    out->x = in->x;
    U256 zero; zero.v[0]=0; zero.v[1]=0; zero.v[2]=0; zero.v[3]=0;
    field_sub(out->y, zero, in->y);
    out->infinity = in->infinity;
}
