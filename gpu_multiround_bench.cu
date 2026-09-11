
#include <cstdio>
#include <cstdint>
#include <cinttypes>
#include <cstring>
#include <cmath>
#include <cstdlib>

typedef struct { uint64_t v[4]; } U256;
typedef struct { U256 x, y; bool infinity; } ECPoint;
typedef struct { U256 X, Y, Z; } ProjPoint;

__device__ __forceinline__ void field_add(U256 &r, const U256 &a, const U256 &b){
    unsigned long long carry=0;
    #pragma unroll
    for(int i=0;i<4;++i){
        unsigned long long s = a.v[i]+b.v[i];
        unsigned long long c1 = s<a.v[i];
        unsigned long long s2 = s+carry;
        unsigned long long c2 = s2<s;
        r.v[i]=s2; carry=c1+c2;
    }
    const uint64_t P0=0xFFFFFFFEFFFFFC2FULL, P1=0xFFFFFFFFFFFFFFFFULL, P2=0xFFFFFFFFFFFFFFFFULL, P3=0xFFFFFFFFFFFFFFFFULL;
    bool ge = carry!=0;
    if(!ge){
        ge = (r.v[3]>P3)||(r.v[3]==P3&&(r.v[2]>P2||(r.v[2]==P2&&(r.v[1]>P1||(r.v[1]==P1&&r.v[0]>=P0)))));
    }
    if(ge){
        unsigned long long b2=0;
        uint64_t Pv[4]={P0,P1,P2,P3};
        #pragma unroll
        for(int i=0;i<4;++i){
            unsigned long long d = r.v[i]-Pv[i];
            unsigned long long bb = r.v[i]<Pv[i];
            unsigned long long d2 = d-b2;
            unsigned long long bb2 = d<b2;
            r.v[i]=d2; b2=bb+bb2;
        }
    }
}
__device__ void field_sub(U256 &r, const U256 &a, const U256 &b){
    unsigned long long borrow=0;
    U256 t;
    #pragma unroll
    for(int i=0;i<4;++i){
        unsigned long long d=a.v[i]-b.v[i];
        unsigned long long b1=a.v[i]<b.v[i];
        unsigned long long d2=d-borrow;
        unsigned long long b2=d<borrow;
        t.v[i]=d2; borrow=b1+b2;
    }
    if(borrow){
        const uint64_t P0=0xFFFFFFFEFFFFFC2FULL, P1=0xFFFFFFFFFFFFFFFFULL, P2=0xFFFFFFFFFFFFFFFFULL, P3=0xFFFFFFFFFFFFFFFFULL;
        uint64_t Pv[4]={P0,P1,P2,P3};
        unsigned long long carry=0;
        #pragma unroll
        for(int i=0;i<4;++i){
            unsigned long long s=t.v[i]+Pv[i];
            unsigned long long c1=s<t.v[i];
            unsigned long long s2=s+carry;
            unsigned long long c2=s2<s;
            t.v[i]=s2; carry=c1+c2;
        }
    }
    r=t;
}
__device__ void field_mul(U256 &r, const U256 &a, const U256 &b){
    unsigned long long acc[8]={0,0,0,0,0,0,0,0};
    #pragma unroll
    for(int i=0;i<4;++i){
        unsigned long long carry=0;
        #pragma unroll
        for(int j=0;j<4;++j){
            unsigned __int128 p = (unsigned __int128)a.v[i]*b.v[j];
            unsigned long long plo=(unsigned long long)p, phi=(unsigned long long)(p>>64);
            unsigned long long s = acc[i+j]+plo;
            unsigned long long c1 = s<acc[i+j];
            unsigned long long s2 = s+carry;
            unsigned long long c2 = s2<s;
            acc[i+j]=s2;
            unsigned long long newcarry = phi+c1+c2;
            carry = newcarry;
        }
        acc[i+4]+=carry;
    }

    const unsigned long long C=0x1000003D1ULL;
    unsigned long long lo[4]={acc[0],acc[1],acc[2],acc[3]};
    unsigned long long hi[4]={acc[4],acc[5],acc[6],acc[7]};
    unsigned long long t[5]={0,0,0,0,0};
    #pragma unroll
    for(int i=0;i<4;++i){
        unsigned __int128 p=(unsigned __int128)hi[i]*C;
        unsigned long long plo=(unsigned long long)p, phi=(unsigned long long)(p>>64);
        unsigned long long s=t[i]+plo;
        unsigned long long c1=s<t[i];
        t[i]=s;
        unsigned long long s2=t[i+1]+phi+c1;
        t[i+1]=s2;
    }
    U256 R; unsigned long long carry=0;
    #pragma unroll
    for(int i=0;i<4;++i){
        unsigned long long s=lo[i]+t[i];
        unsigned long long c1=s<lo[i];
        unsigned long long s2=s+carry;
        unsigned long long c2=s2<s;
        R.v[i]=s2; carry=c1+c2;
    }
    carry+=t[4];
    while(carry){
        unsigned __int128 p=(unsigned __int128)carry*C;
        unsigned long long plo=(unsigned long long)p, phi=(unsigned long long)(p>>64);
        unsigned long long c2=0;
        unsigned long long s=R.v[0]+plo; unsigned long long c1=s<R.v[0]; R.v[0]=s;
        s=R.v[1]+c1; c1=s<R.v[1]; R.v[1]=s;
        s=R.v[2]+c1; c1=s<R.v[2]; R.v[2]=s;
        s=R.v[3]+c1+phi; c2=(s<R.v[3])||(phi>0&&s-phi<c1?0:0); R.v[3]=s;
        carry=0;
    }
    const uint64_t P0=0xFFFFFFFEFFFFFC2FULL, P1=0xFFFFFFFFFFFFFFFFULL, P2=0xFFFFFFFFFFFFFFFFULL, P3=0xFFFFFFFFFFFFFFFFULL;
    bool ge = (R.v[3]>P3)||(R.v[3]==P3&&(R.v[2]>P2||(R.v[2]==P2&&(R.v[1]>P1||(R.v[1]==P1&&R.v[0]>=P0)))));
    if(ge){
        uint64_t Pv[4]={P0,P1,P2,P3};
        unsigned long long b2=0;
        #pragma unroll
        for(int i=0;i<4;++i){
            unsigned long long d=R.v[i]-Pv[i];
            unsigned long long bb=R.v[i]<Pv[i];
            unsigned long long d2=d-b2;
            unsigned long long bb2=d<b2;
            R.v[i]=d2; b2=bb+bb2;
        }
    }
    r=R;
}
__device__ void field_sqr(U256 &r, const U256 &a){ field_mul(r,a,a); }
__device__ void field_inv(U256 &r, const U256 &a){

    uint64_t exp[4] = {0xFFFFFFFEFFFFFC2DULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL};
    U256 result; result.v[0]=1; result.v[1]=0; result.v[2]=0; result.v[3]=0;
    U256 base = a;
    for(int limb=0; limb<4; ++limb){
        uint64_t e = exp[limb];
        for(int bit=0; bit<64; ++bit){
            if(e & 1ULL){
                U256 tmp; field_mul(tmp, result, base); result=tmp;
            }
            U256 tmp2; field_sqr(tmp2, base); base=tmp2;
            e >>= 1;
        }
    }
    r = result;
}
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
    r.X=X3; r.Y=Y3; r.Z=Z3;
}

typedef struct { uint64_t *bits; uint64_t bit_count, nwords; uint32_t k_hashes; uint64_t block_bits; } BloomFilterGPU;
#define DEFAULT_BLOOM_FP_RATE 1e-3
static void bloom_size(BloomFilterGPU *bf, uint64_t entries, double fp_rate){
    if(entries==0) entries=1;
    if(fp_rate<=0.0||fp_rate>=1.0) fp_rate=DEFAULT_BLOOM_FP_RATE;
    const double ln2=0.6931471805599453094172321214581765680755;
    double num_bits=-((double)entries*log(fp_rate))/(ln2*ln2);
    if(num_bits<64.0) num_bits=64.0;
    uint64_t bc=(uint64_t)num_bits;
    double k=(num_bits/(double)entries)*ln2;
    uint32_t kh=(uint32_t)(k+0.5);
    if(kh<1)kh=1; if(kh>32)kh=32;
    bf->bit_count=bc; bf->nwords=(bc+63)/64; bf->k_hashes=kh; bf->block_bits=0;
}
__device__ __forceinline__ void bloom_positions_gpu(uint64_t bit_count, uint32_t k_hashes, uint64_t fp, uint64_t *out){
    uint64_t h1=fp;
    uint64_t h2=fp+0x9e3779b97f4a7c15ULL;
    h2=(h2^(h2>>30))*0xbf58476d1ce4e5b9ULL;
    h2=(h2^(h2>>27))*0x94d049bb133111ebULL;
    h2^=h2>>31; if(h2==0)h2=1;
    for(uint32_t i=0;i<k_hashes;++i) out[i]=(h1+(uint64_t)i*h2)%bit_count;
}
__device__ int bloom_check_gpu(const uint64_t *bits, uint64_t bit_count, uint32_t k_hashes, uint64_t fp){
    uint64_t positions[32];
    bloom_positions_gpu(bit_count, k_hashes, fp, positions);
    for(uint32_t i=0;i<k_hashes;++i) if(!(bits[positions[i]>>6] & (1ULL<<(positions[i]&63)))) return 0;
    return 1;
}
__device__ void bloom_add_gpu(uint64_t *bits, uint64_t bit_count, uint32_t k_hashes, uint64_t fp){
    uint64_t positions[32];
    bloom_positions_gpu(bit_count, k_hashes, fp, positions);
    for(uint32_t i=0;i<k_hashes;++i) atomicOr((unsigned long long*)&bits[positions[i]>>6], (unsigned long long)(1ULL<<(positions[i]&63)));
}
__device__ uint64_t fingerprint_gpu_h(const U256 &x){
    uint64_t v = x.v[3];
    v += 0x9e3779b97f4a7c15ULL;
    v = (v^(v>>30)) * 0xbf58476d1ce4e5b9ULL;
    v = (v^(v>>27)) * 0x94d049bb133111ebULL;
    v ^= v>>31;
    return v;
}
__host__ __device__ static uint64_t mix64(uint64_t v){
    v+=0x9e3779b97f4a7c15ULL;
    v=(v^(v>>30))*0xbf58476d1ce4e5b9ULL;
    v=(v^(v>>27))*0x94d049bb133111ebULL;
    v^=v>>31;
    return v;
}

#define BLOOM1_SCAN_BLOCK 256

__global__ void bloom1_multiround_kernel(ECPoint *points, int n_points, ECPoint shift, int K,
                                          const uint64_t *bloom1_bits, uint64_t bloom1_bitcount,
                                          uint32_t bloom1_k, uint64_t bloom1_block_bits,
                                          unsigned long long *out_count){
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
            if(bloom_check_gpu(bloom1_bits, bloom1_bitcount, bloom1_k, fp)) atomicAdd(out_count, 1ULL);
        }
        ProjPoint shifted;
        if(valid){
            ProjPoint Pproj; Pproj.X=cur.x; Pproj.Y=cur.y; Pproj.Z=one;
            mixed_add_projective(shifted, Pproj, shift);
        } else { shifted.X=one; shifted.Y=one; shifted.Z=one; }
        prefix[tid]=shifted.Z; suffix[tid]=shifted.Z;
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
        U256 left = (tid>0)?prefix[tid-1]:one;
        U256 right = (tid<BLOOM1_SCAN_BLOCK-1)?suffix[tid+1]:one;
        U256 lr; field_mul(lr,left,right);
        U256 zinv; field_mul(zinv,total_inv,lr);
        if(valid){
            field_mul(cur.x, shifted.X, zinv);
            field_mul(cur.y, shifted.Y, zinv);
        }
        __syncthreads();
    }
    if(valid) points[idx]=cur;
}

__global__ void inversion_only_kernel(int n_points, int K, unsigned long long *dummy_out){
    __shared__ U256 prefix[BLOOM1_SCAN_BLOCK];
    __shared__ U256 suffix[BLOOM1_SCAN_BLOCK];
    __shared__ U256 total_inv;
    int tid = threadIdx.x;
    int idx = blockIdx.x*BLOOM1_SCAN_BLOCK + tid;
    bool valid = (idx < n_points);
    U256 one; one.v[0]=1; one.v[1]=0; one.v[2]=0; one.v[3]=0;
    U256 z; z.v[0]=mix64(idx); z.v[1]=mix64(idx+1); z.v[2]=mix64(idx+2); z.v[3]=mix64(idx+3)%0xFFFFFFFEFFFFFC2FULL;
    for(int r=0;r<K;++r){
        prefix[tid]= valid? z : one; suffix[tid]=prefix[tid];
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
        if(tid==0 && r==K-1) atomicAdd(dummy_out, total_inv.v[0]&1);
    }
}

int main(int argc, char **argv){
    int N = 2097152, K = 32;
    for(int i=1;i<argc;++i){
        if(!strcmp(argv[i],"--n")&&i+1<argc) N=atoi(argv[++i]);
        if(!strcmp(argv[i],"--k")&&i+1<argc) K=atoi(argv[++i]);
    }
    printf("[bench] N=%d lanes, K=%d rounds/launch\n", N, K);

    {
        unsigned long long *d_dummy; cudaMalloc(&d_dummy, sizeof(unsigned long long));
        cudaMemset(d_dummy, 0, sizeof(unsigned long long));
        int blocks=(N+255)/256;
        inversion_only_kernel<<<blocks,256>>>(N, K, d_dummy);
        cudaDeviceSynchronize();
        cudaMemset(d_dummy, 0, sizeof(unsigned long long));
        cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
        cudaEventRecord(t0);
        inversion_only_kernel<<<blocks,256>>>(N, K, d_dummy);
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float ms=0; cudaEventElapsedTime(&ms,t0,t1);
        printf("[bench] Test A (scan+inversion only, no Bloom1/mixed_add): %.3f ms for %d rounds -> %.3f us/round\n",
               ms, K, ms*1000.0/K);
        cudaFree(d_dummy); cudaEventDestroy(t0); cudaEventDestroy(t1);
    }

    {
        uint64_t M = 2147483648ULL;
        BloomFilterGPU bf; bloom_size(&bf, M, DEFAULT_BLOOM_FP_RATE);
        double gb = bf.nwords*8.0/1e9;
        printf("[bench] Test B Bloom1 size: %.2f GB\n", gb);
        uint64_t *d_bits; cudaError_t err = cudaMalloc(&d_bits, bf.nwords*sizeof(uint64_t));
        if(err!=cudaSuccess){ printf("[bench] Test B SKIPPED - could not allocate %.2fGB (%s)\n", gb, cudaGetErrorString(err)); return 1; }
        cudaMemset(d_bits, 0, bf.nwords*sizeof(uint64_t));

        ECPoint *h_pts = new ECPoint[N];
        for(int i=0;i<N;++i){
            h_pts[i].x.v[0]=mix64(i); h_pts[i].x.v[1]=mix64(i+1); h_pts[i].x.v[2]=mix64(i+2); h_pts[i].x.v[3]=mix64(i+3)%0xFFFFFFFEFFFFFC2FULL;
            h_pts[i].y.v[0]=mix64(i+100); h_pts[i].y.v[1]=mix64(i+101); h_pts[i].y.v[2]=mix64(i+102); h_pts[i].y.v[3]=mix64(i+103)%0xFFFFFFFEFFFFFC2FULL;
            h_pts[i].infinity=false;
        }
        ECPoint *d_pts; cudaMalloc(&d_pts, N*sizeof(ECPoint));
        cudaMemcpy(d_pts, h_pts, N*sizeof(ECPoint), cudaMemcpyHostToDevice);
        ECPoint h_shift; h_shift.x.v[0]=12345; h_shift.x.v[1]=0; h_shift.x.v[2]=0; h_shift.x.v[3]=0;
        h_shift.y.v[0]=67890; h_shift.y.v[1]=0; h_shift.y.v[2]=0; h_shift.y.v[3]=0; h_shift.infinity=false;

        unsigned long long *d_count; cudaMalloc(&d_count, sizeof(unsigned long long));
        cudaMemset(d_count, 0, sizeof(unsigned long long));
        int blocks=(N+255)/256;
        bloom1_multiround_kernel<<<blocks,256>>>(d_pts, N, h_shift, K, d_bits, bf.bit_count, bf.k_hashes, bf.block_bits, d_count);
        cudaDeviceSynchronize();
        cudaMemcpy(d_pts, h_pts, N*sizeof(ECPoint), cudaMemcpyHostToDevice);
        cudaMemset(d_count, 0, sizeof(unsigned long long));

        cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
        cudaEventRecord(t0);
        bloom1_multiround_kernel<<<blocks,256>>>(d_pts, N, h_shift, K, d_bits, bf.bit_count, bf.k_hashes, bf.block_bits, d_count);
        cudaEventRecord(t1);
        cudaError_t kerr = cudaEventSynchronize(t1);
        if(kerr!=cudaSuccess){ printf("[bench] Test B kernel FAILED: %s\n", cudaGetErrorString(kerr)); return 1; }
        float ms=0; cudaEventElapsedTime(&ms,t0,t1);
        double total_checks = (double)N*K;
        printf("[bench] Test B (full kernel, real %.2fGB Bloom1): %.3f ms for %d rounds -> %.3f us/round, %.2f Mchecks/sec\n",
               gb, ms, K, ms*1000.0/K, total_checks/ms/1000.0);

        delete[] h_pts; cudaFree(d_pts); cudaFree(d_bits); cudaFree(d_count);
        cudaEventDestroy(t0); cudaEventDestroy(t1);
    }

    return 0;
}
