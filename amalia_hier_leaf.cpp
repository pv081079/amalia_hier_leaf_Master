
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cinttypes>
#include <cerrno>
#include <ctime>
#include <cmath>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <atomic>
#include <thread>
#include <mutex>
#include <functional>
#ifdef USE_NUMA
#include <numa.h>

#endif
static int g_numa_aware = 0;
static int g_unsafe_prune = 0;
static int g_prune_repeat_n = 4;
#include <chrono>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <netdb.h>
#include <string>
#include <unistd.h>
#include <sys/mman.h>
#include <omp.h>
#include "secp256k1/SECP256k1.h"
#ifdef GPU_ENABLED
#include "gpu_hierarchy.h"

extern "C" void gpu_i2loop_debug(GpuShiftTable*, const GpuECPoint*, uint64_t*);
#endif

#define PUBKEY_SIZE 33
#define NARROW_STEPS 32
#define DEFAULT_BLOOM_FP_RATE 1e-3

static void die(const char *m){ fprintf(stderr,"ERROR: %s\n",m); exit(EXIT_FAILURE); }
static void die_errno(const char *m){ perror(m); exit(EXIT_FAILURE); }

static double now_seconds(void){
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC,&ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec/1e9;
}

static uint64_t fingerprint_32(const unsigned char *x){
    uint64_t v=0;
    for(int i=0;i<8;++i) v=(v<<8)|x[i];
    v+=UINT64_C(0x9e3779b97f4a7c15);
    v=(v^(v>>30))*UINT64_C(0xbf58476d1ce4e5b9);
    v=(v^(v>>27))*UINT64_C(0x94d049bb133111eb);
    v^=v>>31;
    return v;
}

static int hex_value(char c){
    if(c>='0'&&c<='9') return c-'0';
    if(c>='a'&&c<='f') return c-'a'+10;
    if(c>='A'&&c<='F') return c-'A'+10;
    return -1;
}
static int parse_pubkey(const char *hex, unsigned char out[PUBKEY_SIZE]){
    if(!hex) return 0;
    size_t len=strlen(hex); if(len!=66) return 0;
    for(int i=0;i<PUBKEY_SIZE;++i){
        int hi=hex_value(hex[i*2]), lo=hex_value(hex[i*2+1]);
        if(hi<0||lo<0) return 0;
        out[i]=(unsigned char)((hi<<4)|lo);
    }
    if(*out!= 0x02 && *out!= 0x03) return 0;
    return 1;
}
static int pub_to_point(Secp256K1 &secp, const unsigned char pub[PUBKEY_SIZE], Point &out){
    char hex[PUBKEY_SIZE*2+1];
    for(int i=0;i<PUBKEY_SIZE;++i) snprintf(hex+i*2,3,"%02x",pub[i]);
    bool isComp;
    if(!secp.ParsePublicKeyHex(hex,out,isComp)) return 0;
    out.Reduce();
    return 1;
}

typedef struct { uint64_t *bits; uint64_t bit_count, nwords; uint32_t k_hashes; uint64_t block_bits; } BloomFilter;

static void bloom_init(BloomFilter *bf, uint64_t entries, double fp_rate, int force_k, uint64_t block_bits){
    if(entries==0) entries=1;
    if(fp_rate<=0.0||fp_rate>=1.0) fp_rate=DEFAULT_BLOOM_FP_RATE;
    const double ln2=0.6931471805599453094172321214581765680755;
    double num_bits=-((double)entries*log(fp_rate))/(ln2*ln2);
    if(num_bits<64.0) num_bits=64.0;
    uint64_t bc=(uint64_t)num_bits;
    if(block_bits>0){

        bc = ((bc + block_bits - 1) / block_bits) * block_bits;
    }
    uint32_t kh;
    if(force_k>0){

        kh = (uint32_t)force_k;
    } else {
        double k=(num_bits/(double)entries)*ln2;
        kh=(uint32_t)(k+0.5);
    }
    if(kh<1)kh=1; if(kh>32)kh=32;
    bf->bit_count=bc;
    bf->nwords=(bc+63)/64;
    bf->block_bits=block_bits;

    size_t bytes = (size_t)bf->nwords*sizeof(uint64_t);
    void *mem = mmap(NULL, bytes, PROT_READ|PROT_WRITE,
                      MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    if(mem==MAP_FAILED) die("oom bloom (mmap)");
#ifdef MADV_HUGEPAGE
    madvise(mem, bytes, MADV_HUGEPAGE);
#endif
    bf->bits = (uint64_t*)mem;
    bf->k_hashes=kh;
#ifdef USE_NUMA
    if(g_numa_aware && numa_available()>=0 && numa_num_configured_nodes()>1){

        numa_interleave_memory(mem, bytes, numa_all_nodes_ptr);
    }
#endif
}
static void bloom_free(BloomFilter *bf){
    if(bf->bits) munmap(bf->bits, (size_t)bf->nwords*sizeof(uint64_t));
    bf->bits=nullptr; bf->bit_count=bf->nwords=bf->k_hashes=0; bf->block_bits=0;
}

static inline void bloom_positions(const BloomFilter *bf, uint64_t fp, uint64_t *out){
    uint64_t h1=fp;
    uint64_t h2=fp+UINT64_C(0x9e3779b97f4a7c15);
    h2=(h2^(h2>>30))*UINT64_C(0xbf58476d1ce4e5b9);
    h2=(h2^(h2>>27))*UINT64_C(0x94d049bb133111eb);
    h2^=h2>>31; if(h2==0)h2=1;
    if(bf->block_bits>0){
        uint64_t num_blocks = bf->bit_count / bf->block_bits;
        if(num_blocks<1) num_blocks=1;
        uint64_t block = h1 % num_blocks;
        uint64_t block_base_bit = block * bf->block_bits;
        for(uint32_t i=0;i<bf->k_hashes;++i){
            uint64_t sub = (h1 + (uint64_t)i*h2) % bf->block_bits;
            out[i] = block_base_bit + sub;
        }
    } else {
        for(uint32_t i=0;i<bf->k_hashes;++i) out[i]=(h1+(uint64_t)i*h2)%bf->bit_count;
    }
}

static int bloom_check(const BloomFilter *bf, uint64_t fp){
    uint64_t positions[32];
    bloom_positions(bf,fp,positions);
    for(uint32_t i=0;i<bf->k_hashes;++i){
        uint64_t pos=positions[i];
        uint64_t word = bf->bits[pos>>6];
        if(!(word & (UINT64_C(1)<<(pos&63)))) return 0;
    }
    return 1;
}
static inline void bloom_prefetch(const BloomFilter *bf, uint64_t fp){
    uint64_t positions[32];
    bloom_positions(bf,fp,positions);
    for(uint32_t i=0;i<bf->k_hashes;++i){
        __builtin_prefetch(&bf->bits[positions[i]>>6], 0, 1);
    }
}

static inline void bloom_add_atomic(BloomFilter *bf, uint64_t fp){
    uint64_t positions[32];
    bloom_positions(bf,fp,positions);
    for(uint32_t i=0;i<bf->k_hashes;++i){
        uint64_t pos=positions[i];
        uint64_t mask = UINT64_C(1)<<(pos&63);
        __atomic_fetch_or(&bf->bits[pos>>6], mask, __ATOMIC_RELAXED);
    }
}

typedef struct { uint64_t fp; uint32_t idx; } BabyEntry;

static inline uint32_t baby_index(uint64_t fp, int K){
    return (uint32_t)(fp >> (64-K));
}
static void baby_insert(BabyEntry *table, uint64_t array_size, int K, uint64_t fp, uint32_t idx){
    uint32_t pos = baby_index(fp, K);
    while(table[pos].idx!=UINT32_MAX){
        pos = (uint32_t)((pos+1) & (array_size-1));
    }
    table[pos].fp = fp;
    table[pos].idx = idx;
}
static inline int32_t baby_find(BabyEntry *table, uint64_t array_size, int K, uint64_t fp){
    uint32_t pos = baby_index(fp, K);
    for(uint64_t probes=0; probes<array_size; ++probes){
        if(table[pos].idx==UINT32_MAX) return -1;
        if(table[pos].fp==fp) return (int32_t)pos;
        pos = (uint32_t)((pos+1) & (array_size-1));
    }
    return -1;
}

#define R4_LEVELS 4
#define R4_RADIX 4

struct Radix4State {
    uint64_t level_M[R4_LEVELS];
    uint64_t exact_M;
    BloomFilter bloom[R4_LEVELS];
    Point shift[R4_LEVELS][R4_RADIX];
    Point pos[R4_LEVELS][R4_RADIX];
    BabyEntry *baby_table = nullptr;
    uint64_t baby_array_size;
    int baby_K;
};

static bool net_recv_line(int fd, std::string &out){
    out.clear();
    char c;
    while(true){
        ssize_t n = recv(fd, &c, 1, 0);
        if(n<=0) return !out.empty();
        if(c=='\n') return true;
        if(c!='\r') out.push_back(c);
        if(out.size()>4096) return false;
    }
}

static bool net_send_line(int fd, const std::string &line){
    std::string full = line + "\n";
    size_t sent = 0;
    while(sent < full.size()){
        ssize_t n = send(fd, full.data()+sent, full.size()-sent, 0);
        if(n<=0) return false;
        sent += (size_t)n;
    }
    return true;
}

static int net_listen(int port){
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if(fd<0) return -1;
    int one=1; setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    struct sockaddr_in addr; memset(&addr,0,sizeof(addr));
    addr.sin_family = AF_INET; addr.sin_addr.s_addr = INADDR_ANY; addr.sin_port = htons((uint16_t)port);
    if(bind(fd,(struct sockaddr*)&addr,sizeof(addr))<0){ close(fd); return -1; }
    if(listen(fd, 64)<0){ close(fd); return -1; }
    return fd;
}

static int net_connect(const char *host, int port){
    struct addrinfo hints; memset(&hints,0,sizeof(hints));
    hints.ai_family = AF_INET; hints.ai_socktype = SOCK_STREAM;
    char portstr[16]; snprintf(portstr,sizeof(portstr),"%d",port);
    struct addrinfo *res=nullptr;
    if(getaddrinfo(host, portstr, &hints, &res)!=0 || !res) return -1;
    int fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if(fd<0){ freeaddrinfo(res); return -1; }
    if(connect(fd, res->ai_addr, res->ai_addrlen)<0){ close(fd); freeaddrinfo(res); return -1; }
    freeaddrinfo(res);
    int one=1; setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    return fd;
}

static std::vector<std::string> net_split(const std::string &line){
    std::vector<std::string> out;
    size_t i=0;
    while(i<line.size()){
        size_t j=line.find(' ', i);
        if(j==std::string::npos){ out.push_back(line.substr(i)); break; }
        if(j>i) out.push_back(line.substr(i,j-i));
        i=j+1;
    }
    return out;
}

struct HierState {
    uint64_t M, M2, M3;
    BloomFilter bloom1, bloom2, bloom3;
    BabyEntry *baby_table = nullptr;
    uint64_t baby_array_size;
    int baby_K;
    Point shift2[NARROW_STEPS];
    Point shift3[NARROW_STEPS];
    Point pos2[NARROW_STEPS];
    Point pos3[NARROW_STEPS];
    Radix4State r4;
    int radix4_enabled = 0;
};

static void precompute_shifts(Secp256K1 &secp, HierState &st){
    Int m2; m2.SetInt64((int64_t)st.M2);
    Int m3; m3.SetInt64((int64_t)st.M3);
    for(int i=1;i<NARROW_STEPS;++i){
        Int k2; k2.SetInt64((int64_t)i); k2.Mult(&m2);
        Point p2 = secp.ComputePublicKey(&k2); p2.Reduce();
        st.pos2[i] = p2;
        st.shift2[i] = secp.Negation(p2); st.shift2[i].Reduce();

        Int k3; k3.SetInt64((int64_t)i); k3.Mult(&m3);
        Point p3 = secp.ComputePublicKey(&k3); p3.Reduce();
        st.pos3[i] = p3;
        st.shift3[i] = secp.Negation(p3); st.shift3[i].Reduce();
    }
}

static void precompute_shifts_radix4(Secp256K1 &secp, HierState &st){
    for(int lvl=0; lvl<R4_LEVELS; ++lvl){
        Int lvlM; lvlM.SetInt64((int64_t)st.r4.level_M[lvl]);
        for(int b=1; b<R4_RADIX; ++b){
            Int k; k.SetInt64((int64_t)b); k.Mult(&lvlM);
            Point p = secp.ComputePublicKey(&k); p.Reduce();
            st.r4.pos[lvl][b] = p;
            st.r4.shift[lvl][b] = secp.Negation(p); st.r4.shift[lvl][b].Reduce();
        }
    }
}

static inline bool same_point(Point &a, Point &b){
    return a.x.IsEqual(&b.x) && (a.y.GetBit(0)==b.y.GetBit(0));
}

static inline uint64_t point_fingerprint(Point &R){
    unsigned char xb[32]; R.x.Get32Bytes(xb);
    return fingerprint_32(xb);
}

static std::atomic<uint64_t> stat_bloom1_pass{0};
static std::atomic<uint64_t> stat_i2_adds{0};
static std::atomic<uint64_t> stat_bloom2_pass{0};
static std::atomic<uint64_t> stat_i3_adds{0};
static std::atomic<uint64_t> stat_bloom3_pass{0};
static std::atomic<uint64_t> stat_exact_lookups{0};
static std::atomic<uint64_t> stat_try_candidate{0};

static std::atomic<uint64_t> time_bloom1_ns{0};
static std::atomic<uint64_t> time_i2_build_ns{0};
static std::atomic<uint64_t> time_i2_batchinv_ns{0};
static std::atomic<uint64_t> time_i2_normalize_bloom2_ns{0};
static std::atomic<uint64_t> time_i3_ns{0};

static inline uint64_t now_ns(){
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static inline Point add_mixed_z2eq1(Point &p1, Point &p2);
static void batch_invert_noalloc(Int *inv_out, Int *z, Int *pref, uint64_t n);

static int hierarchical_lookup(Secp256K1 &secp, HierState &st, Point &R, uint64_t fp0, uint64_t *p_out){
    uint64_t t0 = now_ns();
    unsigned char xb[32]; R.x.Get32Bytes(xb);
    if(!bloom_check(&st.bloom1,fp0)){
        time_bloom1_ns.fetch_add(now_ns()-t0, std::memory_order_relaxed);
        return 0;
    }
    time_bloom1_ns.fetch_add(now_ns()-t0, std::memory_order_relaxed);
    stat_bloom1_pass.fetch_add(1, std::memory_order_relaxed);

    auto try_candidate = [&](uint64_t q)->bool{
        stat_try_candidate.fetch_add(1, std::memory_order_relaxed);
        if(q>=st.M) return false;
        Int pm; pm.SetInt64((int64_t)q);
        Point pj = secp.ComputePublicKey(&pm); pj.Reduce();
        unsigned char checkX[32]; pj.x.Get32Bytes(checkX);
        if(memcmp(checkX,xb,32)==0 && (pj.y.GetBit(0)==R.y.GetBit(0))){
            *p_out = q;
            return true;
        }
        return false;
    };

    {
        uint64_t ti = now_ns();
        Point S2 = R;
        unsigned char xb2[32]; S2.x.Get32Bytes(xb2);
        uint64_t fp2 = fingerprint_32(xb2);
        bool passed = bloom_check(&st.bloom2,fp2);
        time_i2_normalize_bloom2_ns.fetch_add(now_ns()-ti, std::memory_order_relaxed);
        if(passed){
            stat_bloom2_pass.fetch_add(1, std::memory_order_relaxed);
            uint64_t t3 = now_ns();
            for(int i3=0; i3<NARROW_STEPS; ++i3){
                bool degenerate3 = (i3>0) && same_point(S2, st.pos3[i3]);
                if(degenerate3){
                    if(try_candidate((uint64_t)0*st.M2 + (uint64_t)i3*st.M3)){
                        time_i3_ns.fetch_add(now_ns()-t3, std::memory_order_relaxed);
                        return 1;
                    }
                    continue;
                }
                Point S3;
                if(i3==0){ S3 = S2; }
                else { S3 = secp.Add(S2,st.shift3[i3]); S3.Reduce(); stat_i3_adds.fetch_add(1, std::memory_order_relaxed); }
                unsigned char xb3[32]; S3.x.Get32Bytes(xb3);
                uint64_t fp3 = fingerprint_32(xb3);
                if(!bloom_check(&st.bloom3,fp3)) continue;
                stat_bloom3_pass.fetch_add(1, std::memory_order_relaxed);
                stat_exact_lookups.fetch_add(1, std::memory_order_relaxed);
                int32_t pos = baby_find(st.baby_table, st.baby_array_size, st.baby_K, fp3);
                if(pos>=0){
                    uint64_t q = (uint64_t)0*st.M2 + (uint64_t)i3*st.M3 + st.baby_table[pos].idx;
                    if(try_candidate(q)){
                        time_i3_ns.fetch_add(now_ns()-t3, std::memory_order_relaxed);
                        return 1;
                    }
                }
            }
            time_i3_ns.fetch_add(now_ns()-t3, std::memory_order_relaxed);
        }
    }

    Point S2proj[NARROW_STEPS];
    int   S2idx[NARROW_STEPS];
    Int   S2z[NARROW_STEPS], S2zinv[NARROW_STEPS], S2pref[NARROW_STEPS];
    int n_pending = 0;
    uint64_t tb = now_ns();
    for(int i2=1; i2<NARROW_STEPS; ++i2){
        if(same_point(R, st.pos2[i2])){
            if(try_candidate((uint64_t)i2*st.M2)) return 1;
            continue;
        }
        Point p = add_mixed_z2eq1(R, st.shift2[i2]);
        S2proj[n_pending] = p;
        S2idx[n_pending] = i2;
        S2z[n_pending] = p.z;
        n_pending++;
        stat_i2_adds.fetch_add(1, std::memory_order_relaxed);
    }
    time_i2_build_ns.fetch_add(now_ns()-tb, std::memory_order_relaxed);

    uint64_t tinv = now_ns();
    if(n_pending>0){
        batch_invert_noalloc(S2zinv, S2z, S2pref, (uint64_t)n_pending);
    }
    time_i2_batchinv_ns.fetch_add(now_ns()-tinv, std::memory_order_relaxed);

    for(int k=0;k<n_pending;++k){
        uint64_t tn = now_ns();
        int i2 = S2idx[k];
        Point S2;
        S2.x = S2proj[k].x; S2.x.ModMul(&S2zinv[k]);
        S2.y = S2proj[k].y; S2.y.ModMul(&S2zinv[k]);
        S2.z.SetInt32(1);

        unsigned char xb2[32]; S2.x.Get32Bytes(xb2);
        uint64_t fp2 = fingerprint_32(xb2);
        bool passed2 = bloom_check(&st.bloom2,fp2);
        time_i2_normalize_bloom2_ns.fetch_add(now_ns()-tn, std::memory_order_relaxed);
        if(!passed2) continue;
        stat_bloom2_pass.fetch_add(1, std::memory_order_relaxed);

        uint64_t t3b = now_ns();
        for(int i3=0; i3<NARROW_STEPS; ++i3){
            bool degenerate3 = (i3>0) && same_point(S2, st.pos3[i3]);
            if(degenerate3){
                if(try_candidate((uint64_t)i2*st.M2 + (uint64_t)i3*st.M3)){
                    time_i3_ns.fetch_add(now_ns()-t3b, std::memory_order_relaxed);
                    return 1;
                }
                continue;
            }
            Point S3;
            if(i3==0){ S3 = S2; }
            else { S3 = secp.Add(S2,st.shift3[i3]); S3.Reduce(); stat_i3_adds.fetch_add(1, std::memory_order_relaxed); }
            unsigned char xb3[32]; S3.x.Get32Bytes(xb3);
            uint64_t fp3 = fingerprint_32(xb3);
            if(!bloom_check(&st.bloom3,fp3)) continue;
            stat_bloom3_pass.fetch_add(1, std::memory_order_relaxed);

            stat_exact_lookups.fetch_add(1, std::memory_order_relaxed);
            int32_t pos = baby_find(st.baby_table, st.baby_array_size, st.baby_K, fp3);
            if(pos>=0){
                uint64_t q = (uint64_t)i2*st.M2 + (uint64_t)i3*st.M3 + st.baby_table[pos].idx;
                if(try_candidate(q)){
                    time_i3_ns.fetch_add(now_ns()-t3b, std::memory_order_relaxed);
                    return 1;
                }
            }
        }
        time_i3_ns.fetch_add(now_ns()-t3b, std::memory_order_relaxed);
    }
    return 0;
}

static int hierarchical_lookup_radix4(Secp256K1 &secp, HierState &st, Point &R, uint64_t fp0, uint64_t *p_out){
    uint64_t t0 = now_ns();
    unsigned char xb[32]; R.x.Get32Bytes(xb);
    if(!bloom_check(&st.bloom1,fp0)){
        time_bloom1_ns.fetch_add(now_ns()-t0, std::memory_order_relaxed);
        return 0;
    }
    time_bloom1_ns.fetch_add(now_ns()-t0, std::memory_order_relaxed);
    stat_bloom1_pass.fetch_add(1, std::memory_order_relaxed);

    auto try_candidate = [&](uint64_t q)->bool{
        stat_try_candidate.fetch_add(1, std::memory_order_relaxed);
        if(q>=st.M) return false;
        Int pm; pm.SetInt64((int64_t)q);
        Point pj = secp.ComputePublicKey(&pm); pj.Reduce();
        unsigned char checkX[32]; pj.x.Get32Bytes(checkX);
        if(memcmp(checkX,xb,32)==0 && (pj.y.GetBit(0)==R.y.GetBit(0))){
            *p_out = q;
            return true;
        }
        return false;
    };

    std::function<bool(Point&, int, uint64_t)> descend =
        [&](Point &S, int lvl, uint64_t idx_so_far)->bool{
        if(lvl==R4_LEVELS){
            unsigned char sx[32]; S.x.Get32Bytes(sx);
            uint64_t fp = fingerprint_32(sx);
            stat_bloom3_pass.fetch_add(1, std::memory_order_relaxed);
            stat_exact_lookups.fetch_add(1, std::memory_order_relaxed);
            int32_t pos = baby_find(st.r4.baby_table, st.r4.baby_array_size, st.r4.baby_K, fp);
            if(pos>=0){
                uint64_t q = idx_so_far + st.r4.baby_table[pos].idx;
                if(try_candidate(q)) return true;
            }
            return false;
        }
        for(int b=0; b<R4_RADIX; ++b){
            bool degenerate = (b>0) && same_point(S, st.r4.pos[lvl][b]);
            uint64_t branch_idx = idx_so_far + (uint64_t)b * st.r4.level_M[lvl];
            if(degenerate){
                if(try_candidate(branch_idx)) return true;
                continue;
            }
            Point Snext;
            if(b==0){ Snext = S; }
            else { Snext = secp.Add(S, st.r4.shift[lvl][b]); Snext.Reduce();
                   stat_i2_adds.fetch_add(1, std::memory_order_relaxed);  }
            unsigned char snx[32]; Snext.x.Get32Bytes(snx);
            uint64_t fpn = fingerprint_32(snx);
            if(!bloom_check(&st.r4.bloom[lvl], fpn)) continue;
            stat_bloom2_pass.fetch_add(1, std::memory_order_relaxed);
            if(descend(Snext, lvl+1, branch_idx)) return true;
        }
        return false;
    };

    return descend(R, 0, 0) ? 1 : 0;
}

static bool run_net_self_test(int port){
    printf("[net-selftest] listening on 127.0.0.1:%d\n", port);
    int lfd = net_listen(port);
    if(lfd<0){ printf("[net-selftest] FAILED - could not listen (port in use?)\n"); return false; }

    std::atomic<bool> server_ok{false};
    std::atomic<bool> server_done{false};
    std::thread server_thread([&](){
        struct sockaddr_in peer; socklen_t peerlen=sizeof(peer);
        int cfd = accept(lfd, (struct sockaddr*)&peer, &peerlen);
        if(cfd<0){ server_done.store(true); return; }
        std::string line;
        if(net_recv_line(cfd, line) && line=="PING 12345"){
            net_send_line(cfd, "PONG 12345");
            server_ok.store(true);
        }
        close(cfd);
        server_done.store(true);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    int cfd = net_connect("127.0.0.1", port);
    bool ok = false;
    if(cfd>=0){
        if(net_send_line(cfd, "PING 12345")){
            std::string resp;
            if(net_recv_line(cfd, resp) && resp=="PONG 12345") ok = true;
        }
        close(cfd);
    }
    server_thread.join();
    close(lfd);

    ok = ok && server_ok.load();
    printf("[net-selftest] %s - round-trip %s\n", ok?"PASSED":"FAILED",
           ok?"PING 12345 -> PONG 12345 confirmed both directions":"did not complete correctly");
    return ok;
}

static inline Point add_mixed_z2eq1(Point &p1, Point &p2){
    Int u1,v1,u,v,us2,vs2,vs3,us2w,vs2v2,_2vs2v2,a,vs3u2;
    Point r;
    u1.ModMulK1(&p2.y,&p1.z);
    v1.ModMulK1(&p2.x,&p1.z);
    u.ModSub(&u1,&p1.y);
    v.ModSub(&v1,&p1.x);
    us2.ModSquareK1(&u);
    vs2.ModSquareK1(&v);
    vs3.ModMulK1(&vs2,&v);
    us2w.ModMulK1(&us2,&p1.z);
    vs2v2.ModMulK1(&vs2,&p1.x);
    _2vs2v2.ModAdd(&vs2v2,&vs2v2);
    a.ModSub(&us2w,&vs3);
    a.ModSub(&_2vs2v2);
    r.x.ModMulK1(&v,&a);
    vs3u2.ModMulK1(&vs3,&p1.y);
    r.y.ModSub(&vs2v2,&a);
    r.y.ModMulK1(&r.y,&u);
    r.y.ModSub(&vs3u2);
    r.z.ModMulK1(&vs3,&p1.z);
    return r;
}
static void batch_invert_noalloc(Int *inv_out, Int *z, Int *pref, uint64_t n){
    if(n==0) return;
    pref[0]=z[0];
    for(uint64_t i=1;i<n;++i){ pref[i]=pref[i-1]; pref[i].ModMul(&z[i]); }
    Int inv; inv.Set(&pref[n-1]); inv.ModInv();
    for(uint64_t i=n;i-->1;){
        inv_out[i]=inv; inv_out[i].ModMul(&pref[i-1]); inv.ModMul(&z[i]);
    }
    inv_out[0]=inv;
}

static void build_hierarchy(Secp256K1 &secp, HierState &st, uint64_t M,
                            int threads, int lanes_per_thread, double fp_rate, int bloom1_k,
                            uint64_t bloom1_block_bits){
    st.M = M;
    st.M2 = (M+31)/32;
    st.M3 = (st.M2+31)/32;

    printf("[+] Hierarchy: M=%" PRIu64 " M2=%" PRIu64 " M3=%" PRIu64 "\n", st.M, st.M2, st.M3);

    bloom_init(&st.bloom1, st.M,  fp_rate, bloom1_k, bloom1_block_bits);
    bloom_init(&st.bloom2, st.M2, fp_rate, 0, 0);
    bloom_init(&st.bloom3, st.M3, fp_rate, 0, 0);
    printf("[+] Bloom1 (M):  %.2f MB\n", (double)(st.bloom1.nwords*8)/1e6);
    printf("[+] Bloom2 (M2): %.2f MB\n", (double)(st.bloom2.nwords*8)/1e6);
    printf("[+] Bloom3 (M3): %.2f MB\n", (double)(st.bloom3.nwords*8)/1e6);
    printf("[+] Exact table (M3): %.2f MB\n", (double)(st.M3*sizeof(BabyEntry))/1e6);

    BabyEntry *flat = (BabyEntry*)malloc((size_t)st.M3*sizeof(BabyEntry));
    if(!flat) die("oom baby table (temporary)");
    for(uint64_t i=0;i<st.M3;++i){ flat[i].fp=0; flat[i].idx=UINT32_MAX; }

    int T = threads; if(T<1) T=1;
    int B = lanes_per_thread; if(B<1) B=1;
    uint64_t L = (uint64_t)T*(uint64_t)B;
    if(L > M) L = M>0?M:1;
    Int Lint; Lint.SetInt64((int64_t)L);
    Point stepG = secp.ComputePublicKey(&Lint); stepG.Reduce();

    std::atomic<uint64_t> total_done{0};
    double tstart = now_seconds();

    #pragma omp parallel num_threads(T)
    {
        int tid = omp_get_thread_num();
        Point *P = new Point[B];
        Int *ax=new Int[B], *ay=new Int[B];
        Int *zx=new Int[B], *zy=new Int[B], *zz=new Int[B], *zinv=new Int[B], *pref=new Int[B];
        uint64_t *curp = new uint64_t[B];

        for(int l=0;l<B;++l){
            uint64_t gidx = (uint64_t)tid + (uint64_t)l*T;
            uint64_t p0 = gidx+1;
            if(p0>=M){ curp[l]=0; continue; }
            curp[l]=p0;
            Int p0I; p0I.SetInt64((int64_t)p0);
            Point pt = secp.ComputePublicKey(&p0I); pt.Reduce();
            P[l]=pt; ax[l]=pt.x; ay[l]=pt.y;
        }

        bool any_active=true;
        while(any_active){
            any_active=false;
            for(int l=0;l<B;++l){
                if(curp[l]==0) continue;
                any_active=true;
                uint64_t p = curp[l];
                unsigned char xb[32]; Int tmp=ax[l]; tmp.Get32Bytes(xb);
                uint64_t fp = fingerprint_32(xb);
                bloom_add_atomic(&st.bloom1, fp);
                if(p<st.M2) bloom_add_atomic(&st.bloom2, fp);
                if(p<st.M3){
                    bloom_add_atomic(&st.bloom3, fp);
                    flat[p].fp = fp;
                    flat[p].idx = (uint32_t)p;
                }
            }
            if(!any_active) break;
            for(int l=0;l<B;++l){
                if(curp[l]==0) continue;
                uint64_t nextp = curp[l]+L;
                if(nextp>=M){ curp[l]=0; continue; }
                if(curp[l]==L){ P[l]=secp.Double(P[l]); }
                else          { P[l]=add_mixed_z2eq1(P[l],stepG); }
                zx[l]=P[l].x; zy[l]=P[l].y; zz[l]=P[l].z;
                curp[l]=nextp;
            }
            batch_invert_noalloc(zinv,zz,pref,(uint64_t)B);
            for(int l=0;l<B;++l){ ax[l]=zx[l]; ax[l].ModMul(&zinv[l]); ay[l]=zy[l]; ay[l].ModMul(&zinv[l]); }

            uint64_t done = total_done.fetch_add(B)+B;
            if(tid==0 && (done & 0xFFFFF)==0){
                double el = now_seconds()-tstart;
                printf("  [build] ~%" PRIu64 "/%" PRIu64 " points, %.1fs, %.2fM/s\r",
                       done, M, el, el>0?(done/el)/1e6:0.0);
                fflush(stdout);
            }
        }
        delete[] P; delete[] ax; delete[] ay; delete[] zx; delete[] zy; delete[] zz; delete[] zinv; delete[] pref; delete[] curp;
    }
    printf("\n");

    st.baby_K = (int)ceil(log2((double)st.M3)) + 2;
    if(st.baby_K<1) st.baby_K=1;
    st.baby_array_size = (uint64_t)1<<st.baby_K;
    st.baby_table = (BabyEntry*)malloc((size_t)st.baby_array_size*sizeof(BabyEntry));
    if(!st.baby_table) die("oom baby table (MSB)");
    for(uint64_t i=0;i<st.baby_array_size;++i){ st.baby_table[i].fp=0; st.baby_table[i].idx=UINT32_MAX; }
    for(uint64_t i=0;i<st.M3;++i){
        if(flat[i].idx==UINT32_MAX) continue;
        baby_insert(st.baby_table, st.baby_array_size, st.baby_K, flat[i].fp, flat[i].idx);
    }
    free(flat);
    printf("[+] Exact table MSB-indexed (K=%d, array=2^%d=%" PRIu64 ", load factor=%.1f%%)\n",
           st.baby_K, st.baby_K, st.baby_array_size, 100.0*(double)st.M3/(double)st.baby_array_size);

    precompute_shifts(secp, st);
    printf("[+] Shifts (i2/i3) precomputed\n");
}

static void free_hierarchy(HierState &st){
    bloom_free(&st.bloom1); bloom_free(&st.bloom2); bloom_free(&st.bloom3);
    free(st.baby_table); st.baby_table=nullptr;
}

static void build_hierarchy_radix4(Secp256K1 &secp, HierState &st, uint64_t M,
                                   int threads, int lanes_per_thread, double fp_rate,
                                   int bloom1_k, uint64_t bloom1_block_bits){
    st.M = M;
    st.radix4_enabled = 1;
    for(int lvl=0; lvl<R4_LEVELS; ++lvl){
        st.r4.level_M[lvl] = M >> (2*(lvl+1));
    }
    st.r4.exact_M = st.r4.level_M[R4_LEVELS-1];

    printf("[+] Radix-4 hierarchy: M=%" PRIu64 " levels=", M);
    for(int lvl=0; lvl<R4_LEVELS; ++lvl) printf("%" PRIu64 " ", st.r4.level_M[lvl]);
    printf("exact=%" PRIu64 "\n", st.r4.exact_M);

    bloom_init(&st.bloom1, st.M, fp_rate, bloom1_k, bloom1_block_bits);
    for(int lvl=0; lvl<R4_LEVELS; ++lvl){
        bloom_init(&st.r4.bloom[lvl], st.r4.level_M[lvl], fp_rate, 0, 0);
        printf("[+] Bloom-L%d (%" PRIu64 "): %.2f MB\n", lvl+1, st.r4.level_M[lvl],
               (double)(st.r4.bloom[lvl].nwords*8)/1e6);
    }
    printf("[+] Bloom1: %.2f MB\n", (double)(st.bloom1.nwords*8)/1e6);
    printf("[+] Exact table (%" PRIu64 "): %.2f MB\n", st.r4.exact_M,
           (double)(st.r4.exact_M*sizeof(BabyEntry))/1e6);

    BabyEntry *flat = (BabyEntry*)malloc((size_t)st.r4.exact_M*sizeof(BabyEntry));
    if(!flat) die("oom baby table (temporary, radix4)");
    for(uint64_t i=0;i<st.r4.exact_M;++i){ flat[i].fp=0; flat[i].idx=UINT32_MAX; }

    int T = threads; if(T<1) T=1;
    int B = lanes_per_thread; if(B<1) B=1;
    uint64_t L = (uint64_t)T*(uint64_t)B;
    if(L > M) L = M>0?M:1;
    Int Lint; Lint.SetInt64((int64_t)L);
    Point stepG = secp.ComputePublicKey(&Lint); stepG.Reduce();

    std::atomic<uint64_t> total_done{0};
    double tstart = now_seconds();

    #pragma omp parallel num_threads(T)
    {
        int tid = omp_get_thread_num();
        Point *P = new Point[B];
        Int *ax=new Int[B], *ay=new Int[B];
        Int *zx=new Int[B], *zy=new Int[B], *zz=new Int[B], *zinv=new Int[B], *pref=new Int[B];
        uint64_t *curp = new uint64_t[B];

        for(int l=0;l<B;++l){
            uint64_t gidx = (uint64_t)tid + (uint64_t)l*T;
            uint64_t p0 = gidx+1;
            if(p0>=M){ curp[l]=0; continue; }
            curp[l]=p0;
            Int p0I; p0I.SetInt64((int64_t)p0);
            Point pt = secp.ComputePublicKey(&p0I); pt.Reduce();
            P[l]=pt; ax[l]=pt.x; ay[l]=pt.y;
        }

        bool any_active=true;
        while(any_active){
            any_active=false;
            for(int l=0;l<B;++l){
                if(curp[l]==0) continue;
                any_active=true;
                uint64_t p = curp[l];
                unsigned char xb[32]; Int tmp=ax[l]; tmp.Get32Bytes(xb);
                uint64_t fp = fingerprint_32(xb);
                bloom_add_atomic(&st.bloom1, fp);
                for(int lvl=0; lvl<R4_LEVELS; ++lvl){
                    if(p<st.r4.level_M[lvl]) bloom_add_atomic(&st.r4.bloom[lvl], fp);
                }
                if(p<st.r4.exact_M){
                    flat[p].fp = fp;
                    flat[p].idx = (uint32_t)p;
                }
            }
            if(!any_active) break;
            for(int l=0;l<B;++l){
                if(curp[l]==0) continue;
                uint64_t nextp = curp[l]+L;
                if(nextp>=M){ curp[l]=0; continue; }
                if(curp[l]==L){ P[l]=secp.Double(P[l]); }
                else          { P[l]=add_mixed_z2eq1(P[l],stepG); }
                zx[l]=P[l].x; zy[l]=P[l].y; zz[l]=P[l].z;
                curp[l]=nextp;
            }
            batch_invert_noalloc(zinv,zz,pref,(uint64_t)B);
            for(int l=0;l<B;++l){ ax[l]=zx[l]; ax[l].ModMul(&zinv[l]); ay[l]=zy[l]; ay[l].ModMul(&zinv[l]); }

            uint64_t done = total_done.fetch_add(B)+B;
            if(tid==0 && (done & 0xFFFFF)==0){
                double el = now_seconds()-tstart;
                printf("  [build-r4] ~%" PRIu64 "/%" PRIu64 " points, %.1fs, %.2fM/s\r",
                       done, M, el, el>0?(done/el)/1e6:0.0);
                fflush(stdout);
            }
        }
        delete[] P; delete[] ax; delete[] ay; delete[] zx; delete[] zy; delete[] zz; delete[] zinv; delete[] pref; delete[] curp;
    }
    printf("\n");

    st.r4.baby_K = (int)ceil(log2((double)st.r4.exact_M)) + 2;
    if(st.r4.baby_K<1) st.r4.baby_K=1;
    st.r4.baby_array_size = (uint64_t)1<<st.r4.baby_K;
    st.r4.baby_table = (BabyEntry*)malloc((size_t)st.r4.baby_array_size*sizeof(BabyEntry));
    if(!st.r4.baby_table) die("oom baby table (MSB, radix4)");
    for(uint64_t i=0;i<st.r4.baby_array_size;++i){ st.r4.baby_table[i].fp=0; st.r4.baby_table[i].idx=UINT32_MAX; }
    for(uint64_t i=0;i<st.r4.exact_M;++i){
        if(flat[i].idx==UINT32_MAX) continue;
        baby_insert(st.r4.baby_table, st.r4.baby_array_size, st.r4.baby_K, flat[i].fp, flat[i].idx);
    }
    free(flat);
    printf("[+] Radix-4 exact table MSB-indexed (K=%d, array=2^%d=%" PRIu64 ", load factor=%.1f%%)\n",
           st.r4.baby_K, st.r4.baby_K, st.r4.baby_array_size,
           100.0*(double)st.r4.exact_M/(double)st.r4.baby_array_size);

    precompute_shifts_radix4(secp, st);
    printf("[+] Radix-4 shifts precomputed\n");
}

static void free_hierarchy_radix4(HierState &st){
    bloom_free(&st.bloom1);
    for(int lvl=0; lvl<R4_LEVELS; ++lvl) bloom_free(&st.r4.bloom[lvl]);
    free(st.r4.baby_table); st.r4.baby_table=nullptr;
}

#define HIER_MAGIC "AMALIAHR"
#define HIER_MAGIC_SIZE 8
#define HIER_VERSION 3

static void save_hierarchy(HierState &st, const char *path){
    FILE *f=fopen(path,"wb"); if(!f) die_errno("create hierarchy file");
    uint32_t version=HIER_VERSION;
    fwrite(HIER_MAGIC,1,HIER_MAGIC_SIZE,f);
    fwrite(&version,sizeof(uint32_t),1,f);
    fwrite(&st.M,sizeof(uint64_t),1,f);
    fwrite(&st.M2,sizeof(uint64_t),1,f);
    fwrite(&st.M3,sizeof(uint64_t),1,f);
    int32_t baby_K32 = st.baby_K;
    fwrite(&baby_K32,sizeof(int32_t),1,f);
    fwrite(&st.baby_array_size,sizeof(uint64_t),1,f);

    auto write_bloom = [&](BloomFilter &bf){
        fwrite(&bf.bit_count,sizeof(uint64_t),1,f);
        fwrite(&bf.nwords,sizeof(uint64_t),1,f);
        fwrite(&bf.k_hashes,sizeof(uint32_t),1,f);
        fwrite(&bf.block_bits,sizeof(uint64_t),1,f);
        fwrite(bf.bits,sizeof(uint64_t),(size_t)bf.nwords,f);
    };
    write_bloom(st.bloom1);
    write_bloom(st.bloom2);
    write_bloom(st.bloom3);
    fwrite(st.baby_table,sizeof(BabyEntry),(size_t)st.baby_array_size,f);
    fclose(f);
    printf("[+] Hierarchy saved to %s\n", path);
}

static bool load_hierarchy(Secp256K1 &secp, HierState &st, const char *path){
    printf("[+] Opening %s ...\n", path); fflush(stdout);
    FILE *f=fopen(path,"rb"); if(!f) return false;
    char magic[HIER_MAGIC_SIZE];
    if(fread(magic,1,HIER_MAGIC_SIZE,f)!=HIER_MAGIC_SIZE){ fclose(f); return false; }
    if(memcmp(magic,HIER_MAGIC,HIER_MAGIC_SIZE)!=0){ fclose(f); return false; }
    uint32_t version;
    if(fread(&version,sizeof(uint32_t),1,f)!=1 || version!=HIER_VERSION){ fclose(f); return false; }

    if(fread(&st.M,sizeof(uint64_t),1,f)!=1){ fclose(f); return false; }
    if(fread(&st.M2,sizeof(uint64_t),1,f)!=1){ fclose(f); return false; }
    if(fread(&st.M3,sizeof(uint64_t),1,f)!=1){ fclose(f); return false; }
    int32_t baby_K32;
    if(fread(&baby_K32,sizeof(int32_t),1,f)!=1){ fclose(f); return false; }
    st.baby_K = baby_K32;
    if(fread(&st.baby_array_size,sizeof(uint64_t),1,f)!=1){ fclose(f); return false; }
    printf("[+] Header read: M=%" PRIu64 " M2=%" PRIu64 " M3=%" PRIu64
           " baby_K=%d baby_array_size=%" PRIu64 "\n",
           st.M, st.M2, st.M3, st.baby_K, st.baby_array_size);
    fflush(stdout);

    double t0 = now_seconds();
    auto read_bloom = [&](BloomFilter &bf, const char *name)->bool{
        if(fread(&bf.bit_count,sizeof(uint64_t),1,f)!=1) return false;
        if(fread(&bf.nwords,sizeof(uint64_t),1,f)!=1) return false;
        if(fread(&bf.k_hashes,sizeof(uint32_t),1,f)!=1) return false;
        if(fread(&bf.block_bits,sizeof(uint64_t),1,f)!=1) return false;
        printf("[+] Reading %s (%.1f MB) ...\n", name, (double)(bf.nwords*8)/1e6); fflush(stdout);

        size_t bytes = (size_t)bf.nwords*sizeof(uint64_t);
        void *mem = mmap(NULL, bytes, PROT_READ|PROT_WRITE,
                          MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
        if(mem==MAP_FAILED) die("oom loading bloom (mmap)");
#ifdef MADV_HUGEPAGE
        madvise(mem, bytes, MADV_HUGEPAGE);
#endif
        bf.bits = (uint64_t*)mem;
        if(bf.nwords && fread(bf.bits,sizeof(uint64_t),(size_t)bf.nwords,f)!=(size_t)bf.nwords) return false;
        printf("[+] %s read (%.1fs)\n", name, now_seconds()-t0); fflush(stdout);
        return true;
    };
    if(!read_bloom(st.bloom1,"Bloom1")){ fclose(f); return false; }
    if(!read_bloom(st.bloom2,"Bloom2")){ fclose(f); return false; }
    if(!read_bloom(st.bloom3,"Bloom3")){ fclose(f); return false; }

    printf("[+] Reading exact MSB table (%" PRIu64 " slots) ...\n", st.baby_array_size); fflush(stdout);
    st.baby_table=(BabyEntry*)malloc((size_t)st.baby_array_size*sizeof(BabyEntry));
    if(fread(st.baby_table,sizeof(BabyEntry),(size_t)st.baby_array_size,f)!=(size_t)st.baby_array_size){ fclose(f); return false; }
    fclose(f);

    {
        FILE *sf = fopen("/proc/self/status","r");
        if(sf){
            char line[256];
            bool found_line=false;
            while(fgets(line,sizeof(line),sf)){
                if(strncmp(line,"VmHWM",5)==0 || strncmp(line,"VmRSS",5)==0){
                    printf("[diag] %s", line);
                    found_line=true;
                }
            }
            fclose(sf);
            if(!found_line) printf("[diag] /proc/self/status had no VmHWM/VmRSS line\n");
        }
        FILE *sr = fopen("/proc/self/smaps_rollup","r");
        if(sr){
            char line[256];
            bool found_hp=false;
            while(fgets(line,sizeof(line),sr)){
                if(strncmp(line,"AnonHugePages",13)==0){
                    printf("[diag] %s", line);
                    found_hp=true;
                }
            }
            fclose(sr);
            if(!found_hp) printf("[diag] smaps_rollup had no AnonHugePages line\n");
        } else {
            printf("[diag] /proc/self/smaps_rollup not available on this kernel\n");
        }
    }

    precompute_shifts(secp, st);
    printf("[+] Hierarchy loaded from %s\n", path);
    return true;
}

#ifdef GPU_ENABLED
static bool run_gpu_smoke_test(Secp256K1 &secp, HierState &st){
    char gpu_info[256];
    gpu_query_device_info(gpu_info, sizeof(gpu_info));
    printf("[gpu-smoke-test] GPU: %s\n", gpu_info);

    GpuBloom *gb1 = gpu_bloom_upload(st.bloom1.bits, st.bloom1.bit_count, st.bloom1.nwords,
                                      st.bloom1.k_hashes, st.bloom1.block_bits);
    if(!gb1){
        printf("[gpu-smoke-test] FAILED - could not upload Bloom1 (%.2f GB) to GPU\n",
               st.bloom1.nwords*8.0/1e9);
        return false;
    }
    printf("[gpu-smoke-test] Bloom1 uploaded: %.2f GB, block_bits=%" PRIu64 ", k_hashes=%u\n",
           st.bloom1.nwords*8.0/1e9, st.bloom1.block_bits, st.bloom1.k_hashes);

    int fails = 0;

    {
        Int k; k.SetInt64(12345);
        Point p = secp.ComputePublicKey(&k); p.Reduce();
        unsigned char xb[32]; p.x.Get32Bytes(xb);
        uint64_t fp = fingerprint_32(xb);
        int cpu_result = bloom_check(&st.bloom1, fp);
        int gpu_result = gpu_bloom_check_single(gb1, fp);
        bool ok = (cpu_result == gpu_result) && (cpu_result == 1);
        printf("[gpu-smoke-test] known-true (baby-step 12345): CPU=%d GPU=%d: %s\n",
               cpu_result, gpu_result, ok?"PASSED":"FAILED");
        if(!ok) fails++;
    }

    {
        uint64_t fp = 0xDEADBEEFCAFEF00DULL;
        int cpu_result = bloom_check(&st.bloom1, fp);
        int gpu_result = gpu_bloom_check_single(gb1, fp);
        bool ok = (cpu_result == gpu_result);
        printf("[gpu-smoke-test] fixed random fp: CPU=%d GPU=%d: %s%s\n",
               cpu_result, gpu_result, ok?"PASSED":"FAILED",
               (ok && cpu_result==1) ? " (rare Bloom false-positive on BOTH sides - not a bug by itself, but worth re-running with a different fp to confirm this isn't masking a real disagreement)" : "");
        if(!ok) fails++;
    }

    auto point_to_gpu = [](Point &p)->GpuECPoint{
        p.Reduce();
        unsigned char xb[32], yb[32];
        p.x.Get32Bytes(xb);
        p.y.Get32Bytes(yb);
        GpuECPoint g;
        for(int limb=0; limb<4; ++limb){
            uint64_t vx=0, vy=0;
            for(int b=0;b<8;++b){
                vx = (vx<<8) | xb[limb*8+b];
                vy = (vy<<8) | yb[limb*8+b];
            }

            g.x[3-limb] = vx; g.y[3-limb] = vy;
        }
        return g;
    };

    GpuBloom *gb2 = gpu_bloom_upload(st.bloom2.bits, st.bloom2.bit_count, st.bloom2.nwords,
                                      st.bloom2.k_hashes, st.bloom2.block_bits);
    if(!gb2){
        printf("[gpu-smoke-test] Step 2 FAILED - could not upload Bloom2\n");
        gpu_bloom_free(gb1);
        return false;
    }
    printf("[gpu-smoke-test] Bloom2 uploaded: %.2f MB\n", st.bloom2.nwords*8.0/1e6);

    GpuECPoint h_shift2[32];
    for(int i=0;i<32;++i) h_shift2[i] = point_to_gpu(st.shift2[i]);
    GpuShiftTable *shift2_gpu = gpu_shift_table_upload(h_shift2, 32);
    if(!shift2_gpu){
        printf("[gpu-smoke-test] Step 2 FAILED - could not upload shift2 table\n");
        gpu_bloom_free(gb1); gpu_bloom_free(gb2);
        return false;
    }

    Int baby_full_int; baby_full_int.SetInt64(602207762);
    Point R_point = secp.ComputePublicKey(&baby_full_int);
    GpuECPoint R_gpu = point_to_gpu(R_point);

    {
        printf("[gpu-smoke-test] DIAGNOSTIC: shift2[8].x (GetBase16)   = %s\n", st.shift2[8].x.GetBase16());
        printf("[gpu-smoke-test] DIAGNOSTIC: shift2[8].x (via point_to_gpu, limbs hex) = %016" PRIx64 "%016" PRIx64 "%016" PRIx64 "%016" PRIx64 "\n",
               h_shift2[8].x[3], h_shift2[8].x[2], h_shift2[8].x[1], h_shift2[8].x[0]);
        printf("[gpu-smoke-test] DIAGNOSTIC: R_point.x (GetBase16)     = %s\n", R_point.x.GetBase16());
        printf("[gpu-smoke-test] DIAGNOSTIC: R_point.x (via point_to_gpu, limbs hex)   = %016" PRIx64 "%016" PRIx64 "%016" PRIx64 "%016" PRIx64 "\n",
               R_gpu.x[3], R_gpu.x[2], R_gpu.x[1], R_gpu.x[0]);
    }

    {
        Int expected_val; expected_val.SetInt64(65336850);
        Point expected_point = secp.ComputePublicKey(&expected_val); expected_point.Reduce();
        unsigned char xb[32]; expected_point.x.Get32Bytes(xb);
        uint64_t expected_fp = fingerprint_32(xb);
        int cpu_bloom2_result = bloom_check(&st.bloom2, expected_fp);
        int gpu_bloom2_result = gpu_bloom_check_single(gb2, expected_fp);
        printf("[gpu-smoke-test] DIAGNOSTIC: expected fp=0x%016" PRIx64 ", CPU bloom2=%d, GPU bloom2=%d (direct check, bypassing i2-loop kernel)\n",
               expected_fp, cpu_bloom2_result, gpu_bloom2_result);

        Point R_minus_shift = secp.Add(R_point, st.shift2[8]);
        R_minus_shift.Reduce();
        bool decomposition_ok = R_minus_shift.x.IsEqual(&expected_point.x);
        printf("[gpu-smoke-test] DIAGNOSTIC: (R + shift2[8]) == 65336850*G on CPU? %s\n",
               decomposition_ok ? "YES" : "NO - decomposition itself is wrong, not a GPU bug");
    }

    {
        uint64_t results[32*4];
        gpu_i2loop_debug(shift2_gpu, &R_gpu, results);
        Int expected_val; expected_val.SetInt64(65336850);
        Point expected_point = secp.ComputePublicKey(&expected_val); expected_point.Reduce();
        unsigned char xb[32]; expected_point.x.Get32Bytes(xb);
        uint64_t exp_limbs[4];
        for(int limb=0;limb<4;++limb){
            uint64_t v=0;
            for(int b=0;b<8;++b) v=(v<<8)|xb[limb*8+b];
            exp_limbs[3-limb]=v;
        }
        printf("[gpu-smoke-test] DIAGNOSTIC: expected X (65336850*G, limbs) = %016" PRIx64 "%016" PRIx64 "%016" PRIx64 "%016" PRIx64 "\n",
               exp_limbs[3],exp_limbs[2],exp_limbs[1],exp_limbs[0]);
        for(int i2=0;i2<32;++i2){
            bool matches = (results[i2*4+0]==exp_limbs[0] && results[i2*4+1]==exp_limbs[1] &&
                             results[i2*4+2]==exp_limbs[2] && results[i2*4+3]==exp_limbs[3]);
            printf("[gpu-smoke-test] DIAGNOSTIC: i2=%2d GPU-computed X = %016" PRIx64 "%016" PRIx64 "%016" PRIx64 "%016" PRIx64 "%s\n",
                   i2, results[i2*4+3],results[i2*4+2],results[i2*4+1],results[i2*4+0],
                   matches ? "  <-- MATCHES EXPECTED" : "");
        }
    }

    int found_i2 = gpu_i2loop_check_single(shift2_gpu, gb2, &R_gpu);
    bool i2_ok = (found_i2 == 8);
    printf("[gpu-smoke-test] i2-loop on REAL hierarchy data: found i2=%d (expected 8): %s\n",
           found_i2, i2_ok?"PASSED":"FAILED");
    if(!i2_ok) fails++;

    GpuBloom *gb3 = gpu_bloom_upload(st.bloom3.bits, st.bloom3.bit_count, st.bloom3.nwords,
                                      st.bloom3.k_hashes, st.bloom3.block_bits);
    if(!gb3){
        printf("[gpu-smoke-test] Step 3 FAILED - could not upload Bloom3\n");
        gpu_bloom_free(gb1);
        return false;
    }
    printf("[gpu-smoke-test] Bloom3 uploaded: %.2f MB\n", st.bloom3.nwords*8.0/1e6);

    GpuECPoint h_shift3[32];
    for(int i=0;i<32;++i) h_shift3[i] = point_to_gpu(st.shift3[i]);
    GpuShiftTable *shift3_gpu = gpu_shift_table_upload(h_shift3, 32);
    if(!shift3_gpu){
        printf("[gpu-smoke-test] Step 3 FAILED - could not upload shift3 table\n");
        gpu_bloom_free(gb1); gpu_bloom_free(gb3);
        return false;
    }

    GpuBabyTable *baby_gpu = gpu_baby_table_upload(
        reinterpret_cast<const GpuBabyEntry*>(st.baby_table), st.baby_array_size, st.baby_K);
    if(!baby_gpu){
        printf("[gpu-smoke-test] Step 3 FAILED - could not upload baby_table\n");
        gpu_bloom_free(gb1); gpu_bloom_free(gb3); gpu_shift_table_free(shift3_gpu);
        return false;
    }
    printf("[gpu-smoke-test] baby_table uploaded: %" PRIu64 " entries (%.2f MB)\n",
           st.baby_array_size, st.baby_array_size*sizeof(GpuBabyEntry)/1e6);

    Int r2_val; r2_val.SetInt64(65336850);
    Point R2_point = secp.ComputePublicKey(&r2_val);
    GpuECPoint R2_gpu = point_to_gpu(R2_point);

    int found_i3 = -99;
    int found_idx = gpu_i3loop_and_exact_check_single(shift3_gpu, gb3, baby_gpu, &R2_gpu, &found_i3);
    bool i3_ok = (found_i3==31 && found_idx==325138);
    printf("[gpu-smoke-test] i3-loop+exact on REAL hierarchy data: found i3=%d idx=%d (expected i3=31 idx=325138): %s\n",
           found_i3, found_idx, i3_ok?"PASSED":"FAILED");
    if(!i3_ok) fails++;

    {
        const int n_batch = 200000;
        const int true_pos = n_batch/3;
        std::vector<GpuECPoint> batch(n_batch);
        for(int i=0;i<n_batch;++i){
            if(i==true_pos){
                batch[i] = R_gpu;
            } else {
                for(int j=0;j<4;++j){
                    batch[i].x[j] = 0x9e3779b97f4a7c15ULL * (uint64_t)(i*4+j+1) + 0xABCDEF12345ULL;
                    batch[i].y[j] = 0x9e3779b97f4a7c15ULL * (uint64_t)(i*4+j+9999) + 0xFEDCBA98765ULL;
                }
            }
        }
        std::vector<int> out_i2(n_batch, -1), out_i3(n_batch, -1), out_idx(n_batch, -1);
        double t0 = now_seconds();
        gpu_batch_hierarchical_search(gb1, shift2_gpu, gb2, shift3_gpu, gb3, baby_gpu,
                                       batch.data(), n_batch, out_i2.data(), out_i3.data(), out_idx.data());
        double elapsed = now_seconds()-t0;

        bool batch_ok = (out_i2[true_pos]==8 && out_i3[true_pos]==31 && out_idx[true_pos]==325138);
        int other_hits = 0;
        for(int i=0;i<n_batch;++i) if(i!=true_pos && out_idx[i]>=0) other_hits++;
        printf("[gpu-smoke-test] Step 4 batch (%d candidates, %.4fs): true match at %d found i2=%d i3=%d idx=%d: %s (%d other false-positive matches, expected occasionally)\n",
               n_batch, elapsed, true_pos, out_i2[true_pos], out_i3[true_pos], out_idx[true_pos],
               batch_ok?"PASSED":"FAILED", other_hits);
        if(!batch_ok) fails++;
    }

    {
        Int mi; mi.SetInt64((int64_t)st.M);
        Point giant_step_point = secp.ComputePublicKey(&mi); giant_step_point.Reduce();
        Point neg_giant_step_point = secp.Negation(giant_step_point); neg_giant_step_point.Reduce();
        GpuECPoint neg_giant_gpu = point_to_gpu(neg_giant_step_point);

        Int start_val; start_val.SetInt64(9192142354LL);
        Point start_point = secp.ComputePublicKey(&start_val);
        GpuECPoint cur = point_to_gpu(start_point);

        for(int step=0; step<4; ++step){
            gpu_batch_advance(&cur, 1, &neg_giant_gpu);
        }

        bool advance_ok = (cur.x[0]==R_gpu.x[0] && cur.x[1]==R_gpu.x[1] &&
                            cur.x[2]==R_gpu.x[2] && cur.x[3]==R_gpu.x[3]);
        printf("[gpu-smoke-test] Step 5 advance (round 15805 -4*M*G -> round 15809): %s\n",
               advance_ok?"PASSED":"FAILED");
        if(!advance_ok) fails++;

        const int adv_n = 300;
        std::vector<GpuECPoint> batch_pts(adv_n);
        for(int i=0;i<adv_n;++i) batch_pts[i] = point_to_gpu(start_point);
        for(int step=0; step<4; ++step){
            gpu_batch_advance(batch_pts.data(), adv_n, &neg_giant_gpu);
        }
        int batch_advance_fails=0;
        for(int i=0;i<adv_n;++i){
            if(!(batch_pts[i].x[0]==R_gpu.x[0] && batch_pts[i].x[1]==R_gpu.x[1] &&
                 batch_pts[i].x[2]==R_gpu.x[2] && batch_pts[i].x[3]==R_gpu.x[3])) batch_advance_fails++;
        }
        printf("[gpu-smoke-test] Step 5 advance at batch scale (%d points, tests padding path): %d/%d mismatched: %s\n",
               adv_n, batch_advance_fails, adv_n, batch_advance_fails==0?"PASSED":"FAILED");
        if(batch_advance_fails>0) fails++;
    }

    {
        Int a_val; a_val.SetInt64(33950171198994LL);
        Point leaf_pubkey = secp.ComputePublicKey(&a_val);

        Int mi; mi.SetInt64((int64_t)st.M);
        Point giant_step_point = secp.ComputePublicKey(&mi); giant_step_point.Reduce();
        GpuECPoint giant_step_gpu = point_to_gpu(giant_step_point);

        uint64_t start_i = 15812;
        Int start_offset; start_offset.SetInt64((int64_t)start_i);
        start_offset.Mult(&mi);
        Point start_p = secp.ComputePublicKey(&start_offset); start_p.Reduce();
        Point neg_start_p = secp.Negation(start_p); neg_start_p.Reduce();
        Point P_start = secp.Add(leaf_pubkey, neg_start_p); P_start.Reduce();
        GpuECPoint cur = point_to_gpu(P_start);

        const int max_rounds = 10;
        bool found = false;
        int found_round=-1, found_i2=-1, found_i3=-1, found_idx=-1;
        for(int r=0; r<max_rounds && !found; ++r){
            int oi2=-1, oi3=-1, oidx=-1;
            gpu_batch_hierarchical_search(gb1, shift2_gpu, gb2, shift3_gpu, gb3, baby_gpu,
                                           &cur, 1, &oi2, &oi3, &oidx);
            if(oidx>=0){
                found=true; found_round=r; found_i2=oi2; found_i3=oi3; found_idx=oidx;
                break;
            }
            gpu_batch_advance(&cur, 1, &giant_step_gpu);
        }

        uint64_t resolved_i = start_i - found_round;
        bool step6_ok = found && found_round==3 && resolved_i==15809 &&
                         found_i2==8 && found_i3==31 && found_idx==325138;
        printf("[gpu-smoke-test] Step 6 round loop: found=%s round=%d resolved_i=%" PRIu64
               " i2=%d i3=%d idx=%d (expected round=3, i=15809, i2=8, i3=31, idx=325138): %s\n",
               found?"yes":"no", found_round, resolved_i, found_i2, found_i3, found_idx,
               step6_ok?"PASSED":"FAILED");
        if(!step6_ok) fails++;

        if(step6_ok){

            uint64_t M2v = st.M2, M3v = st.M3;
            uint64_t reconstructed_A = resolved_i*st.M + (uint64_t)found_i2*M2v + (uint64_t)found_i3*M3v + (uint64_t)found_idx;
            printf("[gpu-smoke-test] Step 6 reconstructed A = %" PRIu64 " (expected 33950171198994): %s\n",
                   reconstructed_A, reconstructed_A==33950171198994ULL?"PASSED":"FAILED");
            if(reconstructed_A!=33950171198994ULL) fails++;
        }
    }

    gpu_shift_table_free(shift2_gpu);
    gpu_bloom_free(gb2);
    gpu_shift_table_free(shift3_gpu);
    gpu_bloom_free(gb3);
    gpu_baby_table_free(baby_gpu);

    gpu_bloom_free(gb1);

    if(fails==0) printf("[gpu-smoke-test] ALL CHECKS PASSED - CPU<->GPU Bloom1 bridge confirmed correct\n");
    else printf("[gpu-smoke-test] %d CHECK(S) FAILED - do not proceed with further GPU integration until fixed\n", fails);
    return fails==0;
}
#endif

struct BloomFilterDummy { std::vector<uint64_t> bits; uint64_t m; uint32_t k; };
struct LeafExact { unsigned char pub[PUBKEY_SIZE]; uint64_t b; };
struct PrecomputedTargets {
    BloomFilterDummy bloom;
    std::unordered_map<uint64_t, LeafExact> exact;
    uint64_t n_leaves = 0;
};
#define TGT_MAGIC "AMALIATGT"
#define TGT_MAGIC_SIZE 8
#define TGT_VERSION 2

static PrecomputedTargets* load_targets(const char *infile){
    FILE *f = fopen(infile,"rb"); if(!f) die_errno("open target table");
    char magic[TGT_MAGIC_SIZE];
    if(fread(magic,1,TGT_MAGIC_SIZE,f)!=TGT_MAGIC_SIZE){ fclose(f); die("read target magic"); }
    if(memcmp(magic,TGT_MAGIC,TGT_MAGIC_SIZE)!=0){ fclose(f); die("bad target magic"); }
    uint32_t version;
    if(fread(&version,sizeof(uint32_t),1,f)!=1){ fclose(f); die("read target version"); }
    if(version!=TGT_VERSION){ fclose(f); die("incompatible target version (needs v2)"); }
    PrecomputedTargets *tg = new PrecomputedTargets();
    if(fread(&tg->bloom.m,sizeof(uint64_t),1,f)!=1){ fclose(f); die("read m"); }
    if(fread(&tg->bloom.k,sizeof(uint32_t),1,f)!=1){ fclose(f); die("read k"); }
    if(fread(&tg->n_leaves,sizeof(uint64_t),1,f)!=1){ fclose(f); die("read n_leaves"); }
    uint64_t nw;
    if(fread(&nw,sizeof(uint64_t),1,f)!=1){ fclose(f); die("read nw"); }
    tg->bloom.bits.resize(nw);
    if(nw && fread(tg->bloom.bits.data(),sizeof(uint64_t),nw,f)!=nw){ fclose(f); die("read bits"); }
    tg->exact.reserve((size_t)tg->n_leaves*2);
    for(uint64_t i=0;i<tg->n_leaves;++i){
        uint64_t fp; LeafExact le;
        if(fread(&fp,sizeof(uint64_t),1,f)!=1){ fclose(f); die("read leaf fp"); }
        if(fread(le.pub,1,PUBKEY_SIZE,f)!=PUBKEY_SIZE){ fclose(f); die("read leaf pub"); }
        if(fread(&le.b,sizeof(uint64_t),1,f)!=1){ fclose(f); die("read leaf b"); }
        tg->exact[fp]=le;
    }
    fclose(f);
    printf("[+] %" PRIu64 " leaves loaded from %s\n", tg->n_leaves, infile);
    return tg;
}

#define EC_ORDER_HEX "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141"

struct TreeGenContext {
    Point R;
    Point E;
    Point negE;
    Int h;
    int steps;
};

static Point safe_scalar_mult(Secp256K1 &secp, Point &P, Int &scalar){
    int nbits = scalar.GetBitLength();
    if(nbits==0) die("safe_scalar_mult: zero scalar not supported here");
    Point R;
    bool R_set = false;
    for(int bit=nbits-1; bit>=0; --bit){
        if(R_set){ R = secp.Double(R); R.Reduce(); }
        if(scalar.GetBit(bit)){
            if(!R_set){ R = P; R_set = true; }
            else { R = secp.Add(R, P); R.Reduce(); }
        }
    }
    return R;
}

static void tree_gen_init(Secp256K1 &secp, TreeGenContext &ctx, Point &root, int steps){
    Int order; order.SetBase16((char*)EC_ORDER_HEX);
    Int two; two.SetInt64(2);
    Int h; h.SetInt64(1);
    for(int i=0;i<steps;++i){
        if(h.GetBit(0)) h.Add(&order);
        Int rem; h.Div(&two,&rem);
    }
    ctx.h = h;
    ctx.steps = steps;
    ctx.R = safe_scalar_mult(secp, root, h); ctx.R.Reduce();
    ctx.E = secp.ComputePublicKey(&h); ctx.E.Reduce();
    ctx.negE = secp.Negation(ctx.E); ctx.negE.Reduce();
}

static Point tree_gen_leaf_at(Secp256K1 &secp, TreeGenContext &ctx, Int &b){
    if(b.IsZero()){
        Point r = ctx.R; r.Reduce();
        return r;
    }
    Point bE = safe_scalar_mult(secp, ctx.E, b);
    Point negBE = secp.Negation(bE); negBE.Reduce();
    Point leaf = secp.Add(ctx.R, negBE); leaf.Reduce();
    return leaf;
}

static inline int child_decision(Point &Q, int level){
    (void)Q; (void)level;
    return -1;
}

static bool run_prune_self_test(Secp256K1 &secp, TreeGenContext &ctx,
                                              uint64_t b_true, int levels_to_check){
    printf("[prune-selftest] checking b_true=%" PRIu64 " (0b", b_true);
    for(int lv=levels_to_check-1; lv>=0; --lv) printf("%d", (int)((b_true>>lv)&1));
    printf(") against the rule for %d level(s)\n", levels_to_check);

    Point root = ctx.R; root.Reduce();
    for(int i=0;i<ctx.steps;++i){ root = secp.Double(root); root.Reduce(); }

    Int order; order.SetBase16((char*)EC_ORDER_HEX);
    Int two; two.SetInt64(2);
    Point Q = root;
    bool all_ok = true;
    for(int level=0; level<levels_to_check; ++level){
        int true_bit = (int)((b_true>>level)&1);
        int decision = child_decision(Q, level);
        if(decision==-1){
            printf("[prune-selftest] level %d: rule undecided (-1)\n", level);
        } else if(decision==true_bit){
            printf("[prune-selftest] level %d: rule keeps bit=%d, matches b_true's real bit\n",
                   level, decision);
        } else {
            printf("[prune-selftest] level %d: FAIL - rule would keep bit=%d but b_true's real bit "
                   "is %d\n",
                   level, decision, true_bit);
            all_ok = false;
        }

        Point Qn = Q;
        if(true_bit){
            Int one; one.SetInt64(1);
            Point G1 = secp.ComputePublicKey(&one); G1.Reduce();
            Point negG1 = secp.Negation(G1); negG1.Reduce();
            Qn = secp.Add(Q, negG1); Qn.Reduce();
        }
        Int hlvl; hlvl.SetInt64(1);
        if(hlvl.GetBit(0)) hlvl.Add(&order);
        Int rem; hlvl.Div(&two,&rem);
        Qn = safe_scalar_mult(secp, Qn, hlvl); Qn.Reduce();
        Q = Qn;
    }
    if(all_ok) printf("[prune-selftest] PASSED - rule does not discard b_true at any checked level\n");
    else { printf("[prune-selftest] FAILED\n"); exit(1); }
    return all_ok;
}

static inline bool has_n_repeat(uint64_t b, int lv, int n){
    for(int i=0; i+n<=lv; ++i){
        int first = (int)((b>>i)&1);
        bool all_same = true;
        for(int k=1;k<n;++k){ if((int)((b>>(i+k))&1)!=first){ all_same=false; break; } }
        if(all_same) return true;
    }
    return false;
}
static inline bool has_4_repeat(uint64_t b, int lv){ return has_n_repeat(b, lv, 4); }

static inline bool has_n_repeat_int(Int &b, int lv, int n){
    for(int i=0; i+n<=lv; ++i){
        int first = b.GetBit(i);
        bool all_same = true;
        for(int k=1;k<n;++k){ if(b.GetBit(i+k)!=first){ all_same=false; break; } }
        if(all_same) return true;
    }
    return false;
}

static inline bool block_fully_excluded(Int &block_offset, int lv, int n, int block_bits){
    for(int i=block_bits; i+n<=lv; ++i){
        int first = block_offset.GetBit(i);
        bool all_same = true;
        for(int k=1;k<n;++k){ if(block_offset.GetBit(i+k)!=first){ all_same=false; break; } }
        if(all_same) return true;
    }
    return false;
}

/* Saturating add (caps at UINT64_MAX instead of wrapping) - for
   display/--max-batches-limit counters that track how many blocks a
   pruning jump has skipped. The REAL search cursor is always a
   separate Int elsewhere (block_offset / cursor), unaffected by this;
   these counters are cosmetic/limit-only, but a single jump can be up
   to 2^60, and repeated large jumps at a low --prune-repeat-n can add
   up to more than 2^64 in total, silently wrapping a plain uint64_t
   counter back to a small value - confirmed as a real, reachable bug
   (a GPU dispatcher's own equivalent counter wrapped and repeated
   already-visited block numbers, before being replaced with an Int
   cursor entirely; this file's other uint64_t skip-counters share the
   same risk without needing the same full Int-cursor rewrite, since
   they're display-only here). */
static inline void sat_add_u64(uint64_t &acc, uint64_t add){
    if(add > UINT64_MAX - acc) acc = UINT64_MAX; else acc += add;
}

static inline uint64_t block_fully_excluded_jump(Int &block_offset, int lv, int n, int block_bits){
    int best_j = -1;
    for(int i=block_bits; i+n<=lv; ++i){
        int first = block_offset.GetBit(i);
        bool all_same = true;
        for(int k=1;k<n;++k){ if(block_offset.GetBit(i+k)!=first){ all_same=false; break; } }
        if(!all_same) continue;

        if(i>best_j) best_j = i;
    }
    if(best_j<block_bits) return 0;

    int aligned_j = best_j;
    for(int p=block_bits; p<best_j; ++p){
        if(block_offset.GetBit(p)!=0){ aligned_j = p; break; }
    }

    int exp = aligned_j-block_bits;
    if(exp>60) exp=60;
    uint64_t jump_blocks = (uint64_t)1 << exp;
    return jump_blocks;
}

struct MasterWorkUnit {
    Int offset;
    uint64_t block_size;
    double assigned_at;
};

static void run_master(const char *pubkey_hex, int tree_steps, int root_bits,
                        uint64_t block_size, bool restrict_upper_half,
                        int unsafe_prune, int prune_repeat_n,
                        int port, double timeout_seconds, const char *checkpoint_file,
                        const char *serve_hier_table){
    printf("[master] listening on port %d - tree_steps=%d root_bits=%d block_size=%" PRIu64
           " prune=%s(n=%d)\n", port, tree_steps, root_bits, block_size,
           unsafe_prune?"on":"off", prune_repeat_n);

    int lfd = net_listen(port);
    if(lfd<0){ printf("[master] FATAL: could not listen on port %d\n", port); return; }

    int block_bits = -1;
    if(unsafe_prune && block_size>0 && (block_size&(block_size-1))==0){
        block_bits = 0; uint64_t bs=block_size; while(bs>1){ bs>>=1; block_bits++; }
    }

    std::mutex state_mu;
    Int cursor; cursor.SetInt64(0);
    Int block_size_i; block_size_i.SetInt64((int64_t)block_size);
    uint64_t next_work_id = 1;
    std::unordered_map<uint64_t, MasterWorkUnit> in_flight;
    bool found = false;
    std::string found_K;
    uint64_t blocks_skipped = 0, blocks_assigned = 0, blocks_completed = 0;
    double last_status = now_seconds();
    double last_checkpoint = now_seconds();
    uint64_t blocks_since_checkpoint = 0;

    if(checkpoint_file){
        FILE *f = fopen(checkpoint_file, "r");
        if(f){
            char buf[256];
            if(fgets(buf, sizeof(buf), f)){
                size_t len = strlen(buf);
                while(len>0 && (buf[len-1]=='\n' || buf[len-1]=='\r')) buf[--len]=0;
                if(len>2 && buf[0]=='0' && buf[1]=='x'){
                    cursor.SetBase16(buf+2);
                    printf("[master] resumed from checkpoint: cursor=0x%s (%s)\n",
                           cursor.GetBase16(), checkpoint_file);
                } else {
                    printf("[master] checkpoint file exists but is malformed, starting from 0x0: %s\n",
                           checkpoint_file);
                }
            }
            fclose(f);
        } else {
            printf("[master] no checkpoint found (or --master-checkpoint not reused before) - "
                   "starting from 0x0, will write to %s\n", checkpoint_file);
        }
    }

    auto handle_connection = [&](int cfd){
        std::string line;
        if(!net_recv_line(cfd, line)){ close(cfd); return; }
        auto tok = net_split(line);
        if(tok.empty()){ close(cfd); return; }

        if(tok[0]=="GETFILE" && tok.size()>=2){

            const char *path = nullptr;
            if(tok[1]=="hier") path = serve_hier_table;
            if(!path){ net_send_line(cfd, "NOFILE"); close(cfd); return; }
            FILE *f = fopen(path, "rb");
            if(!f){ net_send_line(cfd, "NOFILE"); close(cfd); return; }
            fseek(f, 0, SEEK_END); long fsize = ftell(f); fseek(f, 0, SEEK_SET);
            if(fsize<0){ fclose(f); net_send_line(cfd, "NOFILE"); close(cfd); return; }
            char hdr[64]; snprintf(hdr,sizeof(hdr),"FILE %ld", fsize);
            net_send_line(cfd, hdr);
            printf("[master] serving %s (%ld bytes) to a worker\n", path, fsize);
            char chunk[1<<20];
            size_t total_sent = 0;
            while(!feof(f)){
                size_t n = fread(chunk, 1, sizeof(chunk), f);
                if(n==0) break;
                size_t off=0;
                while(off<n){
                    ssize_t s = send(cfd, chunk+off, n-off, 0);
                    if(s<=0){ fclose(f); close(cfd); printf("[master] file transfer interrupted\n"); return; }
                    off += (size_t)s;
                }
                total_sent += n;
            }
            fclose(f); close(cfd);
            printf("[master] finished serving %s (%zu bytes sent)\n", path, total_sent);
            return;
        }

        std::lock_guard<std::mutex> lock(state_mu);
        double now = now_seconds();

        if(tok[0]=="WORK"){
            if(found){
                net_send_line(cfd, "STOP");
                close(cfd); return;
            }

            uint64_t reclaim_id = 0;
            for(auto &kv : in_flight){
                if(now - kv.second.assigned_at > timeout_seconds){ reclaim_id = kv.first; break; }
            }
            uint64_t wid; Int offset;
            if(reclaim_id){
                MasterWorkUnit &w = in_flight[reclaim_id];
                wid = next_work_id++;
                offset = w.offset;
                in_flight.erase(reclaim_id);
                printf("[master] reclaiming timed-out work as #%" PRIu64 " (offset=0x%s)\n",
                       wid, offset.GetBase16());
            } else {
                if(block_bits>=0){
                    uint64_t jump = block_fully_excluded_jump(cursor, tree_steps, prune_repeat_n, block_bits);
                    while(jump>0){
                        sat_add_u64(blocks_skipped, jump);
                        Int jump_amt; jump_amt.Set(&block_size_i); jump_amt.Mult(jump);
                        cursor.Add(&jump_amt);
                        jump = block_fully_excluded_jump(cursor, tree_steps, prune_repeat_n, block_bits);
                    }
                }
                wid = next_work_id++;
                offset = cursor;
                cursor.Add(&block_size_i);
                blocks_assigned++;
                blocks_since_checkpoint++;
            }
            MasterWorkUnit w; w.offset=offset; w.block_size=block_size; w.assigned_at=now;
            in_flight[wid] = w;
            char buf[256];
            snprintf(buf,sizeof(buf),"WORK %" PRIu64 " 0x%s %" PRIu64 " %d %d",
                     wid, offset.GetBase16(), block_size,
                     unsafe_prune, prune_repeat_n);
            net_send_line(cfd, buf);
        } else if(tok[0]=="RESULT" && tok.size()>=3){
            uint64_t wid = strtoull(tok[1].c_str(), NULL, 10);
            if(tok[2]=="FOUND" && tok.size()>=4){
                found = true; found_K = tok[3];
                printf("[master] *** FOUND *** K=%s (reported by worker, work #%" PRIu64 ")\n",
                       found_K.c_str(), wid);
            } else {
                in_flight.erase(wid);
                blocks_completed++;
            }
            net_send_line(cfd, "OK");
        } else {
            net_send_line(cfd, "ERR unknown command");
        }
        close(cfd);

        if(now - last_status >= 5.0){
            printf("[master] status: %" PRIu64 " assigned, %" PRIu64 " completed, %" PRIu64
                   " skipped (fully-excluded), %zu in-flight, cursor=0x%s\n",
                   blocks_assigned, blocks_completed, blocks_skipped, in_flight.size(),
                   cursor.GetBase16());
            last_status = now;
        }
        if(checkpoint_file && (blocks_since_checkpoint>=20 || now-last_checkpoint>=10.0) && blocks_since_checkpoint>0){
            FILE *cf = fopen(checkpoint_file, "w");
            if(cf){
                fprintf(cf, "0x%s\n", cursor.GetBase16());
                fclose(cf);
            } else {
                printf("[master] WARNING: could not write checkpoint to %s\n", checkpoint_file);
            }
            blocks_since_checkpoint = 0;
            last_checkpoint = now;
        }
        if(found){
            printf("[master] stopping - answer found (K=%s). Remaining connections will be told to stop.\n",
                   found_K.c_str());

        }
    };

    while(true){
        struct sockaddr_in peer; socklen_t peerlen=sizeof(peer);
        int cfd = accept(lfd, (struct sockaddr*)&peer, &peerlen);
        if(cfd<0) continue;

        std::thread(handle_connection, cfd).detach();
    }
}

static void run_repeat_test(uint64_t b_true, int levels, int repeat_n){
    bool b_true_excluded = has_n_repeat(b_true, levels, repeat_n);
    printf("[repeat-test] repeat_n=%d, b_true=%" PRIu64 " (0b", repeat_n, b_true);
    for(int lv=levels-1; lv>=0; --lv) printf("%d", (int)((b_true>>lv)&1));
    printf(") would be %s by the rule\n", b_true_excluded?"EXCLUDED":"kept");

    if(levels<=24){
        uint64_t total = (uint64_t)1<<levels;
        uint64_t excluded = 0;
        for(uint64_t b=0; b<total; ++b) if(has_n_repeat(b, levels, repeat_n)) ++excluded;
        printf("[repeat-test] exhaustive scan of all %" PRIu64 " possible b values (levels=%d, repeat_n=%d):\n",
               total, levels, repeat_n);
        printf("[repeat-test]   %" PRIu64 " (%.1f%%) would be excluded by this rule\n",
               excluded, 100.0*(double)excluded/(double)total);
    } else {
        printf("[repeat-test] levels=%d too large for an exhaustive scan here - the exclusion\n"
               "[repeat-test]   rate does not depend meaningfully on level count once it's larger\n"
               "[repeat-test]   than a handful of windows though - see the smaller-level result.\n", levels);
    }
}

static inline void point_to_leaf(Point &leaf, uint64_t b_low64, LeafExact &out){
    unsigned char xb[32]; leaf.x.Get32Bytes(xb);
    out.pub[0] = leaf.y.IsEven() ? 0x02 : 0x03;
    memcpy(out.pub+1, xb, 32);
    out.b = b_low64;
}

static void gen_random_leaf_batch(Secp256K1 &secp, TreeGenContext &ctx, uint64_t count,
                                  std::vector<LeafExact> &out, std::vector<Int> &b_full_out){
    out.resize(count);
    b_full_out.resize(count);
    for(uint64_t i=0;i<count;++i){
        Int b;
        if(g_unsafe_prune){

            do { b.Rand(ctx.steps); } while(has_n_repeat_int(b, ctx.steps, g_prune_repeat_n));
        } else {
            b.Rand(ctx.steps);
        }
        b_full_out[i] = b;
        Point leaf = tree_gen_leaf_at(secp, ctx, b);
        point_to_leaf(leaf, b.bits64[0], out[i]);
    }
}

static void gen_sequential_leaf_batch(Secp256K1 &secp, TreeGenContext &ctx, uint64_t count,
                                      int threads, std::vector<LeafExact> &out,
                                      Int *global_offset=nullptr){
    int T = threads; if(T<1) T=1;
    uint64_t per = (count+(uint64_t)T-1)/(uint64_t)T;
    bool prune = g_unsafe_prune!=0;
    if(!prune){
        out.resize(count);
        #pragma omp parallel num_threads(T)
        {
            int tid = omp_get_thread_num();
            uint64_t k0 = (uint64_t)tid*per;
            uint64_t k1 = k0+per; if(k1>count) k1=count;
            if(k0<k1){
                Point cur;
                if(k0==0){ cur = ctx.R; cur.Reduce(); }
                else {
                    Int k0i; k0i.SetInt64((int64_t)k0);
                    cur = tree_gen_leaf_at(secp, ctx, k0i);
                }
                for(uint64_t k=k0;k<k1;++k){
                    cur.Reduce();
                    point_to_leaf(cur, k, out[k]);
                    cur = secp.Add(cur, ctx.negE); cur.Reduce();
                }
            }
        }
        return;
    }

    std::vector<std::vector<LeafExact>> local(T);
    #pragma omp parallel num_threads(T)
    {
        int tid = omp_get_thread_num();
        uint64_t k0 = (uint64_t)tid*per;
        uint64_t k1 = k0+per; if(k1>count) k1=count;
        if(k0<k1){
            Point cur;
            if(k0==0){ cur = ctx.R; cur.Reduce(); }
            else {
                Int k0i; k0i.SetInt64((int64_t)k0);
                cur = tree_gen_leaf_at(secp, ctx, k0i);
            }

            Int full_b;
            if(global_offset){ full_b.Set(global_offset); Int k0i2; k0i2.SetInt64((int64_t)k0); full_b.Add(&k0i2); }
            else full_b.SetInt64((int64_t)k0);
            Int one; one.SetInt64(1);
            for(uint64_t k=k0;k<k1;++k){
                cur.Reduce();
                if(!has_n_repeat_int(full_b, ctx.steps, g_prune_repeat_n)){
                    LeafExact le; point_to_leaf(cur, k, le);
                    local[tid].push_back(le);
                }
                cur = secp.Add(cur, ctx.negE); cur.Reduce();
                full_b.Add(&one);
            }
        }
    }
    out.clear();
    for(int t=0;t<T;++t) out.insert(out.end(), local[t].begin(), local[t].end());
}

#define PRUNED_MAGIC "AMALIAPRN"
#define PRUNED_MAGIC_SIZE 9
#define PRUNED_VERSION 1

static void run_gen_pruned_leaves(Secp256K1 &secp, TreeGenContext &ctx, uint64_t count,
                                   int threads, bool random_mode, uint64_t start_b,
                                   const char *output_file, int preview_n){
    printf("[gen-pruned-leaves] tree_steps=%d repeat_n=%d mode=%s requested_count=%" PRIu64 "\n",
           ctx.steps, g_prune_repeat_n, random_mode?"random":"sequential", count);

    std::vector<LeafExact> leaves;
    double t0 = now_seconds();
    if(random_mode){
        std::vector<Int> b_full;
        gen_random_leaf_batch(secp, ctx, count, leaves, b_full);
    } else {
        Int sbi; sbi.SetInt64((int64_t)start_b);
        if(start_b>0){
            Point shifted = tree_gen_leaf_at(secp, ctx, sbi); shifted.Reduce();
            ctx.R = shifted;
        }
        gen_sequential_leaf_batch(secp, ctx, count, threads, leaves, &sbi);
        if(start_b>0) for(auto &le : leaves) le.b += start_b;
    }
    double el = now_seconds()-t0;

    uint64_t survived = leaves.size();
    printf("[gen-pruned-leaves] generated in %.2fs: %" PRIu64 "/%" PRIu64
           " survived (%.1f%% kept, %.1f%% pruned)\n",
           el, survived, count, 100.0*(double)survived/(double)count,
           100.0*(1.0-(double)survived/(double)count));

    int shown = preview_n < (int)leaves.size() ? preview_n : (int)leaves.size();
    if(shown>0){
        printf("[gen-pruned-leaves] preview (first %d surviving leaves):\n", shown);
        for(int i=0;i<shown;++i){
            printf("  b=%-10" PRIu64 " pub=", leaves[i].b);
            for(int j=0;j<PUBKEY_SIZE;++j) printf("%02x", leaves[i].pub[j]);
            printf("\n");
        }
    }

    if(output_file){
        FILE *f = fopen(output_file, "wb");
        if(!f) die("could not open --output-leaves file for writing");
        fwrite(PRUNED_MAGIC, 1, PRUNED_MAGIC_SIZE, f);
        uint32_t version = PRUNED_VERSION; fwrite(&version, sizeof(uint32_t), 1, f);
        uint32_t steps32 = (uint32_t)ctx.steps; fwrite(&steps32, sizeof(uint32_t), 1, f);
        uint32_t rn32 = (uint32_t)g_prune_repeat_n; fwrite(&rn32, sizeof(uint32_t), 1, f);
        fwrite(&survived, sizeof(uint64_t), 1, f);
        for(auto &le : leaves){
            fwrite(&le.b, sizeof(uint64_t), 1, f);
            fwrite(le.pub, 1, PUBKEY_SIZE, f);
        }
        fclose(f);
        printf("[gen-pruned-leaves] wrote %" PRIu64 " leaves to %s (custom format, see source "
               "comment above run_gen_pruned_leaves for layout)\n", survived, output_file);
    }
}

static int64_t g_debug_b = -1;

static bool run_hier_search(Secp256K1 &secp, HierState &st, PrecomputedTargets &tg,
                            int tree_steps, int a_bits, int threads, int batch_size,
                            bool restrict_upper_half, const char *dump_round_stats_path,
                            uint64_t region_lo=UINT64_MAX, uint64_t region_hi=UINT64_MAX,
                            std::atomic<bool> *external_stop=nullptr,
                            uint64_t *out_A=nullptr, uint64_t *out_b=nullptr,
                            uint64_t max_rounds_cap=UINT64_MAX, bool quiet=false,
                            std::atomic<uint64_t> *out_round_progress=nullptr,
                            uint64_t *out_total_rounds=nullptr){

    uint64_t R = (uint64_t)1<<a_bits;
    uint64_t giant_count = (R + st.M - 1)/st.M;

    uint64_t search_start = restrict_upper_half ? giant_count/2 : 0;
    if(region_lo != UINT64_MAX) search_start = region_lo;
    if(region_hi != UINT64_MAX) giant_count = region_hi;
    uint64_t restricted_count = giant_count - search_start;
    uint64_t half_giant = restricted_count / 2;
    if(out_total_rounds) *out_total_rounds = half_giant;

    std::vector<LeafExact> leaves;
    leaves.reserve(tg.exact.size());
    for(auto &kv: tg.exact) leaves.push_back(kv.second);
    uint64_t nleaf = leaves.size();

    Int Mi; Mi.SetInt64((int64_t)st.M);
    Point giant_step = secp.ComputePublicKey(&Mi); giant_step.Reduce();
    Point neg_giant_step = secp.Negation(giant_step); neg_giant_step.Reduce();

    if(!quiet){
        printf("[+] Bidirectional search (Meet-in-the-Middle): %" PRIu64 " leaves, max rounds per front = %" PRIu64 "\n", nleaf, half_giant);
        if(restrict_upper_half){
            printf("[+] Upper-half restriction ACTIVE: search_start=%" PRIu64 "/%" PRIu64
                   " (skipping the lower half - guaranteed impossible for a puzzle-N target)\n",
                   search_start, giant_count);
        }
        printf("[+] expected total cost reduced by half ~ %.3e curve operations\n", (double)nleaf*(double)half_giant);
    }

    std::atomic<int> found{0};
    uint64_t result_p=0, result_i=0, result_b=0;
    int Wc = threads; if(Wc<1) Wc=1;
    (void)batch_size;
    stat_bloom1_pass.store(0); stat_i2_adds.store(0); stat_bloom2_pass.store(0);
    stat_i3_adds.store(0); stat_bloom3_pass.store(0); stat_exact_lookups.store(0);
    stat_try_candidate.store(0);
    time_bloom1_ns.store(0); time_i2_build_ns.store(0); time_i2_batchinv_ns.store(0);
    time_i2_normalize_bloom2_ns.store(0); time_i3_ns.store(0);

    {
        double tw0 = now_seconds();
        auto warm_bloom = [](BloomFilter &bf, int T){
            if(!bf.bits || bf.nwords==0) return;
            uint64_t page_words = 4096/sizeof(uint64_t);
            uint64_t n_pages = (bf.nwords + page_words - 1)/page_words;
            uint64_t sink = 0;
            #pragma omp parallel for num_threads(T) reduction(+:sink) schedule(static)
            for(uint64_t p=0; p<n_pages; ++p){
                uint64_t idx = p*page_words;
                if(idx < bf.nwords) sink += bf.bits[idx];
            }
            (void)sink;
        };
        warm_bloom(st.bloom1, Wc);
        warm_bloom(st.bloom2, Wc);
        warm_bloom(st.bloom3, Wc);
        printf("[+] Pre-warmed Bloom1/2/3 memory in %.2fs (moved out of search time)\n",
               now_seconds()-tw0);
    }

    double t_start = now_seconds();

    FILE *dump_f = NULL;
    uint64_t dump_prev_bloom1_pass=0, dump_prev_i2_adds=0, dump_prev_bloom2_pass=0,
             dump_prev_i3_adds=0, dump_prev_bloom3_pass=0;
    uint64_t dump_prev_t_bloom1=0, dump_prev_t_i2build=0, dump_prev_t_i2inv=0,
             dump_prev_t_i2norm=0, dump_prev_t_i3=0;
    if(dump_round_stats_path){
        dump_f = fopen(dump_round_stats_path, "w");
        if(!dump_f){
            fprintf(stderr, "[!] WARNING: could not open --dump-round-stats file '%s', continuing without it\n",
                    dump_round_stats_path);
        } else {
            fprintf(dump_f, "round,elapsed_s,d_bloom1_pass,d_i2_adds,d_bloom2_pass,d_i3_adds,d_bloom3_pass,"
                            "d_time_bloom1_ns,d_time_i2build_ns,d_time_i2inv_ns,d_time_i2norm_ns,d_time_i3_ns\n");
        }
    }

    #pragma omp parallel num_threads(Wc)
    {
        int tid = omp_get_thread_num();
#ifdef USE_NUMA
        if(g_numa_aware && numa_available()>=0){
            int n_nodes = numa_num_configured_nodes();
            if(n_nodes>1){
                numa_run_on_node(tid % n_nodes);
            }
        }
#endif

        int lower_threads = Wc/2;
        int upper_threads = Wc - Wc/2;
        bool single_front_mode = false;
        if(Wc==1){

            lower_threads = 1;
            upper_threads = 0;
            single_front_mode = true;
        } else if(lower_threads==0){
            lower_threads = 1;
            upper_threads = Wc-1;
        }

        bool is_upper_front = single_front_mode ? false : (tid >= lower_threads);
        int active_threads = is_upper_front ? upper_threads : lower_threads;
        int my_tid = is_upper_front ? (tid - lower_threads) : tid;

        uint64_t my_half_giant = single_front_mode ? (giant_count-1) : (search_start + half_giant);

        uint64_t per_thread = (nleaf + (uint64_t)active_threads - 1)/(uint64_t)active_threads;
        uint64_t start_li = (uint64_t)my_tid*per_thread;
        uint64_t end_li = start_li+per_thread; if(end_li>nleaf) end_li=nleaf;
        uint64_t chunk = end_li-start_li;
        int B = (int)chunk;

        if(B > 0){
            Point *P = new Point[B];
            Int *ax=new Int[B], *ay=new Int[B];
            Int *zx=new Int[B], *zy=new Int[B], *zz=new Int[B], *zinv=new Int[B], *pref=new Int[B];
            uint64_t *cur_i = new uint64_t[B];
            bool *active = new bool[B];

            for(int l=0;l<B;++l){
                Point pt;
                if(pub_to_point(secp, leaves[start_li+l].pub, pt)){
                    active[l]=true;
                    if(!is_upper_front) {
                        cur_i[l] = search_start;
                        if(search_start==0){
                            P[l] = pt;
                        } else {

                            Int start_offset; start_offset.SetInt64((int64_t)search_start);
                            start_offset.Mult(&Mi);
                            Point start_p = secp.ComputePublicKey(&start_offset); start_p.Reduce();
                            Point neg_start = secp.Negation(start_p); neg_start.Reduce();
                            P[l] = secp.Add(pt, neg_start); P[l].Reduce();
                        }
                    } else {
                        cur_i[l] = giant_count - 1;
                        Int top_offset; top_offset.SetInt64((int64_t)(giant_count - 1));
                        top_offset.Mult(&Mi);
                        Point top_p = secp.ComputePublicKey(&top_offset); top_p.Reduce();
                        Point neg_top = secp.Negation(top_p); neg_top.Reduce();
                        P[l] = secp.Add(pt, neg_top); P[l].Reduce();
                    }
                    ax[l]=P[l].x; ay[l]=P[l].y;
                } else {
                    active[l]=false;
                }
            }

            bool any_active=true;
            uint64_t round_num=0;
            uint64_t *fp0_arr = new uint64_t[B];
            while(any_active && !found.load() && !(external_stop && external_stop->load(std::memory_order_relaxed))
                  && round_num<max_rounds_cap){
                any_active=false;

                int l=0;
                for(; l+3<B; l+=4){
                    bool a0=active[l], a1=active[l+1], a2=active[l+2], a3=active[l+3];
                    if(a0){ Point R0; R0.x=ax[l]; R0.y=ay[l]; R0.z.SetInt32(1);
                            fp0_arr[l]=point_fingerprint(R0); bloom_prefetch(&st.bloom1,fp0_arr[l]); }
                    if(a1){ Point R1; R1.x=ax[l+1]; R1.y=ay[l+1]; R1.z.SetInt32(1);
                            fp0_arr[l+1]=point_fingerprint(R1); bloom_prefetch(&st.bloom1,fp0_arr[l+1]); }
                    if(a2){ Point R2; R2.x=ax[l+2]; R2.y=ay[l+2]; R2.z.SetInt32(1);
                            fp0_arr[l+2]=point_fingerprint(R2); bloom_prefetch(&st.bloom1,fp0_arr[l+2]); }
                    if(a3){ Point R3; R3.x=ax[l+3]; R3.y=ay[l+3]; R3.z.SetInt32(1);
                            fp0_arr[l+3]=point_fingerprint(R3); bloom_prefetch(&st.bloom1,fp0_arr[l+3]); }
                }
                for(; l<B; ++l){
                    if(!active[l]) continue;
                    Point Rpoint; Rpoint.x=ax[l]; Rpoint.y=ay[l]; Rpoint.z.SetInt32(1);
                    fp0_arr[l] = point_fingerprint(Rpoint);
                    bloom_prefetch(&st.bloom1, fp0_arr[l]);
                }

                for(int l=0;l<B;++l){
                    if(!active[l]) continue;
                    any_active=true;
                    Point Rpoint; Rpoint.x=ax[l]; Rpoint.y=ay[l]; Rpoint.z.SetInt32(1);
                    uint64_t p;
                    bool hit = st.radix4_enabled
                             ? (hierarchical_lookup_radix4(secp, st, Rpoint, fp0_arr[l], &p)!=0)
                             : hierarchical_lookup(secp, st, Rpoint, fp0_arr[l], &p);
                    uint64_t li = start_li+l;
                    if(g_debug_b>=0 && leaves[li].b==(uint64_t)g_debug_b){
                        #pragma omp critical(hier_debug)
                        { printf("[DEBUG] leaf b=%" PRIu64 " tested: lane=%d cur_i=%" PRIu64
                                 " hit=%d\n", leaves[li].b, l, cur_i[l], hit?1:0); fflush(stdout); }
                    }
                    if(hit){
                        #pragma omp critical(hier_result)
                        {
                            if(!found.load()){
                                found.store(1);
                                result_p=p; result_i=cur_i[l]; result_b=leaves[li].b;
                                if(external_stop) external_stop->store(true, std::memory_order_relaxed);
                            }
                        }
                        active[l]=false;
                        continue;
                    }

                    if(!is_upper_front) {
                        cur_i[l]++;
                        if(cur_i[l] > my_half_giant) active[l]=false;
                    } else {
                        if(cur_i[l] <= my_half_giant) {
                            active[l]=false;
                        } else {
                            cur_i[l]--;
                        }
                    }
                }
                if(!any_active || found.load() || (external_stop && external_stop->load(std::memory_order_relaxed))
                   || round_num>=max_rounds_cap) break;

                round_num++;
                if(tid==0 && (round_num & 0x3F)==0){
                    double el = now_seconds() - t_start;

                    if(out_round_progress) out_round_progress->store(round_num, std::memory_order_relaxed);
                    if(!quiet){
                        #pragma omp critical(hier_progress)
                        { printf("  [bidir] round %" PRIu64 "/%" PRIu64 ", %.1fs\r", round_num, half_giant, el);
                          fflush(stdout); }
                    }
                    if(dump_f){
                        uint64_t cur_bloom1_pass = stat_bloom1_pass.load();
                        uint64_t cur_i2_adds = stat_i2_adds.load();
                        uint64_t cur_bloom2_pass = stat_bloom2_pass.load();
                        uint64_t cur_i3_adds = stat_i3_adds.load();
                        uint64_t cur_bloom3_pass = stat_bloom3_pass.load();
                        uint64_t cur_t_bloom1 = time_bloom1_ns.load();
                        uint64_t cur_t_i2build = time_i2_build_ns.load();
                        uint64_t cur_t_i2inv = time_i2_batchinv_ns.load();
                        uint64_t cur_t_i2norm = time_i2_normalize_bloom2_ns.load();
                        uint64_t cur_t_i3 = time_i3_ns.load();
                        #pragma omp critical(hier_dump)
                        {
                            fprintf(dump_f, "%" PRIu64 ",%.3f,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64
                                            ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 "\n",
                                    round_num, el,
                                    cur_bloom1_pass-dump_prev_bloom1_pass, cur_i2_adds-dump_prev_i2_adds,
                                    cur_bloom2_pass-dump_prev_bloom2_pass, cur_i3_adds-dump_prev_i3_adds,
                                    cur_bloom3_pass-dump_prev_bloom3_pass,
                                    cur_t_bloom1-dump_prev_t_bloom1, cur_t_i2build-dump_prev_t_i2build,
                                    cur_t_i2inv-dump_prev_t_i2inv, cur_t_i2norm-dump_prev_t_i2norm,
                                    cur_t_i3-dump_prev_t_i3);
                            dump_prev_bloom1_pass=cur_bloom1_pass; dump_prev_i2_adds=cur_i2_adds;
                            dump_prev_bloom2_pass=cur_bloom2_pass; dump_prev_i3_adds=cur_i3_adds;
                            dump_prev_bloom3_pass=cur_bloom3_pass;
                            dump_prev_t_bloom1=cur_t_bloom1; dump_prev_t_i2build=cur_t_i2build;
                            dump_prev_t_i2inv=cur_t_i2inv; dump_prev_t_i2norm=cur_t_i2norm; dump_prev_t_i3=cur_t_i3;
                        }
                    }
                }

                for(int l=0;l<B;++l){
                    if(!active[l]) continue;
                    P[l] = add_mixed_z2eq1(P[l], is_upper_front ? giant_step : neg_giant_step);
                    zx[l]=P[l].x; zy[l]=P[l].y; zz[l]=P[l].z;
                }
                batch_invert_noalloc(zinv, zz, pref, (uint64_t)B);
                for(int l=0;l<B;++l){
                    if(!active[l]) continue;
                    ax[l]=zx[l]; ax[l].ModMul(&zinv[l]);
                    ay[l]=zy[l]; ay[l].ModMul(&zinv[l]);
                }
            }

            delete[] P; delete[] ax; delete[] ay;
            delete[] zx; delete[] zy; delete[] zz; delete[] zinv; delete[] pref;
            delete[] cur_i; delete[] active; delete[] fp0_arr;
        }
    }
    printf("\n");
    double elapsed = now_seconds() - t_start;

    if(found.load()){
        Int A; A.SetInt64((int64_t)result_i); A.Mult(&Mi);
        Int pI; pI.SetInt64((int64_t)result_p); A.Add(&pI);
        Int two; two.SetInt64(2);
        Int pow2steps; pow2steps.SetInt64(1);
        for(int i=0;i<tree_steps;++i) pow2steps.Mult(&two);
        Int K; K.Set(&A); K.Mult(&pow2steps);
        Int bI; bI.SetInt64((int64_t)result_b);
        K.Add(&bI);
        if(!quiet){
            printf("\n[+] FOUND\n");
            printf("    b = %" PRIu64 "\n", result_b);
            printf("    A = %s\n", A.GetBase16());
            printf("    K = A*2^%d + b = %s\n", tree_steps, K.GetBase16());
        }
        if(out_A) *out_A = result_i*st.M + result_p;
        if(out_b) *out_b = result_b;
    } else if(!quiet){
        printf("\n[!] NOT FOUND in any of the %" PRIu64 " leaves.\n", nleaf);
    }
    if(!quiet){
        printf("\n[+] Real ECC work breakdown:\n");
        printf("    Bloom1 pass (enter i2-loop):       %" PRIu64 "\n", stat_bloom1_pass.load());
        printf("    Add() in i2-loop (non-degen.):     %" PRIu64 "\n", stat_i2_adds.load());
        printf("    Bloom2 pass (enter i3-loop):       %" PRIu64 "\n", stat_bloom2_pass.load());
        printf("    Add() in i3-loop (non-degen.):     %" PRIu64 "\n", stat_i3_adds.load());
        printf("    Bloom3 pass (reach exact table):   %" PRIu64 "\n", stat_bloom3_pass.load());
        printf("    Exact table lookups:               %" PRIu64 "\n", stat_exact_lookups.load());
        printf("    try_candidate (ComputePublicKey):  %" PRIu64 "\n", stat_try_candidate.load());
    }
    {
        double t_bloom1 = time_bloom1_ns.load()/1e9;
        double t_i2build = time_i2_build_ns.load()/1e9;
        double t_i2inv = time_i2_batchinv_ns.load()/1e9;
        double t_i2norm = time_i2_normalize_bloom2_ns.load()/1e9;
        double t_i3 = time_i3_ns.load()/1e9;
        double t_soma = t_bloom1+t_i2build+t_i2inv+t_i2norm+t_i3;
        if(!quiet){
        printf("\n[+] Accumulated time per block (sum across all threads, not wall-clock time):\n");
        printf("    Bloom1 check:                %8.2fs (%5.1f%%)\n", t_bloom1, t_soma>0?100.0*t_bloom1/t_soma:0);
        printf("    i2 build (projective Add):   %8.2fs (%5.1f%%)\n", t_i2build, t_soma>0?100.0*t_i2build/t_soma:0);
        printf("    i2 batch_invert_noalloc:     %8.2fs (%5.1f%%)\n", t_i2inv, t_soma>0?100.0*t_i2inv/t_soma:0);
        printf("    i2 normalize + Bloom2:       %8.2fs (%5.1f%%)\n", t_i2norm, t_soma>0?100.0*t_i2norm/t_soma:0);
        printf("    i3 subloop (whole):          %8.2fs (%5.1f%%)\n", t_i3, t_soma>0?100.0*t_i3/t_soma:0);
        printf("    Sum of measured blocks:      %8.2fs\n", t_soma);
        }
    }
    if(!quiet) printf("[+] search time: %.2fs\n", elapsed);
    if(dump_f) fclose(dump_f);
    return found.load()!=0;
}

#ifdef GPU_ENABLED

static GpuECPoint point_to_gpu_point(Point &p){
    p.Reduce();
    unsigned char xb[32], yb[32];
    p.x.Get32Bytes(xb);
    p.y.Get32Bytes(yb);
    GpuECPoint g;
    for(int limb=0; limb<4; ++limb){
        uint64_t vx=0, vy=0;
        for(int b=0;b<8;++b){
            vx = (vx<<8) | xb[limb*8+b];
            vy = (vy<<8) | yb[limb*8+b];
        }
        g.x[3-limb] = vx; g.y[3-limb] = vy;
    }
    return g;
}

static bool run_hier_search_gpu(Secp256K1 &secp, HierState &st, GpuHierarchy *hgpu,
                                 PrecomputedTargets &tg, int tree_steps, int a_bits,
                                 bool restrict_upper_half, uint64_t *out_A, uint64_t *out_b){

    int m_bits = 0; { uint64_t m=st.M; while(m>1){ m>>=1; m_bits++; } }
    uint64_t giant_count = (uint64_t)1 << (a_bits - m_bits);
    uint64_t search_start = restrict_upper_half ? giant_count/2 : 0;

    uint64_t restricted_count = giant_count - search_start;
    uint64_t half_giant = restricted_count / 2;

    std::vector<LeafExact> leaves;
    leaves.reserve(tg.exact.size());
    for(auto &kv: tg.exact) leaves.push_back(kv.second);
    int nleaf = (int)leaves.size();
    if(nleaf==0) return false;

    Int mi; mi.SetInt64((int64_t)st.M);
    Point giant_step_point = secp.ComputePublicKey(&mi); giant_step_point.Reduce();
    GpuECPoint giant_step_gpu = point_to_gpu_point(giant_step_point);

    Int start_offset; start_offset.SetInt64((int64_t)search_start);
    start_offset.Mult(&mi);
    Point start_p = secp.ComputePublicKey(&start_offset); start_p.Reduce();
    Point neg_start_p = secp.Negation(start_p); neg_start_p.Reduce();

    Int top_offset; top_offset.SetInt64((int64_t)(giant_count-1));
    top_offset.Mult(&mi);
    Point top_p = secp.ComputePublicKey(&top_offset); top_p.Reduce();
    Point neg_top_p = secp.Negation(top_p); neg_top_p.Reduce();

    std::vector<GpuECPoint> lower_start(nleaf), upper_start(nleaf);
    for(int l=0;l<nleaf;++l){
        Point pt;
        if(!pub_to_point(secp, leaves[l].pub, pt)) continue;
        Point Pl = secp.Add(pt, neg_start_p); Pl.Reduce();
        Point Pu = secp.Add(pt, neg_top_p); Pu.Reduce();
        lower_start[l] = point_to_gpu_point(Pl);
        upper_start[l] = point_to_gpu_point(Pu);
    }

    int is_upper=0, found_i2=-1, found_i3=-1, found_idx=-1;
    uint64_t found_round=0;
    int leaf_idx = gpu_hierarchy_search_block(hgpu, lower_start.data(), upper_start.data(), nleaf, half_giant,
                                               &is_upper, &found_round, &found_i2, &found_i3, &found_idx);
    if(leaf_idx<0) return false;

    uint64_t resolved_i = is_upper ? (giant_count-1-found_round) : (search_start-found_round);
    uint64_t A = resolved_i*st.M + (uint64_t)found_i2*st.M2 + (uint64_t)found_i3*st.M3 + (uint64_t)found_idx;
    printf("[gpu-search DIAGNOSTIC] leaf_idx=%d is_upper=%d found_round=%" PRIu64
           " search_start=%" PRIu64 " giant_count=%" PRIu64 " half_giant=%" PRIu64
           " resolved_i=%" PRIu64 " i2=%d i3=%d idx=%d leaf_b=%" PRIu64 " A=%" PRIu64 "\n",
           leaf_idx, is_upper, found_round, search_start, giant_count, half_giant,
           resolved_i, found_i2, found_i3, found_idx, leaves[leaf_idx].b, A);
    *out_A = A;
    *out_b = leaves[leaf_idx].b;
    return true;
}
#endif

static void run_gen_self_test(Secp256K1 &secp, int steps, uint64_t test_b){
    printf("[gen-selftest] validating generation for steps=%d, b=%" PRIu64 "\n", steps, test_b);

    Int Ktrue; Ktrue.Rand(steps+20);
    Point root = secp.ComputePublicKey(&Ktrue); root.Reduce();

    TreeGenContext ctx;
    tree_gen_init(secp, ctx, root, steps);

    {
        Point R_scaled = ctx.R;
        for(int i=0;i<steps;++i){ R_scaled = secp.Double(R_scaled); R_scaled.Reduce(); }
        bool r0_ok = same_point(R_scaled, root);
        printf("[gen-selftest][DIAG] R*2^steps == root (isolated b=0 case): %s\n", r0_ok?"OK":"FAILED");
        if(!r0_ok){
            printf("[gen-selftest][DIAG] root.x=%s\n", root.x.GetBase16());
            printf("[gen-selftest][DIAG] R_scaled.x=%s\n", R_scaled.x.GetBase16());
        }
    }

    std::vector<LeafExact> seq_batch;
    gen_sequential_leaf_batch(secp, ctx, test_b+1, 1, seq_batch);
    Point leaf_seq;
    { bool ok = pub_to_point(secp, seq_batch[test_b].pub, leaf_seq); if(!ok) die("pub_to_point failed (seq)"); }

    Int bI; bI.SetInt64((int64_t)test_b);
    Point leaf_direct = tree_gen_leaf_at(secp, ctx, bI);

    bool caminhos_batem = same_point(leaf_seq, leaf_direct);
    printf("[gen-selftest] sequential vs. direct-scalar for the same b: %s\n",
           caminhos_batem?"OK":"FAILED");

    Point leaf_scaled = leaf_direct;
    for(int i=0;i<steps;++i){ leaf_scaled = secp.Double(leaf_scaled); leaf_scaled.Reduce(); }
    Point bG = secp.ComputePublicKey(&bI); bG.Reduce();
    Point reconstructed = secp.Add(leaf_scaled, bG); reconstructed.Reduce();
    bool equacao_bate = same_point(reconstructed, root);
    printf("[gen-selftest] leaf*2^steps + b*G == root: %s\n", equacao_bate?"OK":"FAILED");

    std::vector<LeafExact> rand_batch; std::vector<Int> b_full;
    gen_random_leaf_batch(secp, ctx, 100, rand_batch, b_full);
    int falhas_random=0;
    for(int i=0;i<100;++i){
        Point leaf_r;
        if(!pub_to_point(secp, rand_batch[i].pub, leaf_r)){ falhas_random++; continue; }
        Point ls = leaf_r;
        for(int s=0;s<steps;++s){ ls = secp.Double(ls); ls.Reduce(); }
        Point bGr = secp.ComputePublicKey(&b_full[i]); bGr.Reduce();
        Point rec = secp.Add(ls, bGr); rec.Reduce();
        if(!same_point(rec, root)) falhas_random++;
    }
    printf("[gen-selftest] 100 random leaves, equation valid in: %d/100\n", 100-falhas_random);

    if(caminhos_batem && equacao_bate && falhas_random==0)
        printf("[gen-selftest] ALL OK\n");
    else
        printf("[gen-selftest] FAILED - there is a bug in generation\n");
}

static void run_self_test(Secp256K1 &secp, int test_a_bits, uint64_t test_n, int test_M_bits,
                          int threads, int gen_batch, double fp_rate, int force_degenerate){
    uint64_t range = (uint64_t)1<<test_a_bits;
    uint64_t M = (uint64_t)1<<test_M_bits;
    uint64_t M2 = (M+31)/32;
    uint64_t M3 = (M2+31)/32;

    uint64_t A_true;
    if(force_degenerate==1){
        uint64_t i2 = 3; if(i2>=NARROW_STEPS) i2=1;
        A_true = i2*M2;
        if(A_true>=range) A_true%=range;
        printf("[selftest] DEGENERATE MODE r2=0 forced (i2=%" PRIu64 ", A_true=i2*M2=%" PRIu64 ")\n", i2, A_true);
    } else if(force_degenerate==2){
        uint64_t i2 = 2; if(i2>=NARROW_STEPS) i2=1;
        uint64_t i3 = 5; if(i3>=NARROW_STEPS) i3=1;
        A_true = i2*M2 + i3*M3;
        if(A_true>=range) A_true%=range;
        printf("[selftest] DEGENERATE MODE r3=0 forced (i2=%" PRIu64 " i3=%" PRIu64 ", A_true=%" PRIu64 ")\n", i2, i3, A_true);
    } else if(force_degenerate==3){

        uint64_t giant_count = (range + M - 1)/M;
        uint64_t i_true = giant_count - 1;
        A_true = i_true*M + (M/3);
        if(A_true>=range) A_true = range-1;
        printf("[selftest] UPPER-HALF MODE forced (i_true=%" PRIu64 "/%" PRIu64 ", A_true=%" PRIu64 ")\n",
               i_true, giant_count, A_true);
    } else {
        A_true = (range/3) ^ 0x1357ULL; if(A_true>=range) A_true%=range;
    }
    uint64_t b_true = test_n/2;

    printf("[selftest] scenario: A_true=%" PRIu64 " b_true=%" PRIu64
           " range=2^%d N=%" PRIu64 "\n", A_true, b_true, test_a_bits, test_n);
    printf("[selftest] expected K = A_true = %" PRIu64 " (0x%llx)\n\n",
           A_true, (unsigned long long)A_true);

    PrecomputedTargets tg;
    tg.n_leaves = test_n;
    srand(12345);
    for(uint64_t i=0;i<test_n;++i){
        Int scalar;
        if(i==b_true){
            scalar.SetInt64((int64_t)A_true);
        } else {
            uint64_t hi = ((uint64_t)rand()<<32) ^ (uint64_t)rand() ^ (i*0x9E3779B97F4A7C15ULL);
            hi |= (UINT64_C(1)<<50);
            hi &= 0x7FFFFFFFFFFFFFFFULL;
            scalar.SetInt64((int64_t)hi);
        }
        Point p = secp.ComputePublicKey(&scalar); p.Reduce();
        char *hex = secp.GetPublicKeyHex(true, p);
        unsigned char pubbytes[PUBKEY_SIZE];
        if(!parse_pubkey(hex, pubbytes)) continue;
        unsigned char xb[32]; p.x.Get32Bytes(xb);
        uint64_t fp = fingerprint_32(xb);
        LeafExact le; memcpy(le.pub, pubbytes, PUBKEY_SIZE); le.b = i;
        tg.exact[fp] = le;
    }

    HierState st;
    build_hierarchy(secp, st, M, threads, gen_batch, fp_rate, 0, 0);

    if(force_degenerate) g_debug_b = (int64_t)(test_n/2);
    run_hier_search(secp, st, tg, 0, test_a_bits, threads, gen_batch, false, NULL);
    g_debug_b = -1;
    free_hierarchy(st);
}

static void usage(const char *prog){
    printf(
      "Usage:\n"
      "  %s --self-test [--test-a-bits N] [--test-n N] [--m-bits N]\n"
      "     [-t N] [--gen-batch N] [--fp-rate F] [--test-degenerate N]\n"
      "     Validates with a known answer before attacking real data.\n"
      "     --test-degenerate 1: force r2=0 (level-2 degenerate case)\n"
      "     --test-degenerate 2: force r3=0 (level-3 degenerate case)\n"
      "     --test-degenerate 3: force the answer into the UPPER half (tests the upper front)\n"
      "\n"
      "  %s --gen-self-test [--tree-steps N] [--gen-test-b N]\n"
      "     Validates the leaf-generation formula (sequential and random)\n"
      "     BEFORE using --attack-loop. Always run this first.\n"
      "\n"
      "  %s --build --m-bits N --hier-table FILE [-t N] [--gen-batch N] [--fp-rate F]\n"
      "     [--bloom1-k N]\n"
      "     Builds the hierarchy (M/M2/M3 + 3 blooms) and saves it to FILE.\n"
      "     --bloom1-k N: override Bloom1's hash count manually (0/omitted =\n"
      "     theoretically optimal for --fp-rate). For experimenting with fewer\n"
      "     hashes (fewer memory accesses per check) against a higher false-\n"
      "     positive rate, since Bloom1 dominates search time (~94%% measured).\n"
      "\n"
      "  %s --search --hier-table FILE --target-bloom FILE --tree-steps N\n"
      "     --root-bits N [-t N] [--restrict-upper-half]\n"
      "     Runs the bidirectional search against all leaves using the loaded hierarchy.\n"
      "\n"
      "  %s --attack-loop --hier-table FILE --start-pubkey HEX --tree-steps N\n"
      "     [--root-bits N] [--batch-size N] [--max-batches N] [--sequential]\n"
      "     [--restrict-upper-half] [-t N]\n"
      "     Generates batches of leaves (random by default, or --sequential) from\n"
      "     the master key, and searches each batch until found or until\n"
      "     --max-batches is reached (0 = no limit, runs indefinitely).\n"
      "     tree_steps has NO limit (supports up to 256 bits).\n"
      "     --restrict-upper-half: valid ONLY for a real puzzle-N target - A is\n"
      "     PROVABLY in the upper half of its range (K's guaranteed top bit\n"
      "     shifts into A's top bit), so the lower half is skipped entirely -\n"
      "     a real ~2x speedup, not a heuristic. Never use with synthetic or\n"
      "     unknown-range targets.\n"
      "     --dump-round-stats FILE: writes a CSV of per-stage cost deltas\n"
      "     (Bloom1/i2/Bloom2/i3, both operation counts and time) every 64\n"
      "     rounds, for post-hoc analysis of where time actually goes over\n"
      "     the course of a real search. Purely observational - does not\n"
      "     change the search itself. Also available on --search.\n"
      "     --numa-aware: interleaves Bloom1/2/3 across all NUMA nodes and\n"
      "     pins threads round-robin to nodes. Only takes effect if built\n"
      "     with -DUSE_NUMA -lnuma AND the machine has >1 NUMA node -\n"
      "     harmless no-op otherwise (e.g. this session's dev machine,\n"
      "     confirmed single-node via numactl --hardware). Intended for a\n"
      "     multi-socket/multi-node server, not this development machine.\n"
      "     --prune-selftest: a scaffold for testing a proposed 'only one\n"
      "     child is valid at the first N halving levels' rule against a\n"
      "     known answer. The actual decision rule (child_decision() in\n"
      "     the source) currently returns 'undecided' unconditionally.\n"
      "     Options: --prune-b N (default: 70356, Puzzle #64's real,\n"
      "     known b), --prune-levels N (default 19),\n"
      "     --start-pubkey HEX (default: Puzzle #64's real pubkey).\n"
      "     --repeat-test: tests the rule 'exclude any path with N\n"
      "     consecutive identical halving choices anywhere' - purely\n"
      "     combinatorial (no EC operations needed), exhaustively scans\n"
      "     all 2^--prune-levels possible b values (when levels<=24) and\n"
      "     reports what fraction would be excluded. --prune-repeat-n N\n"
      "     (default 4) sets the repeat length.\n"
      "     --unsafe-prune (with --attack-loop): applies the repeat-\n"
      "     exclusion rule to leaf generation (both --sequential and\n"
      "     random modes), using --prune-repeat-n (default 4).\n"
      "     --gen-pruned-leaves: generates a batch of leaves (--tree-steps,\n"
      "     --start-pubkey, --batch-size for count, --sequential/random as\n"
      "     usual) and reports the resulting set AFTER applying whatever\n"
      "     --unsafe-prune/--prune-repeat-n rule\n"
      "     is active (full, unpruned batch if the flag isn't set). Prints\n"
      "     a summary (kept/pruned counts) and a preview of the first\n"
      "     --preview-count (default 20) surviving leaves. --output-leaves\n"
      "     FILE writes the full surviving set to a simple custom binary\n"
      "     format (magic+version+tree_steps+repeat_n+count, then per-leaf\n"
      "     b + compressed pubkey) - NOT the same format --search reads\n"
      "     (which also embeds its own Bloom filter). --start-b N shifts\n"
      "     sequential mode's starting b (default 0).\n"
      "     --block-search: like --attack-loop --sequential, but divides\n"
      "     the b-range into blocks of --batch-size and, if\n"
      "     --rounds-per-block N is set, caps each block's search at N\n"
      "     rounds before moving to the next block automatically (rather\n"
      "     than fully exhausting each block first). Combines with\n"
      "     --unsafe-prune/--prune-repeat-n. Uses Int (arbitrary-precision)\n"
      "     arithmetic for the block offset, so --tree-steps beyond 64 is\n"
      "     handled correctly. --max-batches caps the number of blocks (0 =\n"
      "     unbounded, Ctrl+C to stop).\n",
      prog,prog,prog,prog,prog);
}

static bool process_one_block_race(Secp256K1 &secp, HierState &st, TreeGenContext &ctx,
                                    int tree_steps, int a_bits, int total_threads,
                                    Int &block_offset, uint64_t block_size,
                                    bool restrict_upper_half, std::string &out_K,
                                    uint64_t *out_leaf_count=nullptr){
    TreeGenContext ctx_off = ctx;
    if(!block_offset.IsZero()){
        Point new_R = tree_gen_leaf_at(secp, ctx, block_offset);
        ctx_off.R = new_R;
    }
    std::vector<LeafExact> leaves;
    gen_sequential_leaf_batch(secp, ctx_off, block_size, total_threads, leaves, &block_offset);
    if(out_leaf_count) *out_leaf_count = leaves.size();
    if(leaves.empty()) return false;

    PrecomputedTargets tg;
    tg.n_leaves = leaves.size();
    tg.exact.reserve(leaves.size()*2);
    for(auto &le : leaves){
        unsigned char xb[32]; memcpy(xb, le.pub+1, 32);
        uint64_t fp = fingerprint_32(xb);
        tg.exact[fp] = le;
    }

    uint64_t out_A=0, out_b=0;
    bool hit = run_hier_search(secp, st, tg, tree_steps, a_bits, total_threads, 64,
                                restrict_upper_half, nullptr, UINT64_MAX, UINT64_MAX,
                                nullptr, &out_A, &out_b, UINT64_MAX, true);

    if(!hit) return false;

    Int full_b; full_b.SetInt64((int64_t)out_b); full_b.Add(&block_offset);
    Int Ai; Ai.SetInt64((int64_t)out_A);
    Int two; two.SetInt64(2);
    Int pow2steps; pow2steps.SetInt64(1);
    for(int i=0;i<tree_steps;++i) pow2steps.Mult(&two);
    Int K; K.Set(&Ai); K.Mult(&pow2steps); K.Add(&full_b);
    out_K = K.GetBase16();
    return true;
}

#ifdef GPU_ENABLED

static bool process_one_block_race_gpu(Secp256K1 &secp, HierState &st, GpuHierarchy *hgpu, TreeGenContext &ctx,
                                        int tree_steps, int a_bits, int total_threads,
                                        Int &block_offset, uint64_t block_size,
                                        bool restrict_upper_half, std::string &out_K,
                                        uint64_t *out_leaf_count=nullptr){
    TreeGenContext ctx_off = ctx;
    if(!block_offset.IsZero()){
        Point new_R = tree_gen_leaf_at(secp, ctx, block_offset);
        ctx_off.R = new_R;
    }
    std::vector<LeafExact> leaves;
    gen_sequential_leaf_batch(secp, ctx_off, block_size, total_threads, leaves, &block_offset);
    if(out_leaf_count) *out_leaf_count = leaves.size();
    if(leaves.empty()) return false;

    PrecomputedTargets tg;
    tg.n_leaves = leaves.size();
    tg.exact.reserve(leaves.size()*2);
    for(auto &le : leaves){
        unsigned char xb[32]; memcpy(xb, le.pub+1, 32);
        uint64_t fp = fingerprint_32(xb);
        tg.exact[fp] = le;
    }

    uint64_t out_A=0, out_b=0;
    bool hit = run_hier_search_gpu(secp, st, hgpu, tg, tree_steps, a_bits, restrict_upper_half, &out_A, &out_b);
    if(!hit) return false;

    Int full_b; full_b.SetInt64((int64_t)out_b); full_b.Add(&block_offset);
    Int Ai; Ai.SetInt64((int64_t)out_A);
    Int two; two.SetInt64(2);
    Int pow2steps; pow2steps.SetInt64(1);
    for(int i=0;i<tree_steps;++i) pow2steps.Mult(&two);
    Int K; K.Set(&Ai); K.Mult(&pow2steps); K.Add(&full_b);
    out_K = K.GetBase16();
    return true;
}
#endif

static bool net_download_file(const char *master_host, int master_port,
                               const char *which, const char *local_path){
    int fd = net_connect(master_host, master_port);
    if(fd<0){ printf("[worker] could not connect to master to fetch %s\n", which); return false; }
    char req[64]; snprintf(req,sizeof(req),"GETFILE %s", which);
    if(!net_send_line(fd, req)){ close(fd); return false; }
    std::string hdr;
    if(!net_recv_line(fd, hdr)){ close(fd); return false; }
    auto tok = net_split(hdr);
    if(tok.empty() || tok[0]!="FILE" || tok.size()<2){
        printf("[worker] master has no %s table to serve (%s)\n", which, hdr.c_str());
        close(fd); return false;
    }
    long fsize = atol(tok[1].c_str());
    printf("[worker] downloading %s table from master (%ld bytes) -> %s\n", which, fsize, local_path);
    FILE *f = fopen(local_path, "wb");
    if(!f){ printf("[worker] could not open %s for writing\n", local_path); close(fd); return false; }
    char chunk[1<<20];
    long received=0; double t0=now_seconds(), last_print=t0;
    while(received<fsize){
        ssize_t n = recv(fd, chunk, (size_t)std::min((long)sizeof(chunk), fsize-received), 0);
        if(n<=0){ printf("[worker] download interrupted at %ld/%ld bytes\n", received, fsize);
                  fclose(f); close(fd); return false; }
        fwrite(chunk, 1, (size_t)n, f);
        received += n;
        double now = now_seconds();
        if(now-last_print>=2.0){
            printf("\r[worker] downloading %s: %ld/%ld bytes (%.1f%%)          ",
                   which, received, fsize, 100.0*(double)received/(double)fsize);
            fflush(stdout);
            last_print=now;
        }
    }
    printf("\r[worker] downloaded %s table: %ld bytes                    \n", which, received);
    fclose(f); close(fd);
    return true;
}

static void run_worker(Secp256K1 &secp, const char *master_host, int master_port,
                        const char *pubkey_hex, int tree_steps, int root_bits,
                        const char *hier_table_file,
                        bool restrict_upper_half, int threads){
    unsigned char root_pub[PUBKEY_SIZE];
    if(!parse_pubkey(pubkey_hex, root_pub)) die("invalid --start-pubkey");
    Point root;
    if(!pub_to_point(secp, root_pub, root)) die("could not decode --start-pubkey");
    int a_bits = root_bits - tree_steps;
    if(a_bits<1 || a_bits>45) die("root_bits - tree_steps must be between 1 and 45");
    TreeGenContext ctx;
    tree_gen_init(secp, ctx, root, tree_steps);

    if(!hier_table_file) die("--worker requires --hier-table");

    { FILE *probe = fopen(hier_table_file, "rb");
      if(probe) fclose(probe);
      else net_download_file(master_host, master_port, "hier", hier_table_file); }

    HierState st;
    bool have_bsgs = load_hierarchy(secp, st, hier_table_file);
    if(!have_bsgs) die("--worker could not load --hier-table (required - see above)");

    printf("[worker] ready - bsgs (all threads), connecting to %s:%d\n",
           master_host, master_port);

    while(true){
        int cfd = net_connect(master_host, master_port);
        if(cfd<0){ printf("[worker] could not connect to master, retrying in 5s\n");
                   std::this_thread::sleep_for(std::chrono::seconds(5)); continue; }
        net_send_line(cfd, "WORK");
        std::string line;
        bool ok = net_recv_line(cfd, line);
        close(cfd);
        if(!ok){ std::this_thread::sleep_for(std::chrono::seconds(2)); continue; }
        auto tok = net_split(line);
        if(tok.empty()) continue;

        if(tok[0]=="STOP"){ printf("[worker] master signalled STOP - exiting.\n"); return; }
        if(tok[0]=="WAIT"){
            double secs = tok.size()>=2 ? atof(tok[1].c_str()) : 2.0;
            std::this_thread::sleep_for(std::chrono::duration<double>(secs)); continue;
        }
        if(tok[0]!="WORK" || tok.size()<6){ printf("[worker] unexpected response: %s\n", line.c_str());
            std::this_thread::sleep_for(std::chrono::seconds(2)); continue; }

        uint64_t wid = strtoull(tok[1].c_str(), NULL, 10);
        Int offset; offset.SetBase16(tok[2].c_str()+2);
        uint64_t block_size = strtoull(tok[3].c_str(), NULL, 10);
        int prune = atoi(tok[4].c_str());
        int prune_n = atoi(tok[5].c_str());
        g_unsafe_prune = prune; g_prune_repeat_n = prune_n;

        printf("[worker] work #%" PRIu64 ": offset=0x%s block_size=%" PRIu64
               " (bsgs, %d threads)\n", wid, offset.GetBase16(), block_size, threads);

        std::string found_K; uint64_t leaf_count=0;
        double t0 = now_seconds();
        bool hit = process_one_block_race(secp, st, ctx, tree_steps, a_bits, threads,
                                           offset, block_size, restrict_upper_half,
                                           found_K, &leaf_count);
        printf("[worker] work #%" PRIu64 " done in %.2fs, %" PRIu64 " leaves, %s\n",
               wid, now_seconds()-t0, leaf_count, hit?"FOUND":"not found");

        int rfd = net_connect(master_host, master_port);
        if(rfd<0){ printf("[worker] could not report result (master unreachable)\n"); continue; }
        char buf[256];
        if(hit){
            snprintf(buf,sizeof(buf),"RESULT %" PRIu64 " FOUND %s", wid, found_K.c_str());
            printf("[worker] *** FOUND *** (result withheld from worker log - see master) reporting to master.\n");
        } else {
            snprintf(buf,sizeof(buf),"RESULT %" PRIu64 " NOTFOUND", wid);
        }
        net_send_line(rfd, buf);
        std::string resp; net_recv_line(rfd, resp);
        close(rfd);
        if(hit){ printf("[worker] done - answer found, exiting.\n"); return; }
    }
}

int main(int argc,char **argv){
    int threads=1;
    int self_test_mode=0, build_mode=0, search_mode=0;
    int radix4_test_mode=0;
    int net_selftest_mode=0; int net_selftest_port=23940;
    int gpu_smoke_test_mode=0;
    int gpu_search_mode=0;
    int master_mode=0; int master_port=23941; double master_timeout=300.0;
    const char *master_checkpoint_file=nullptr;
    int worker_mode=0; const char *worker_master_host="127.0.0.1"; int worker_master_port=23941;
    int prune_selftest_mode=0;
    int fourrepeat_test_mode=0;
    int gen_pruned_leaves_mode=0;
    int block_search_mode=0;
    uint64_t rounds_per_block=UINT64_MAX;
    uint64_t start_block=0;
    const char *output_leaves_file=nullptr;
    int preview_count=20;
    uint64_t gpl_start_b=0;
    uint64_t prune_selftest_b = 70356;
    int prune_selftest_levels = 19;
    int gen_self_test_mode=0, attack_loop_mode=0;
    int test_a_bits=20; uint64_t test_n=1024; int test_m_bits=12;
    int m_bits=20;
    char *hier_table_file=NULL;
    char *target_bloom_file=NULL;
    char *start_pubkey_hex=NULL;
    char *target_pubkey_hex=NULL;
    char *range_lo_hex=NULL, *range_hi_hex=NULL;
    uint64_t gpu_lanes=65536;
    int num_gpus=1;
    int tree_steps=-1, root_bits=64;
    int gen_batch=64;
    double fp_rate=DEFAULT_BLOOM_FP_RATE;
    int bloom1_k=0;
    uint64_t bloom1_block_bits=0;
    int force_degenerate=0;
    uint64_t batch_size=524288;
    uint64_t max_batches=0;
    int random_leaves=1;
    int restrict_upper_half=0;
    char *dump_round_stats_path=NULL;
    uint64_t gen_test_b=1000;

    for(int i=1;i<argc;++i){
        auto next=[&](){ if(++i>=argc) die("missing value"); return argv[i]; };
        if(!strcmp(argv[i],"--self-test")){ self_test_mode=1; }
        else if(!strcmp(argv[i],"--radix4-test")){ radix4_test_mode=1; }
        else if(!strcmp(argv[i],"--net-selftest")){ net_selftest_mode=1; }
        else if(!strcmp(argv[i],"--net-port")){ net_selftest_port=atoi(next()); }
        else if(!strcmp(argv[i],"--gpu-smoke-test")){ gpu_smoke_test_mode=1; }
        else if(!strcmp(argv[i],"--gpu-search")){ gpu_search_mode=1; }
        else if(!strcmp(argv[i],"--target-pubkey")){ target_pubkey_hex=next(); }
        else if(!strcmp(argv[i],"--range-lo")){ range_lo_hex=next(); }
        else if(!strcmp(argv[i],"--range-hi")){ range_hi_hex=next(); }
        else if(!strcmp(argv[i],"--lanes")){ gpu_lanes=strtoull(next(),NULL,10); }
        else if(!strcmp(argv[i],"--gpus")){ num_gpus=atoi(next()); }
        else if(!strcmp(argv[i],"--start-block")){ start_block=strtoull(next(),NULL,10); }
        else if(!strcmp(argv[i],"--master")){ master_mode=1; }
        else if(!strcmp(argv[i],"--master-port")){ master_port=atoi(next()); }
        else if(!strcmp(argv[i],"--master-timeout")){ master_timeout=atof(next()); }
        else if(!strcmp(argv[i],"--master-checkpoint")){ master_checkpoint_file=next(); }
        else if(!strcmp(argv[i],"--worker")){ worker_mode=1; }
        else if(!strcmp(argv[i],"--master-host")){ worker_master_host=next(); }
        else if(!strcmp(argv[i],"--worker-master-port")){ worker_master_port=atoi(next()); }
        else if(!strcmp(argv[i],"--prune-selftest")){ prune_selftest_mode=1; }
        else if(!strcmp(argv[i],"--prune-b")){ prune_selftest_b=strtoull(next(),NULL,10); }
        else if(!strcmp(argv[i],"--prune-levels")){ prune_selftest_levels=atoi(next()); }
        else if(!strcmp(argv[i],"--repeat-test")){ fourrepeat_test_mode=1; }
        else if(!strcmp(argv[i],"--unsafe-prune")){ g_unsafe_prune=1; }
        else if(!strcmp(argv[i],"--prune-repeat-n")){ g_prune_repeat_n=atoi(next()); }
        else if(!strcmp(argv[i],"--gen-pruned-leaves")){ gen_pruned_leaves_mode=1; }
        else if(!strcmp(argv[i],"--output-leaves")){ output_leaves_file=next(); }
        else if(!strcmp(argv[i],"--preview-count")){ preview_count=atoi(next()); }
        else if(!strcmp(argv[i],"--start-b")){ gpl_start_b=strtoull(next(),NULL,10); }
        else if(!strcmp(argv[i],"--block-search")){ block_search_mode=1; }
        else if(!strcmp(argv[i],"--rounds-per-block")){ rounds_per_block=strtoull(next(),NULL,10); }
        else if(!strcmp(argv[i],"--build")){ build_mode=1; }
        else if(!strcmp(argv[i],"--search")){ search_mode=1; }
        else if(!strcmp(argv[i],"--test-a-bits")){ test_a_bits=atoi(next()); }
        else if(!strcmp(argv[i],"--test-n")){ test_n=strtoull(next(),NULL,10); }
        else if(!strcmp(argv[i],"--m-bits")){ m_bits=atoi(next()); test_m_bits=m_bits; }
        else if(!strcmp(argv[i],"--hier-table")){ hier_table_file=next(); }
        else if(!strcmp(argv[i],"--target-bloom")){ target_bloom_file=next(); }
        else if(!strcmp(argv[i],"--tree-steps")){ tree_steps=atoi(next()); }
        else if(!strcmp(argv[i],"--root-bits")){ root_bits=atoi(next()); }
        else if(!strcmp(argv[i],"--gen-batch")){ gen_batch=atoi(next()); }
        else if(!strcmp(argv[i],"--fp-rate")){ fp_rate=atof(next()); }
        else if(!strcmp(argv[i],"--bloom1-k")){ bloom1_k=atoi(next()); }
        else if(!strcmp(argv[i],"--bloom1-block-bits")){ bloom1_block_bits=strtoull(next(),NULL,10); }
        else if(!strcmp(argv[i],"--test-degenerate")){ force_degenerate=atoi(next()); }
        else if(!strcmp(argv[i],"--threads")||!strcmp(argv[i],"-t")){ threads=(int)strtol(next(),NULL,10); }
        else if(!strcmp(argv[i],"--gen-self-test")){ gen_self_test_mode=1; }
        else if(!strcmp(argv[i],"--attack-loop")){ attack_loop_mode=1; }
        else if(!strcmp(argv[i],"--start-pubkey")){ start_pubkey_hex=next(); }
        else if(!strcmp(argv[i],"--batch-size")){ batch_size=strtoull(next(),NULL,10); }
        else if(!strcmp(argv[i],"--max-batches")){ max_batches=strtoull(next(),NULL,10); }
        else if(!strcmp(argv[i],"--sequential")){ random_leaves=0; }
        else if(!strcmp(argv[i],"--restrict-upper-half")){ restrict_upper_half=1; }
        else if(!strcmp(argv[i],"--dump-round-stats")){ dump_round_stats_path=next(); }
        else if(!strcmp(argv[i],"--numa-aware")){ g_numa_aware=1; }
        else if(!strcmp(argv[i],"--gen-test-b")){ gen_test_b=strtoull(next(),NULL,10); }
        else if(!strcmp(argv[i],"--help")||!strcmp(argv[i],"-h")){ usage(*argv); return 0; }
        else { fprintf(stderr,"Unknown: %s\n",argv[i]); return 1; }
    }

    Secp256K1 secp; secp.Init();

    if(self_test_mode){
        run_self_test(secp, test_a_bits, test_n, test_m_bits, threads, gen_batch, fp_rate, force_degenerate);
        return 0;
    }

    if(gen_self_test_mode){
        int steps_to_test = tree_steps>0 ? tree_steps : 95;
        run_gen_self_test(secp, steps_to_test, gen_test_b);
        return 0;
    }

    if(master_mode){
        if(!start_pubkey_hex) die("--master requires --start-pubkey HEX");
        if(tree_steps<0) die("--master requires --tree-steps N");
        run_master(start_pubkey_hex, tree_steps, root_bits, batch_size,
                   restrict_upper_half!=0, g_unsafe_prune, g_prune_repeat_n,
                   master_port, master_timeout, master_checkpoint_file,
                   hier_table_file);
        return 0;
    }

    if(worker_mode){
        if(!start_pubkey_hex) die("--worker requires --start-pubkey HEX");
        if(tree_steps<0) die("--worker requires --tree-steps N");
        if(!hier_table_file)
            die("--worker requires --hier-table");
        run_worker(secp, worker_master_host, worker_master_port, start_pubkey_hex, tree_steps, root_bits,
                   hier_table_file, restrict_upper_half!=0, threads);
        return 0;
    }

    if(gpu_smoke_test_mode){
#ifdef GPU_ENABLED
        if(!hier_table_file) die("--gpu-smoke-test requires --hier-table FILE");
        HierState st;
        if(!load_hierarchy(secp, st, hier_table_file)) die("could not load --hier-table");
        bool ok = run_gpu_smoke_test(secp, st);
        return ok ? 0 : 1;
#else
        die("this binary was built without GPU support - rebuild with 'make GPU=1' (requires nvcc/CUDA toolkit)");
#endif
    }

    if(gpu_search_mode){
#ifdef GPU_ENABLED
        if(target_pubkey_hex){

            if(!hier_table_file) die("--gpu-search --target-pubkey requires --hier-table FILE");
            if(!range_lo_hex || !range_hi_hex) die("--target-pubkey requires --range-lo and --range-hi (hex)");

            unsigned char target_pub[PUBKEY_SIZE];
            if(!parse_pubkey(target_pubkey_hex, target_pub)) die("invalid --target-pubkey");
            Point target_point;
            if(!pub_to_point(secp, target_pub, target_point)) die("could not decode --target-pubkey");

            Int range_lo, range_hi;
            range_lo.SetBase16(range_lo_hex);
            range_hi.SetBase16(range_hi_hex);
            Int range_size; range_size.Set(&range_hi); range_size.Sub(&range_lo);
            if(range_size.IsZero()) die("--range-hi must be greater than --range-lo");

            HierState st;
            if(!load_hierarchy(secp, st, hier_table_file)) die("could not load --hier-table (use --build first)");

            int available_gpus = gpu_query_device_count();
            if(available_gpus==0) die("--gpu-search: no usable CUDA GPU found");
            int use_gpus = num_gpus>0 ? num_gpus : 1;
            if(use_gpus > available_gpus) use_gpus = available_gpus;
            printf("[+] --gpu-search (single target): using %d of %d available GPU(s)\n", use_gpus, available_gpus);
            for(int g=0; g<use_gpus; ++g){
                char gpu_info[256];
                gpu_query_device_info_n(g, gpu_info, sizeof(gpu_info));
                printf("[+]   GPU %d: %s\n", g, gpu_info);
            }
            printf("[+] range: [0x%s, 0x%s), size ~2^%d bits\n",
                   range_lo.GetBase16(), range_hi.GetBase16(), range_size.GetBitLength());

            Int mi_gs; mi_gs.SetInt64((int64_t)st.M);
            Point giant_step_setup = secp.ComputePublicKey(&mi_gs); giant_step_setup.Reduce();
            GpuECPoint giant_step_gpu_setup = point_to_gpu_point(giant_step_setup);
            GpuECPoint h_shift2[32], h_shift3[32];
            for(int i=0;i<32;++i){ h_shift2[i]=point_to_gpu_point(st.shift2[i]); h_shift3[i]=point_to_gpu_point(st.shift3[i]); }

            uint64_t N = gpu_lanes;
            printf("[+] Setting up %" PRIu64 " parallel lanes across %d GPU(s)...\n", N, use_gpus);

            Point range_lo_point = secp.ComputePublicKey(&range_lo); range_lo_point.Reduce();
            Point neg_range_lo = secp.Negation(range_lo_point); neg_range_lo.Reduce();
            Point base_point = secp.Add(target_point, neg_range_lo); base_point.Reduce();

            std::vector<GpuECPoint> lanes(N);
            Point cur_lane = base_point;
            lanes[0] = point_to_gpu_point(cur_lane);
            Point neg_giant_setup = secp.Negation(giant_step_setup); neg_giant_setup.Reduce();
            for(uint64_t L=1; L<N; ++L){
                cur_lane = secp.Add(cur_lane, neg_giant_setup); cur_lane.Reduce();
                lanes[L] = point_to_gpu_point(cur_lane);
            }

            Int nm; nm.SetInt64((int64_t)N); nm.Mult(&mi_gs);
            Point stride_point = secp.ComputePublicKey(&nm); stride_point.Reduce();
            GpuECPoint stride_gpu = point_to_gpu_point(stride_point);

            const int K = 32;
            uint64_t giant_count = (range_size.bits64[0] / st.M) + 1;

            std::atomic<bool> found{false};
            std::atomic<uint64_t> total_checked{0};
            std::atomic<int> active_workers{use_gpus};
            std::mutex result_mutex;
            uint64_t found_abs_i=0; int found_i2=-1, found_i3=-1, found_idx=-1;
            bool have_result=false;

            std::vector<uint64_t> slice_start(use_gpus+1);
            for(int g=0; g<=use_gpus; ++g) slice_start[g] = (N*(uint64_t)g)/(uint64_t)use_gpus;

            double t_start = now_seconds();

            std::vector<std::thread> workers;
            for(int g=0; g<use_gpus; ++g){
                workers.emplace_back([&, g](){
                    /* active_workers is decremented on EVERY exit path
                       (RAII, so a return/break anywhere below still
                       triggers it) - the polling loop below needs this,
                       since std::thread::joinable() only means "not yet
                       join()-ed", NOT "still running": a thread whose
                       function already returned stays joinable() until
                       actually joined, so checking it was a real,
                       reachable bug - confirmed directly (both GPUs at
                       0% utilization, meaning both worker lambdas had
                       already finished, yet the main thread's own
                       polling loop kept believing they were still alive
                       forever, since it never actually joined until
                       AFTER that same loop - a real deadlock, not a
                       display glitch). */
                    struct Dec { std::atomic<int> &c; ~Dec(){ c.fetch_sub(1, std::memory_order_relaxed); } } dec{active_workers};
                    if(!gpu_set_device(g)){
                        fprintf(stderr, "[gpu-scan] GPU %d: failed to select device, this worker will not contribute\n", g);
                        return;
                    }
                    GpuHierarchy *hgpu_g = gpu_hierarchy_build(
                        st.bloom1.bits, st.bloom1.bit_count, st.bloom1.nwords, st.bloom1.k_hashes, st.bloom1.block_bits,
                        st.bloom2.bits, st.bloom2.bit_count, st.bloom2.nwords, st.bloom2.k_hashes, st.bloom2.block_bits,
                        st.bloom3.bits, st.bloom3.bit_count, st.bloom3.nwords, st.bloom3.k_hashes, st.bloom3.block_bits,
                        h_shift2, h_shift3, st.baby_table, st.baby_array_size, st.baby_K, &giant_step_gpu_setup);
                    if(!hgpu_g){
                        fprintf(stderr, "[gpu-scan] GPU %d: failed to upload hierarchy, this worker will not contribute\n", g);
                        return;
                    }

                    uint64_t lo = slice_start[g], hi = slice_start[g+1];
                    int slice_n = (int)(hi-lo);
                    if(slice_n<=0){ gpu_hierarchy_free(hgpu_g); return; }
                    std::vector<GpuECPoint> my_lanes(lanes.begin()+lo, lanes.begin()+hi);

                    uint64_t chunk_base = 0;
                    while(!found.load(std::memory_order_relaxed) && chunk_base < giant_count){
                        int out_round=-1, out_i2=-1, out_i3=-1, out_idx=-1;
                        int lane = gpu_range_scan_chunk(hgpu_g, my_lanes.data(), slice_n, &stride_gpu, K,
                                                         &out_round, &out_i2, &out_i3, &out_idx);
                        total_checked.fetch_add((uint64_t)slice_n*(uint64_t)K, std::memory_order_relaxed);
                        if(lane>=0){
                            std::lock_guard<std::mutex> lk(result_mutex);
                            if(!have_result){

                                found_abs_i = chunk_base + lo + (uint64_t)lane + (uint64_t)out_round*N;
                                found_i2=out_i2; found_i3=out_i3; found_idx=out_idx;
                                have_result=true;
                            }
                            found.store(true, std::memory_order_relaxed);
                            break;
                        }
                        chunk_base += N*(uint64_t)K;
                    }
                    gpu_hierarchy_free(hgpu_g);
                });
            }

            double t_last_change = t_start, t_last_print = t_start;
            uint64_t last_checked = 0;
            double displayed_mks = 0.0;
            while(!found.load(std::memory_order_relaxed)){
                if(active_workers.load(std::memory_order_relaxed)<=0) break;
                double now = now_seconds();
                uint64_t tc = total_checked.load(std::memory_order_relaxed);
                if(tc != last_checked){
                    displayed_mks = (tc-last_checked)/1e6/(now-t_last_change);
                    last_checked = tc;
                    t_last_change = now;
                }
                if(now - t_last_print > 0.5){
                    printf("\r[gpu-scan] %.2f Mk/s (%d GPU%s) | checked ~2^%.2f of ~2^%.2f | %.1fs elapsed    ",
                           displayed_mks, use_gpus, use_gpus>1?"s":"",
                           log2((double)tc>0?(double)tc:1), log2((double)giant_count*N), now-t_start);
                    fflush(stdout);
                    t_last_print = now;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            for(auto &w : workers) if(w.joinable()) w.join();
            printf("\n");

            if(!have_result){
                printf("[+] range exhausted, not found.\n");
                return 1;
            }

            Int abs_i; abs_i.SetInt64((int64_t)found_abs_i);
            Int M_int; M_int.SetInt64((int64_t)st.M);
            Int A; A.Set(&abs_i); A.Mult(&M_int);
            Int i2i; i2i.SetInt64((int64_t)found_i2); Int M2i; M2i.SetInt64((int64_t)st.M2); i2i.Mult(&M2i); A.Add(&i2i);
            Int i3i; i3i.SetInt64((int64_t)found_i3); Int M3i; M3i.SetInt64((int64_t)st.M3); i3i.Mult(&M3i); A.Add(&i3i);
            Int idxi; idxi.SetInt64((int64_t)found_idx); A.Add(&idxi);
            Int K_final; K_final.Set(&A); K_final.Add(&range_lo);

            printf("\n*** FOUND ***\nK = %s\n", K_final.GetBase16());
            printf("[+] total time: %.2fs\n", now_seconds()-t_start);
            return 0;
        }
#else
        die("this binary was built without GPU support - rebuild with 'make GPU=1' (requires nvcc/CUDA toolkit)");
#endif
#ifdef GPU_ENABLED
        if(!hier_table_file) die("--gpu-search requires --hier-table FILE");
        if(!start_pubkey_hex) die("--gpu-search requires --start-pubkey HEX");
        if(tree_steps<0) die("--gpu-search requires --tree-steps N");
        if(!restrict_upper_half) die("--gpu-search currently requires --restrict-upper-half "
            "(the only case this project's own answers fall into - see README.md's notice)");

        unsigned char root_pub[PUBKEY_SIZE];
        if(!parse_pubkey(start_pubkey_hex, root_pub)) die("invalid --start-pubkey");
        Point root;
        if(!pub_to_point(secp, root_pub, root)) die("could not decode --start-pubkey");

        HierState st;
        if(!load_hierarchy(secp, st, hier_table_file)) die("could not load --hier-table (use --build first)");

        int a_bits = root_bits - tree_steps;
        if(a_bits<1 || a_bits>45) die("root_bits - tree_steps must be between 1 and 45");

        /* The multi-GPU block dispatcher's own offset cursor (below) uses
           an Int (arbitrary precision), not a uint64_t block number - a
           uint64_t counter cannot represent the full range needed once
           tree_steps-block_bits exceeds 64 (confirmed directly: a real
           run at tree_steps=95 wrapped its uint64_t block counter around
           after repeated maximal jumps, causing already-visited block
           numbers to reappear). Its stopping condition compares the
           cursor's own GetBitLength() against tree_steps directly,
           needing no separately-computed 2^tree_steps value at all. */
        {
            /* batch_size is a uint64_t (< 2^64 always); once tree_steps>=64,
               2^tree_steps >= 2^64 > any possible batch_size, so the check
               only needs real arithmetic (avoiding any Int comparison
               method) for tree_steps<64, where a plain uint64_t shift is
               well-defined (no UB) and exact. */
            if(tree_steps<64){
                uint64_t max_valid_batch = (uint64_t)1 << tree_steps;
                if(batch_size > max_valid_batch){
                    fprintf(stderr, "ERROR: --batch-size %" PRIu64 " exceeds 2^tree_steps = %" PRIu64
                            " - the total b-space for --tree-steps %d. A batch this large asks for leaves "
                            "that cannot exist at this tree_steps, silently producing a WRONG K if allowed "
                            "through (confirmed directly: reconstructed A comes out off by exactly one unit "
                            "of M). Use --batch-size <= %" PRIu64 ".\n",
                            batch_size, max_valid_batch, tree_steps, max_valid_batch);
                    exit(1);
                }
            }
        }

        int available_gpus = gpu_query_device_count();
        if(available_gpus==0) die("--gpu-search: no usable CUDA GPU found");
        int use_gpus = num_gpus>0 ? num_gpus : 1;
        if(use_gpus > available_gpus) use_gpus = available_gpus;
        printf("[+] --gpu-search: using %d of %d available GPU(s)\n", use_gpus, available_gpus);
        for(int g=0; g<use_gpus; ++g){
            char gpu_info[256];
            gpu_query_device_info_n(g, gpu_info, sizeof(gpu_info));
            printf("[+]   GPU %d: %s\n", g, gpu_info);
        }

        GpuECPoint h_shift2[32], h_shift3[32];
        for(int i=0;i<32;++i){ h_shift2[i]=point_to_gpu_point(st.shift2[i]); h_shift3[i]=point_to_gpu_point(st.shift3[i]); }
        Int mi_gs; mi_gs.SetInt64((int64_t)st.M);
        Point giant_step_setup = secp.ComputePublicKey(&mi_gs); giant_step_setup.Reduce();
        GpuECPoint giant_step_gpu_setup = point_to_gpu_point(giant_step_setup);

        TreeGenContext ctx;
        tree_gen_init(secp, ctx, root, tree_steps);

        /* block_bits: log2(batch_size), only meaningful (and only used)
           when --unsafe-prune is active and batch_size is a power of 2 -
           lets block_fully_excluded_jump below skip an entire run of
           blocks whose FIXED high-bit prefix alone already satisfies the
           pruning rule, in O(1), instead of checking each one
           individually via a real (if cheap) GPU dispatch. Matches
           --block-search's own CPU-side jump exactly (same function,
           same math) - added here after a real, reachable slowdown: a
           multi-GPU --gpu-search run with pruning active checked blocks
           one at a time near the start of a huge range, where small
           sequential block numbers are near-certain to be trivially
           excluded (their own binary representation has a long run of
           leading zeros by construction) - without the jump, escaping
           that region requires checking on the order of 2^(tree_steps-
           block_bits-prune_repeat_n) blocks one by one, impractical even
           at a fraction of a second each. */
        int block_bits = -1;
        if(g_unsafe_prune && batch_size>0 && (batch_size&(batch_size-1))==0){
            block_bits = 0; uint64_t bs=batch_size; while(bs>1){ bs>>=1; block_bits++; }
        }

        printf("[+] GPU search: tree_steps=%d, batch=%" PRIu64 " (%d GPU%s, blocks distributed round-robin)\n",
               tree_steps, batch_size, use_gpus, use_gpus>1?"s":"");

        /* Shared cursor is an Int (arbitrary precision), not a uint64_t
           block number - a uint64_t block counter cannot represent the
           full range once tree_steps-block_bits exceeds 64, and silently
           wraps around instead of stopping, confirmed directly at
           tree_steps=95. */
        Int cursor_offset;
        { Int start_block_i; start_block_i.SetInt64((int64_t)start_block);
          Int batch_size_i0; batch_size_i0.SetInt64((int64_t)batch_size);
          cursor_offset.Set(&start_block_i); cursor_offset.Mult(&batch_size_i0); }
        std::mutex cursor_mutex;
        std::atomic<uint64_t> blocks_processed{0};
        std::atomic<bool> found{false};
        std::mutex result_mutex;
        std::string found_K;
        bool have_result=false;
        double t_start = now_seconds();

        std::vector<std::thread> workers;
        for(int g=0; g<use_gpus; ++g){
            workers.emplace_back([&, g](){
                if(!gpu_set_device(g)){
                    fprintf(stderr, "[gpu-search] GPU %d: failed to select device, this worker will not contribute\n", g);
                    return;
                }
                GpuHierarchy *hgpu_g = gpu_hierarchy_build(
                    st.bloom1.bits, st.bloom1.bit_count, st.bloom1.nwords, st.bloom1.k_hashes, st.bloom1.block_bits,
                    st.bloom2.bits, st.bloom2.bit_count, st.bloom2.nwords, st.bloom2.k_hashes, st.bloom2.block_bits,
                    st.bloom3.bits, st.bloom3.bit_count, st.bloom3.nwords, st.bloom3.k_hashes, st.bloom3.block_bits,
                    h_shift2, h_shift3, st.baby_table, st.baby_array_size, st.baby_K, &giant_step_gpu_setup);
                if(!hgpu_g){
                    fprintf(stderr, "[gpu-search] GPU %d: failed to upload hierarchy, this worker will not contribute\n", g);
                    return;
                }
                printf("[gpu-search] GPU %d: hierarchy ready.\n", g);

                Int batch_size_i; batch_size_i.SetInt64((int64_t)batch_size);

                while(!found.load(std::memory_order_relaxed)){
                    Int my_offset;
                    bool have_block = false;
                    {
                        std::lock_guard<std::mutex> lk(cursor_mutex);
                        while(true){
                            if(max_batches>0 && blocks_processed.load(std::memory_order_relaxed)>=(uint64_t)max_batches) break;
                            /* cursor_offset >= 2^tree_steps  <=>  its bit
                               length exceeds tree_steps (a number with
                               bit_length B satisfies 2^(B-1)<=x<2^B, so
                               bit_length>tree_steps already implies
                               x>=2^tree_steps) - uses only GetBitLength(),
                               already confirmed elsewhere in this file,
                               rather than a Sub()+sign-check whose exact
                               semantics on underflow aren't confirmed. */
                            if(cursor_offset.GetBitLength() > tree_steps) break;
                            my_offset.Set(&cursor_offset);
                            if(block_bits>=0){
                                uint64_t jump = block_fully_excluded_jump(my_offset, tree_steps, g_prune_repeat_n, block_bits);
                                if(jump>1){
                                    /* this block, and (jump-1) more after
                                       it, are ALL fully excluded by their
                                       shared fixed prefix alone - advance
                                       the shared cursor past all of them
                                       at once and keep looking, entirely
                                       under the lock (cheap, no GPU work
                                       involved) rather than doing this one
                                       block at a time. */
                                    Int jump_amount; jump_amount.SetInt64((int64_t)jump);
                                    jump_amount.Mult(&batch_size_i);
                                    cursor_offset.Add(&jump_amount);
                                    continue;
                                }
                            }
                            cursor_offset.Add(&batch_size_i);
                            have_block = true;
                            break;
                        }
                    }
                    if(!have_block) break;
                    blocks_processed.fetch_add(1, std::memory_order_relaxed);

                    std::string my_K; uint64_t leaf_count=0;
                    double t0 = now_seconds();
                    bool hit = process_one_block_race_gpu(secp, st, hgpu_g, ctx, tree_steps, a_bits, threads,
                                                            my_offset, batch_size, restrict_upper_half!=0,
                                                            my_K, &leaf_count);
                    double dt = now_seconds()-t0;
                    printf("[gpu-search] GPU %d (offset=0x%s): %" PRIu64 " leaves, %.3fs, %s\n",
                           g, my_offset.GetBase16(), leaf_count, dt, hit?"FOUND":"not found");
                    if(hit){
                        std::lock_guard<std::mutex> lk(result_mutex);
                        if(!have_result){ found_K = my_K; have_result=true; }
                        found.store(true, std::memory_order_relaxed);
                        break;
                    }
                }
                gpu_hierarchy_free(hgpu_g);
            });
        }
        for(auto &w : workers) w.join();

        if(have_result){
            printf("\n*** FOUND ***\nK = %s\n", found_K.c_str());
            printf("[+] total time: %.2fs\n", now_seconds()-t_start);
            return 0;
        }
        printf("[+] search ended, not found (max-batches reached or all GPUs exhausted their share).\n");
        return 1;
#else
        die("this binary was built without GPU support - rebuild with 'make GPU=1' (requires nvcc/CUDA toolkit)");
#endif
    }

    if(net_selftest_mode){
        bool ok = run_net_self_test(net_selftest_port);
        return ok ? 0 : 1;
    }

    if(fourrepeat_test_mode){
        run_repeat_test(prune_selftest_b, prune_selftest_levels, g_prune_repeat_n);
        return 0;
    }

    if(prune_selftest_mode){

        const char *default_pub = "03100611c54dfef604163b8358f7b7fac13ce478e02cb224ae16d45526b25d9d4d";
        const char *use_pub = start_pubkey_hex ? start_pubkey_hex : default_pub;
        unsigned char root_pub[PUBKEY_SIZE];
        if(!parse_pubkey(use_pub, root_pub)) die("invalid pubkey for --prune-selftest");
        Point root;
        if(!pub_to_point(secp, root_pub, root)) die("could not decode pubkey for --prune-selftest");
        if(prune_selftest_levels<1 || prune_selftest_levels>64)
            die("--prune-levels must be between 1 and 64");
        TreeGenContext ctx;
        tree_gen_init(secp, ctx, root, prune_selftest_levels);
        run_prune_self_test(secp, ctx, prune_selftest_b, prune_selftest_levels);
        return 0;
    }

    if(gen_pruned_leaves_mode){
        if(!start_pubkey_hex) die("--gen-pruned-leaves requires --start-pubkey HEX");
        if(tree_steps<0) die("--gen-pruned-leaves requires --tree-steps N");
        unsigned char root_pub[PUBKEY_SIZE];
        if(!parse_pubkey(start_pubkey_hex, root_pub)) die("invalid --start-pubkey");
        Point root;
        if(!pub_to_point(secp, root_pub, root)) die("could not decode --start-pubkey");
        TreeGenContext ctx;
        tree_gen_init(secp, ctx, root, tree_steps);
        run_gen_pruned_leaves(secp, ctx, batch_size, threads, random_leaves!=0, gpl_start_b,
                               output_leaves_file, preview_count);
        return 0;
    }

    if(build_mode){
        if(!hier_table_file) die("--build requires --hier-table FILE");
        uint64_t M = (uint64_t)1<<m_bits;
        HierState st;
        build_hierarchy(secp, st, M, threads, gen_batch, fp_rate, bloom1_k, bloom1_block_bits);
        save_hierarchy(st, hier_table_file);
        free_hierarchy(st);
        return 0;
    }

    if(search_mode){
        if(!hier_table_file) die("--search requires --hier-table FILE");
        if(!target_bloom_file) die("--search requires --target-bloom FILE");
        if(tree_steps<0) die("--tree-steps is required");
        HierState st;
        if(!load_hierarchy(secp, st, hier_table_file)) die("could not load --hier-table (use --build first)");
        PrecomputedTargets *tg = load_targets(target_bloom_file);
        int a_bits = root_bits - tree_steps;
        run_hier_search(secp, st, *tg, tree_steps, a_bits, threads, gen_batch, restrict_upper_half!=0, dump_round_stats_path);
        free_hierarchy(st);
        return 0;
    }

    if(attack_loop_mode){
        if(!hier_table_file) die("--attack-loop requires --hier-table FILE");
        if(!start_pubkey_hex) die("--attack-loop requires --start-pubkey HEX");
        if(tree_steps<0) die("--attack-loop requires --tree-steps N");

        unsigned char root_pub[PUBKEY_SIZE];
        if(!parse_pubkey(start_pubkey_hex, root_pub)) die("invalid --start-pubkey (66 hex chars, 02/03)");
        Point root;
        if(!pub_to_point(secp, root_pub, root)) die("could not decode --start-pubkey");

        HierState st;
        if(!load_hierarchy(secp, st, hier_table_file)) die("could not load --hier-table (use --build first)");

        int a_bits = root_bits - tree_steps;
        if(a_bits<1 || a_bits>45) die("root_bits - tree_steps must be between 1 and 45 (practical hierarchy limit)");

        TreeGenContext ctx;
        tree_gen_init(secp, ctx, root, tree_steps);

        printf("[+] Attack-loop mode: tree_steps=%d, batch=%" PRIu64 ", mode=%s%s\n",
               tree_steps, batch_size, random_leaves?"random":"sequential",
               max_batches>0?"":" (no batch limit - Ctrl+C to stop)");

        uint64_t batch_num = 0;
        bool found_final = false;

        int seq_block_bits = -1;
        if(!random_leaves && g_unsafe_prune && batch_size>0 && (batch_size&(batch_size-1))==0){
            seq_block_bits = 0; uint64_t bs=batch_size; while(bs>1){ bs>>=1; seq_block_bits++; }
        }
        Int seq_offset; seq_offset.SetInt64(0);
        Int seq_batch_size_i; seq_batch_size_i.SetInt64((int64_t)batch_size);
        uint64_t seq_blocks_skipped = 0;
        double seq_last_progress = now_seconds();
        while(true){
            batch_num++;
            if(seq_block_bits>=0){
                uint64_t jump = block_fully_excluded_jump(seq_offset, tree_steps, g_prune_repeat_n, seq_block_bits);
                if(jump>0){
                    seq_blocks_skipped += jump;
                    Int jump_amount; jump_amount.Set(&seq_batch_size_i); jump_amount.Mult(jump);
                    seq_offset.Add(&jump_amount);
                    if(jump>1) batch_num += (jump-1);
                    double now = now_seconds();
                    if(now-seq_last_progress>=2.0){
                        printf("[+] ...skipping fully-excluded batches: %" PRIu64 " so far (batch #%"
                               PRIu64 " now), offset=0x%s\n", seq_blocks_skipped, batch_num,
                               seq_offset.GetBase16());
                        seq_last_progress = now;
                    }
                    if(max_batches>0 && batch_num>=max_batches){
                        printf("[!] Limit of %" PRIu64 " batches reached without finding a match "
                               "(%" PRIu64 " fully-excluded, skipped without generating leaves).\n",
                               max_batches, seq_blocks_skipped);
                        break;
                    }
                    continue;
                }
            }
            double tgen0 = now_seconds();
            std::vector<LeafExact> leaves;
            std::vector<Int> b_full;
            if(random_leaves){
                gen_random_leaf_batch(secp, ctx, batch_size, leaves, b_full);
            } else {
                TreeGenContext ctx_off = ctx;
                if(!seq_offset.IsZero()){

                    Point new_R = tree_gen_leaf_at(secp, ctx, seq_offset);
                    ctx_off.R = new_R;
                }
                gen_sequential_leaf_batch(secp, ctx_off, batch_size, threads, leaves, &seq_offset);

                for(auto &le : leaves){
                    Int lb; lb.SetInt64((int64_t)le.b); lb.Add(&seq_offset);
                    le.b = lb.bits64[0];
                }
                seq_offset.Add(&seq_batch_size_i);
            }
            double tgen1 = now_seconds();
            printf("[+] Batch #%" PRIu64 ": %" PRIu64 " leaves generated in %.2fs\n",
                   batch_num, (uint64_t)leaves.size(), tgen1-tgen0);

            PrecomputedTargets tg;
            tg.n_leaves = leaves.size();
            tg.exact.reserve(leaves.size()*2);
            for(auto &le : leaves){
                unsigned char xb[32]; memcpy(xb, le.pub+1, 32);
                uint64_t fp = fingerprint_32(xb);
                tg.exact[fp] = le;
            }

            bool hit = run_hier_search(secp, st, tg, tree_steps, a_bits, threads, gen_batch, restrict_upper_half!=0, dump_round_stats_path);
            if(hit){
                found_final = true;
                if(random_leaves){

                    if(tree_steps>64){
                        printf("[!] WARNING: tree_steps>64 in random mode - the 'b' printed above\n");
                        printf("    is TRUNCATED to 64 bits. You need to correlate against b_full[]\n");
                        printf("    by fingerprint to get the full b.\n");
                    }
                }
                break;
            }
            if(max_batches>0 && batch_num>=max_batches){
                printf("[!] Limit of %" PRIu64 " batches reached without finding a match.\n", max_batches);
                break;
            }
        }
        free_hierarchy(st);
        return found_final ? 0 : 1;
    }

    if(block_search_mode){
        if(!hier_table_file) die("--block-search requires --hier-table FILE");
        if(!start_pubkey_hex) die("--block-search requires --start-pubkey HEX");
        if(tree_steps<0) die("--block-search requires --tree-steps N");

        unsigned char root_pub[PUBKEY_SIZE];
        if(!parse_pubkey(start_pubkey_hex, root_pub)) die("invalid --start-pubkey (66 hex chars, 02/03)");
        Point root;
        if(!pub_to_point(secp, root_pub, root)) die("could not decode --start-pubkey");

        HierState st;
        if(!load_hierarchy(secp, st, hier_table_file)) die("could not load --hier-table (use --build first)");

        int a_bits = root_bits - tree_steps;
        if(a_bits<1 || a_bits>45) die("root_bits - tree_steps must be between 1 and 45 (practical hierarchy limit)");

        TreeGenContext ctx;
        tree_gen_init(secp, ctx, root, tree_steps);

        printf("[+] Block-search mode: tree_steps=%d, block_size=%" PRIu64
               ", rounds_per_block=%s%s\n",
               tree_steps, batch_size,
               rounds_per_block==UINT64_MAX?"unbounded (full search per block)":std::to_string(rounds_per_block).c_str(),
               max_batches>0?"":" (no block limit - Ctrl+C to stop)");

        int block_bits = -1;
        if(g_unsafe_prune && batch_size>0 && (batch_size&(batch_size-1))==0){
            block_bits = 0; uint64_t bs=batch_size; while(bs>1){ bs>>=1; block_bits++; }
        }

        Int block_offset; block_offset.SetInt64(0);
        Int block_size_i; block_size_i.SetInt64((int64_t)batch_size);
        uint64_t block_num = 0;
        uint64_t blocks_skipped_fully_excluded = 0;
        bool found_final = false;
        double last_progress_print = now_seconds();

        while(true){
            sat_add_u64(block_num, 1);
            uint64_t jump = block_bits>=0 ? block_fully_excluded_jump(block_offset, tree_steps, g_prune_repeat_n, block_bits) : 0;
            if(jump>0){
                sat_add_u64(blocks_skipped_fully_excluded, jump);
                Int jump_amount; jump_amount.Set(&block_size_i); jump_amount.Mult(jump);
                block_offset.Add(&jump_amount);
                if(jump>1) sat_add_u64(block_num, jump-1);
                double now = now_seconds();
                if(now-last_progress_print>=2.0){
                    printf("[+] ...skipping fully-excluded blocks: %" PRIu64 " so far (block #%"
                           PRIu64 " now, last jump=%" PRIu64 "), offset=0x%s\n",
                           blocks_skipped_fully_excluded, block_num, jump, block_offset.GetBase16());
                    last_progress_print = now;
                }
                if(max_batches>0 && block_num>=max_batches){
                    printf("[!] Limit of %" PRIu64 " blocks reached without finding a match "
                           "(%" PRIu64 " fully-excluded, skipped without generating leaves).\n",
                           max_batches, blocks_skipped_fully_excluded);
                    break;
                }
                continue;
            }
            double tgen0 = now_seconds();

            TreeGenContext ctx_off = ctx;
            if(!block_offset.IsZero()){
                Point new_R = tree_gen_leaf_at(secp, ctx, block_offset);
                ctx_off.R = new_R;
            }
            std::vector<LeafExact> leaves;
            gen_sequential_leaf_batch(secp, ctx_off, batch_size, threads, leaves, &block_offset);

            double tgen1 = now_seconds();
            printf("[+] Block #%" PRIu64 ": %" PRIu64 " leaves generated in %.2fs\n",
                   block_num, (uint64_t)leaves.size(), tgen1-tgen0);

            if(leaves.empty()){

                block_offset.Add(&block_size_i);
                if(max_batches>0 && block_num>=max_batches){
                    printf("[!] Limit of %" PRIu64 " blocks reached without finding a match "
                           "(%" PRIu64 " fully-excluded, skipped without generating leaves).\n",
                           max_batches, blocks_skipped_fully_excluded);
                    break;
                }
                continue;
            }

            PrecomputedTargets tg;
            tg.n_leaves = leaves.size();
            tg.exact.reserve(leaves.size()*2);
            for(auto &le : leaves){
                unsigned char xb[32]; memcpy(xb, le.pub+1, 32);
                uint64_t fp = fingerprint_32(xb);
                tg.exact[fp] = le;
            }

            uint64_t out_A=0, out_b_local=0;
            bool hit = run_hier_search(secp, st, tg, tree_steps, a_bits, threads, gen_batch,
                                        restrict_upper_half!=0, dump_round_stats_path,
                                        UINT64_MAX, UINT64_MAX, nullptr,
                                        &out_A, &out_b_local, rounds_per_block);
            if(hit){
                found_final = true;
                Int full_b; full_b.SetInt64((int64_t)out_b_local);
                full_b.Add(&block_offset);
                Int Ai; Ai.SetInt64((int64_t)out_A);
                Int two; two.SetInt64(2);
                Int pow2steps; pow2steps.SetInt64(1);
                for(int i=0;i<tree_steps;++i) pow2steps.Mult(&two);
                Int K; K.Set(&Ai); K.Mult(&pow2steps);
                K.Add(&full_b);
                printf("[+] FOUND in block #%" PRIu64 "\n", block_num);
                printf("    b (full) = %s\n", full_b.GetBase16());
                printf("    A = %s\n", Ai.GetBase16());
                printf("    K = A*2^%d + b = %s\n", tree_steps, K.GetBase16());
                break;
            }
            block_offset.Add(&block_size_i);
            if(max_batches>0 && block_num>=max_batches){
                printf("[!] Limit of %" PRIu64 " blocks reached without finding a match "
                       "(%" PRIu64 " fully-excluded, skipped without generating leaves).\n",
                       max_batches, blocks_skipped_fully_excluded);
                break;
            }
        }
        free_hierarchy(st);
        return found_final ? 0 : 1;
    }

    if(radix4_test_mode){
        if(!start_pubkey_hex) die("--radix4-test requires --start-pubkey HEX");
        if(tree_steps<0) die("--radix4-test requires --tree-steps N");

        unsigned char root_pub[PUBKEY_SIZE];
        if(!parse_pubkey(start_pubkey_hex, root_pub)) die("invalid --start-pubkey (66 hex chars, 02/03)");
        Point root;
        if(!pub_to_point(secp, root_pub, root)) die("could not decode --start-pubkey");

        int a_bits = root_bits - tree_steps;
        if(a_bits<1 || a_bits>45) die("root_bits - tree_steps must be between 1 and 45");

        uint64_t M = (uint64_t)1<<m_bits;
        printf("[+] --radix4-test: building experimental radix-4/5-level hierarchy (M=2^%d)\n", m_bits);
        HierState st;
        build_hierarchy_radix4(secp, st, M, threads, gen_batch, fp_rate, bloom1_k, bloom1_block_bits);

        TreeGenContext ctx;
        tree_gen_init(secp, ctx, root, tree_steps);

        printf("[+] Radix4-test mode: tree_steps=%d, batch=%" PRIu64 ", mode=%s\n",
               tree_steps, batch_size, random_leaves?"random":"sequential");

        std::vector<LeafExact> leaves;
        std::vector<Int> b_full;
        if(random_leaves) gen_random_leaf_batch(secp, ctx, batch_size, leaves, b_full);
        else gen_sequential_leaf_batch(secp, ctx, batch_size, threads, leaves);
        printf("[+] %" PRIu64 " leaves generated\n", (uint64_t)leaves.size());

        PrecomputedTargets tg;
        tg.n_leaves = leaves.size();
        tg.exact.reserve(leaves.size()*2);
        for(auto &le : leaves){
            unsigned char xb[32]; memcpy(xb, le.pub+1, 32);
            uint64_t fp = fingerprint_32(xb);
            tg.exact[fp] = le;
        }

        bool hit = run_hier_search(secp, st, tg, tree_steps, a_bits, threads, gen_batch,
                                    restrict_upper_half!=0, dump_round_stats_path);
        free_hierarchy_radix4(st);
        return hit ? 0 : 1;
    }

    usage(*argv);
    return 1;
}
