
#ifndef GPU_HIERARCHY_H
#define GPU_HIERARCHY_H

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct GpuBloom GpuBloom;

GpuBloom* gpu_bloom_upload(const uint64_t *host_bits, uint64_t bit_count,
                           uint64_t nwords, uint32_t k_hashes, uint64_t block_bits);

void gpu_bloom_free(GpuBloom *b);

int gpu_bloom_check_single(GpuBloom *b, uint64_t fingerprint);

void gpu_query_device_info(char *out_buf, int buflen);

int gpu_query_device_count();

int gpu_set_device(int device_id);

void gpu_query_device_info_n(int device_id, char *out_buf, int buflen);

typedef struct { uint64_t x[4], y[4]; } GpuECPoint;

typedef struct GpuShiftTable GpuShiftTable;

GpuShiftTable* gpu_shift_table_upload(const GpuECPoint *host_points, int count);
void gpu_shift_table_free(GpuShiftTable *t);

int gpu_i2loop_check_single(GpuShiftTable *shift_table, GpuBloom *bloom2, const GpuECPoint *R);

typedef struct { uint64_t fp; uint32_t idx; } GpuBabyEntry;

typedef struct GpuBabyTable GpuBabyTable;

GpuBabyTable* gpu_baby_table_upload(const GpuBabyEntry *host_table, uint64_t array_size, int K);
void gpu_baby_table_free(GpuBabyTable *t);

int gpu_i3loop_and_exact_check_single(GpuShiftTable *shift_table, GpuBloom *bloom3,
                                       GpuBabyTable *baby_table, const GpuECPoint *R,
                                       int *out_found_i3);

void gpu_batch_hierarchical_search(GpuBloom *bloom1, GpuShiftTable *shift2, GpuBloom *bloom2,
                                    GpuShiftTable *shift3, GpuBloom *bloom3, GpuBabyTable *baby_table,
                                    const GpuECPoint *R_points, int n_points,
                                    int *out_found_i2, int *out_found_i3, int *out_found_idx);

void gpu_batch_advance(GpuECPoint *points, int n_points, const GpuECPoint *shift);

typedef struct GpuHierarchy GpuHierarchy;

GpuHierarchy* gpu_hierarchy_build(
    const uint64_t *bloom1_bits, uint64_t bloom1_bitcount, uint64_t bloom1_nwords, uint32_t bloom1_k, uint64_t bloom1_block_bits,
    const uint64_t *bloom2_bits, uint64_t bloom2_bitcount, uint64_t bloom2_nwords, uint32_t bloom2_k, uint64_t bloom2_block_bits,
    const uint64_t *bloom3_bits, uint64_t bloom3_bitcount, uint64_t bloom3_nwords, uint32_t bloom3_k, uint64_t bloom3_block_bits,
    const GpuECPoint *shift2_32, const GpuECPoint *shift3_32,
    const void *baby_table_raw, uint64_t baby_array_size, int baby_K,
    const GpuECPoint *giant_step_point);

void gpu_hierarchy_free(GpuHierarchy *h);

int gpu_hierarchy_search_block(GpuHierarchy *h, const GpuECPoint *lower_start, const GpuECPoint *upper_start,
                                int n_leaves, uint64_t half_giant,
                                int *out_is_upper_front, uint64_t *out_round,
                                int *out_i2, int *out_i3, int *out_idx);

int gpu_range_scan_chunk(GpuHierarchy *h, GpuECPoint *lane_points, int n_lanes,
                          const GpuECPoint *stride_shift, int K,
                          int *out_round, int *out_i2, int *out_i3, int *out_idx);

#ifdef __cplusplus
}
#endif

#endif
