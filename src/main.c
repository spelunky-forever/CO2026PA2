/**
 * @file main.c
 * @brief Student implementation of mel spectrogram core functions.
 *
 * Implement the three functions below. The remaining pipeline functions
 * (hann_window, stft, melspectrogram) are provided in utils.c.
 *
 * Python source of truth: scripts/mel_spectrogram.py
 * Include RVV intrinsics via: #include <riscv_vector.h>
 */

#include "mel_spectrogram.h"
#include <riscv_vector.h>
#include <math.h>
#define PI 3.14159265358979323846


/* RVV hint: each butterfly stage is data-parallel across independent pairs. */
void fft(float *__restrict real, float *__restrict imag, size_t n) {
    // ==========================================
    // 階段一：位元反轉 (Bit-Reversal) - 用 Scalar 寫即可
    // ==========================================
    size_t j = 0;
    for (size_t i = 0; i < n - 1; i++) {
        if (i < j) {
            // 交換 real[i] 和 real[j]
            float temp_r = real[i]; real[i] = real[j]; real[j] = temp_r;
            // 交換 imag[i] 和 imag[j]
            float temp_i = imag[i]; imag[i] = imag[j]; imag[j] = temp_i;
        }
        size_t k = n >> 1;
        while (k <= j) {
            j -= k;
            k >>= 1;
        }
        j += k;
    }

    // ==========================================
    // 階段二：蝴蝶運算 (使用 RVV 加速)
    // ==========================================
    for (size_t step = 2; step <= n; step *= 2) {
        size_t half_step = step / 2;
        for (size_t j = 0; j < half_step; j++) {
            // 1. 用 double 精確計算角度，消除誤差放大效應
            double angle = -2.0 * PI * (double)j / (double)step;
            float W_re = (float)cos(angle);
            float W_im = (float)sin(angle);

            // 算出這個內迴圈總共要處理幾隻蝴蝶
            size_t elements_to_process = n / step; 
            size_t i_start = j;

            // 2. 開始 RVV 向量化迴圈
            while (elements_to_process > 0) {
                // 設定向量長度 (vl)
                size_t vl = __riscv_vsetvl_e32m4(elements_to_process);

                // 計算跨距 (Stride in bytes)
                ptrdiff_t stride = step * sizeof(float);

                // 3. 跨步載入資料 (A 和 B)
                // A 的位址是 &real[i_start], B 的位址是 &real[i_start + half_step]
                vfloat32m4_t v_A_re = __riscv_vlse32_v_f32m4(&real[i_start], stride, vl);
                vfloat32m4_t v_A_im = __riscv_vlse32_v_f32m4(&imag[i_start], stride, vl);
                vfloat32m4_t v_B_re = __riscv_vlse32_v_f32m4(&real[i_start + half_step], stride, vl);
                vfloat32m4_t v_B_im = __riscv_vlse32_v_f32m4(&imag[i_start + half_step], stride, vl);

                // 4. 將純量 W_re 和 W_im 轉成向量
                vfloat32m4_t v_W_re = __riscv_vfmv_v_f_f32m4(W_re, vl);
                vfloat32m4_t v_W_im = __riscv_vfmv_v_f_f32m4(W_im, vl);

                // 5. 執行複數乘法：T = W * B
                // T_re = W_re * B_re - W_im * B_im
                // T_im = W_re * B_im + W_im * B_re
                vfloat32m4_t v_T_re = __riscv_vfsub_vv_f32m4(
                    __riscv_vfmul_vv_f32m4(v_W_re, v_B_re, vl),
                    __riscv_vfmul_vv_f32m4(v_W_im, v_B_im, vl), vl);
                
                vfloat32m4_t v_T_im = __riscv_vfadd_vv_f32m4(
                    __riscv_vfmul_vv_f32m4(v_W_re, v_B_im, vl),
                    __riscv_vfmul_vv_f32m4(v_W_im, v_B_re, vl), vl);

                // 6. 計算新的 B = A - T (必須先算 B，因為 A 的值等下會被蓋掉)
                vfloat32m4_t v_new_B_re = __riscv_vfsub_vv_f32m4(v_A_re, v_T_re, vl);
                vfloat32m4_t v_new_B_im = __riscv_vfsub_vv_f32m4(v_A_im, v_T_im, vl);

                // 7. 計算新的 A = A + T
                vfloat32m4_t v_new_A_re = __riscv_vfadd_vv_f32m4(v_A_re, v_T_re, vl);
                vfloat32m4_t v_new_A_im = __riscv_vfadd_vv_f32m4(v_A_im, v_T_im, vl);

                // 8. 跨步儲存回記憶體
                __riscv_vsse32_v_f32m4(&real[i_start], stride, v_new_A_re, vl);
                __riscv_vsse32_v_f32m4(&imag[i_start], stride, v_new_A_im, vl);
                __riscv_vsse32_v_f32m4(&real[i_start + half_step], stride, v_new_B_re, vl);
                __riscv_vsse32_v_f32m4(&imag[i_start + half_step], stride, v_new_B_im, vl);

                // 更新下一波迴圈的狀態
                elements_to_process -= vl;
                i_start += vl * step; // 注意：移動的距離是處理的數量 * 每一步的跨距
            }
        }
    }
}

/* RVV hint: vlse32 with stride=8 bytes extracts all re (or im) values in one pass. */
void power_spectrum(const float *__restrict stft_data, size_t num_frames,
                    float *__restrict output) {
    size_t n = num_frames * N_FREQ_BINS; 
    const float *in = stft_data;
    float *out = output;
    ptrdiff_t stride = 8; // 2 * sizeof(float)
    
    // 取得 m8 所能支援的最大向量長度
    size_t vlmax = __riscv_vsetvlmax_e32m8();
    
    // 終極優化：LMUL=8 搭配雙路展開 (Unroll by 2)，精準使用 32 個向量暫存器
    while (n >= 2 * vlmax) {
        // 1. 載入實部並立即平方，這樣做能讓編譯器有最佳的暫存器分配
        vfloat32m8_t r1 = __riscv_vlse32_v_f32m8(in, stride, vlmax);
        vfloat32m8_t r2 = __riscv_vlse32_v_f32m8(in + 2 * vlmax, stride, vlmax);
        
        r1 = __riscv_vfmul_vv_f32m8(r1, r1, vlmax);
        r2 = __riscv_vfmul_vv_f32m8(r2, r2, vlmax);
        
        // 2. 在硬體算乘法時，利用記憶體空窗期去載入虛部
        vfloat32m8_t i1 = __riscv_vlse32_v_f32m8(in + 1, stride, vlmax);
        vfloat32m8_t i2 = __riscv_vlse32_v_f32m8(in + 2 * vlmax + 1, stride, vlmax);
        
        // 3. 融合乘加計算功率 (r = r + i * i)
        r1 = __riscv_vfmacc_vv_f32m8(r1, i1, i1, vlmax);
        r2 = __riscv_vfmacc_vv_f32m8(r2, i2, i2, vlmax);
        
        // 4. 寫回結果
        __riscv_vse32_v_f32m8(out, r1, vlmax);
        __riscv_vse32_v_f32m8(out + vlmax, r2, vlmax);
        
        in += 4 * vlmax;
        out += 2 * vlmax;
        n -= 2 * vlmax;
    }
    
    // 處理尾巴剩下的資料
    while (n > 0) {
        size_t vl = __riscv_vsetvl_e32m8(n);
        
        vfloat32m8_t r = __riscv_vlse32_v_f32m8(in, stride, vl);
        r = __riscv_vfmul_vv_f32m8(r, r, vl);
        
        vfloat32m8_t i = __riscv_vlse32_v_f32m8(in + 1, stride, vl);
        r = __riscv_vfmacc_vv_f32m8(r, i, i, vl);
        
        __riscv_vse32_v_f32m8(out, r, vl);
        
        in += 2 * vl;
        out += vl;
        n -= vl;
    }
}

/* RVV hint: vfmul + vfredusum computes one dot product per (frame, mel) pair. */
void mel_filter_bank(const float *__restrict power,
                     const float *__restrict mel_bank, size_t num_frames,
                     size_t n_mels, size_t n_freq_bins,
                     float *__restrict output) {

    for (size_t f = 0; f < num_frames; f++) {
        size_t m = 0;
        
        // 1. 迴圈展開 (Loop Unrolling)：一次計算 4 個 mel bands
        for (; m + 3 < n_mels; m += 4) {
            // 使用純量來累加，打破 vfredusum 向量操作的管線相依性
            float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
            
            size_t remaining = n_freq_bins;
            size_t k = 0;
            
            while (remaining > 0) {
                size_t vl = __riscv_vsetvl_e32m4(remaining);
                
                // 2. 關鍵優化：power 陣列只需要載入 1 次！
                vfloat32m4_t v_p = __riscv_vle32_v_f32m4(&power[f * n_freq_bins + k], vl);
                
                // 分別載入 4 列的 mel_bank
                vfloat32m4_t v_m0 = __riscv_vle32_v_f32m4(&mel_bank[(m + 0) * n_freq_bins + k], vl);
                vfloat32m4_t v_m1 = __riscv_vle32_v_f32m4(&mel_bank[(m + 1) * n_freq_bins + k], vl);
                vfloat32m4_t v_m2 = __riscv_vle32_v_f32m4(&mel_bank[(m + 2) * n_freq_bins + k], vl);
                vfloat32m4_t v_m3 = __riscv_vle32_v_f32m4(&mel_bank[(m + 3) * n_freq_bins + k], vl);
                
                // 各自相乘
                vfloat32m4_t v_prod0 = __riscv_vfmul_vv_f32m4(v_p, v_m0, vl);
                vfloat32m4_t v_prod1 = __riscv_vfmul_vv_f32m4(v_p, v_m1, vl);
                vfloat32m4_t v_prod2 = __riscv_vfmul_vv_f32m4(v_p, v_m2, vl);
                vfloat32m4_t v_prod3 = __riscv_vfmul_vv_f32m4(v_p, v_m3, vl);
                
                // 3. 歸約相加：每次都以一個乾淨的 0.0f 向量作為基底
                vfloat32m1_t v_zero = __riscv_vfmv_v_f_f32m1(0.0f, 1);
                
                // 將結果抽出成純量累加
                sum0 += __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(v_prod0, v_zero, vl));
                sum1 += __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(v_prod1, v_zero, vl));
                sum2 += __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(v_prod2, v_zero, vl));
                sum3 += __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(v_prod3, v_zero, vl));
                
                remaining -= vl;
                k += vl;
            }
            
            // 寫回 output
            output[f * n_mels + m + 0] = sum0;
            output[f * n_mels + m + 1] = sum1;
            output[f * n_mels + m + 2] = sum2;
            output[f * n_mels + m + 3] = sum3;
        }
        
        // 4. 處理剩餘的 mels (尾端處理)
        for (; m < n_mels; m++) {
            float sum = 0.0f;
            size_t remaining = n_freq_bins;
            size_t k = 0;
            while (remaining > 0) {
                size_t vl = __riscv_vsetvl_e32m4(remaining);
                vfloat32m4_t v_p = __riscv_vle32_v_f32m4(&power[f * n_freq_bins + k], vl);
                vfloat32m4_t v_m = __riscv_vle32_v_f32m4(&mel_bank[m * n_freq_bins + k], vl);
                vfloat32m4_t v_prod = __riscv_vfmul_vv_f32m4(v_p, v_m, vl);
                
                vfloat32m1_t v_zero = __riscv_vfmv_v_f_f32m1(0.0f, 1);
                sum += __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(v_prod, v_zero, vl));
                
                remaining -= vl;
                k += vl;
            }
            output[f * n_mels + m] = sum;
        }
    }
}