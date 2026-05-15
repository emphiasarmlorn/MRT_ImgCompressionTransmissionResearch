/*
 * compress_pipeline.cpp
 * Full implementation of: motion compensation → DCT → quantization
 *
 * Compile: g++ -O2 -std=c++17 -o pipeline compress_pipeline.cpp \
 *               $(pkg-config --cflags --libs opencv4) -lm
 */

#include <opencv2/opencv.hpp>
#include <cmath>
#include <cstdint>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <limits>

// ══════════════════════════════════════════════════════════════════════════════
// SECTION 0 — CONSTANTS
// ══════════════════════════════════════════════════════════════════════════════

static const int BLOCK  = 8;
static const int SEARCH = 8;

static const float Q_LUMA[8][8] = {
    {16, 11, 10, 16,  24,  40,  51,  61},
    {12, 12, 14, 19,  26,  58,  60,  55},
    {14, 13, 16, 24,  40,  57,  69,  56},
    {14, 17, 22, 29,  51,  87,  80,  62},
    {18, 22, 37, 56,  68, 109, 103,  77},
    {24, 35, 55, 64,  81, 104, 113,  92},
    {49, 64, 78, 87, 103, 121, 120, 101},
    {72, 92, 95, 98, 112, 100, 103,  99},
};

static const int ZIGZAG[64] = {
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
};

// ══════════════════════════════════════════════════════════════════════════════
// SECTION 1 — TYPES
// ══════════════════════════════════════════════════════════════════════════════

typedef float   BlockF[8][8];
typedef int16_t BlockI[8][8];

struct MotionVector { int8_t dy, dx; };
typedef std::vector<std::vector<MotionVector>> MvField;

struct Stats {
    int   total_coefs;
    int   nonzero;
    float sparsity_pct;
};

// ══════════════════════════════════════════════════════════════════════════════
// SECTION 2 — Q MATRIX SCALING
// ══════════════════════════════════════════════════════════════════════════════

void scaled_q(int quality, float q_out[8][8])
{
    float scale = (quality < 50)
                ? (50.0f / quality)
                : (2.0f - quality / 50.0f);
    for (int u = 0; u < 8; u++)
        for (int v = 0; v < 8; v++)
            q_out[u][v] = std::clamp(std::round(Q_LUMA[u][v] * scale),
                                     1.0f, 255.0f);
}

// ══════════════════════════════════════════════════════════════════════════════
// SECTION 3 — DCT AND IDCT
// ══════════════════════════════════════════════════════════════════════════════

void dct2d(const float in[8][8], float out[8][8])
{
    cv::Mat src(8, 8, CV_32F);
    for (int y = 0; y < 8; y++)
        for (int x = 0; x < 8; x++)
            src.at<float>(y, x) = in[y][x];

    cv::Mat dst;
    cv::dct(src, dst);

    for (int y = 0; y < 8; y++)
        for (int x = 0; x < 8; x++)
            out[y][x] = dst.at<float>(y, x);
}

void idct2d(const float in[8][8], float out[8][8])
{
    cv::Mat src(8, 8, CV_32F);
    for (int y = 0; y < 8; y++)
        for (int x = 0; x < 8; x++)
            src.at<float>(y, x) = in[y][x];

    cv::Mat dst;
    cv::idct(src, dst);

    for (int y = 0; y < 8; y++)
        for (int x = 0; x < 8; x++)
            out[y][x] = dst.at<float>(y, x);
}

// ══════════════════════════════════════════════════════════════════════════════
// SECTION 4 — QUANTIZATION, DEQUANTIZATION, ZIGZAG
// ══════════════════════════════════════════════════════════════════════════════

void quantize(const float dct[8][8], const float q[8][8], int16_t out[8][8])
{
    for (int u = 0; u < 8; u++)
        for (int v = 0; v < 8; v++)
            out[u][v] = static_cast<int16_t>(std::round(dct[u][v] / q[u][v]));
}

void dequantize(const int16_t in[8][8], const float q[8][8], float out[8][8])
{
    for (int u = 0; u < 8; u++)
        for (int v = 0; v < 8; v++)
            out[u][v] = static_cast<float>(in[u][v]) * q[u][v];
}

void zigzag_scan(const int16_t in[8][8], int16_t out[64])
{
    const int16_t* flat = reinterpret_cast<const int16_t*>(in);
    for (int k = 0; k < 64; k++)
        out[k] = flat[ZIGZAG[k]];
}

// ══════════════════════════════════════════════════════════════════════════════
// SECTION 5 — MOTION ESTIMATION
// ══════════════════════════════════════════════════════════════════════════════

void motion_estimate_frame(const cv::Mat& ref, const cv::Mat& cur,
                           MvField& mv, cv::Mat& residual)
{
    const int H  = cur.rows, W  = cur.cols;
    const int bH = H / BLOCK, bW = W / BLOCK;

    mv.assign(bH, std::vector<MotionVector>(bW, {0, 0}));
    residual = cv::Mat::zeros(H, W, CV_32F);

    cv::Mat ref_pad;
    cv::copyMakeBorder(ref, ref_pad, SEARCH, SEARCH, SEARCH, SEARCH,
                       cv::BORDER_REPLICATE);

    for (int by = 0; by < bH; by++) {
        for (int bx = 0; bx < bW; bx++) {
            const int cy = by * BLOCK, cx = bx * BLOCK;

            cv::Mat cur_block;
            cur(cv::Rect(cx, cy, BLOCK, BLOCK)).convertTo(cur_block, CV_32F);

            float best_sad = std::numeric_limits<float>::max();
            int   best_dy = 0, best_dx = 0;

            for (int dy = -SEARCH; dy <= SEARCH; dy++) {
                for (int dx = -SEARCH; dx <= SEARCH; dx++) {
                    const int ry = cy + SEARCH + dy;
                    const int rx = cx + SEARCH + dx;

                    cv::Mat ref_block;
                    ref_pad(cv::Rect(rx, ry, BLOCK, BLOCK))
                        .convertTo(ref_block, CV_32F);

                    cv::Mat diff;
                    cv::absdiff(cur_block, ref_block, diff);
                    float sad = static_cast<float>(cv::sum(diff)[0]);

                    if (sad < best_sad) {
                        best_sad = sad;
                        best_dy  = dy;
                        best_dx  = dx;
                    }
                }
            }

            mv[by][bx] = { static_cast<int8_t>(best_dy),
                           static_cast<int8_t>(best_dx) };

            const int ry = cy + SEARCH + best_dy;
            const int rx = cx + SEARCH + best_dx;
            cv::Mat ref_block;
            ref_pad(cv::Rect(rx, ry, BLOCK, BLOCK)).convertTo(ref_block, CV_32F);

            cv::Mat res_roi = residual(cv::Rect(cx, cy, BLOCK, BLOCK));
            cv::subtract(cur_block, ref_block, res_roi);
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// SECTION 6 — ENCODE ONE FRAME
// ══════════════════════════════════════════════════════════════════════════════

Stats encode_frame(const cv::Mat& residual, int quality,
                   cv::Mat& quant_frame,
                   std::vector<std::vector<int16_t>>& zz_blocks)
{
    const int H  = residual.rows, W  = residual.cols;
    const int bH = H / BLOCK,     bW = W / BLOCK;

    float q[8][8];
    scaled_q(quality, q);

    quant_frame = cv::Mat::zeros(H, W, CV_16S);
    zz_blocks.clear();
    zz_blocks.reserve(bH * bW);
    int nonzero = 0;

    for (int by = 0; by < bH; by++) {
        for (int bx = 0; bx < bW; bx++) {
            const int cy = by * BLOCK, cx = bx * BLOCK;

            // extract residual block into BlockF
            BlockF res_block;
            cv::Mat roi = residual(cv::Rect(cx, cy, BLOCK, BLOCK));
            for (int y = 0; y < 8; y++)
                for (int x = 0; x < 8; x++)
                    res_block[y][x] = roi.at<float>(y, x);

            // DCT
            BlockF dct_block;
            dct2d(res_block, dct_block);

            // quantize
            BlockI q_block;
            quantize(dct_block, q, q_block);

            // write into quant_frame
            cv::Mat qroi = quant_frame(cv::Rect(cx, cy, BLOCK, BLOCK));
            for (int y = 0; y < 8; y++)
                for (int x = 0; x < 8; x++)
                    qroi.at<int16_t>(y, x) = q_block[y][x];

            // zigzag
            int16_t zz[64];
            zigzag_scan(q_block, zz);
            zz_blocks.push_back(std::vector<int16_t>(zz, zz + 64));

            // count nonzero
            for (int k = 0; k < 64; k++)
                if (zz[k] != 0) nonzero++;
        }
    }

    Stats s;
    s.total_coefs  = bH * bW * 64;
    s.nonzero      = nonzero;
    s.sparsity_pct = 100.0f * (1.0f - static_cast<float>(nonzero) / s.total_coefs);
    return s;
}

// ══════════════════════════════════════════════════════════════════════════════
// SECTION 7 — DECODE ONE FRAME
// ══════════════════════════════════════════════════════════════════════════════

void decode_frame(const cv::Mat& quant_frame, int quality, cv::Mat& recon)
{
    const int H  = quant_frame.rows, W  = quant_frame.cols;
    const int bH = H / BLOCK,        bW = W / BLOCK;

    float q[8][8];
    scaled_q(quality, q);
    recon = cv::Mat::zeros(H, W, CV_32F);

    for (int by = 0; by < bH; by++) {
        for (int bx = 0; bx < bW; bx++) {
            const int cy = by * BLOCK, cx = bx * BLOCK;

            BlockI q_block;
            cv::Mat qroi = quant_frame(cv::Rect(cx, cy, BLOCK, BLOCK));
            for (int y = 0; y < 8; y++)
                for (int x = 0; x < 8; x++)
                    q_block[y][x] = qroi.at<int16_t>(y, x);

            BlockF deq, rec;
            dequantize(q_block, q, deq);
            idct2d(deq, rec);

            cv::Mat rroi = recon(cv::Rect(cx, cy, BLOCK, BLOCK));
            for (int y = 0; y < 8; y++)
                for (int x = 0; x < 8; x++)
                    rroi.at<float>(y, x) = rec[y][x];
        }
    }
}

void reconstruct_frame(const cv::Mat& ref, const MvField& mv,
                       const cv::Mat& recon_residual, cv::Mat& output)
{
    const int H  = ref.rows, W  = ref.cols;
    const int bH = H / BLOCK, bW = W / BLOCK;

    cv::Mat ref_pad;
    cv::copyMakeBorder(ref, ref_pad, SEARCH, SEARCH, SEARCH, SEARCH,
                       cv::BORDER_REPLICATE);

    cv::Mat predicted = cv::Mat::zeros(H, W, CV_32F);

    for (int by = 0; by < bH; by++) {
        for (int bx = 0; bx < bW; bx++) {
            const int cy = by * BLOCK, cx = bx * BLOCK;
            const int dy = mv[by][bx].dy, dx = mv[by][bx].dx;
            const int ry = cy + SEARCH + dy, rx = cx + SEARCH + dx;

            cv::Mat ref_block;
            ref_pad(cv::Rect(rx, ry, BLOCK, BLOCK)).convertTo(ref_block, CV_32F);
            ref_block.copyTo(predicted(cv::Rect(cx, cy, BLOCK, BLOCK)));
        }
    }

    cv::Mat combined;
    cv::add(predicted, recon_residual, combined);
    combined = cv::max(combined, 0.0f);
    combined = cv::min(combined, 255.0f);
    combined.convertTo(output, CV_8U);
}

// ══════════════════════════════════════════════════════════════════════════════
// SECTION 8 — METRICS AND VISUALISATION
// ══════════════════════════════════════════════════════════════════════════════

float psnr(const cv::Mat& original, const cv::Mat& reconstructed)
{
    cv::Mat a, b, diff;
    original.convertTo(a, CV_32F);
    reconstructed.convertTo(b, CV_32F);
    cv::absdiff(a, b, diff);
    diff = diff.mul(diff);
    float mse = static_cast<float>(cv::mean(diff)[0]);
    if (mse < 1e-10f) return 99.0f;
    return 10.0f * std::log10(255.0f * 255.0f / mse);
}

cv::Mat visualize_mv(const cv::Mat& frame, const MvField& mv)
{
    cv::Mat vis;
    cv::cvtColor(frame, vis, cv::COLOR_GRAY2BGR);
    const int scale = 3;
    const int bH = static_cast<int>(mv.size());
    const int bW = bH ? static_cast<int>(mv[0].size()) : 0;

    for (int by = 0; by < bH; by++) {
        for (int bx = 0; bx < bW; bx++) {
            int dy = mv[by][bx].dy, dx = mv[by][bx].dx;
            if (dy == 0 && dx == 0) continue;
            cv::Point centre(bx * BLOCK + BLOCK / 2, by * BLOCK + BLOCK / 2);
            cv::Point tip(centre.x + dx * scale, centre.y + dy * scale);
            cv::arrowedLine(vis, centre, tip, cv::Scalar(0, 60, 220), 1,
                            cv::LINE_AA, 0, 0.3);
        }
    }
    return vis;
}

cv::Mat visualize_residual(const cv::Mat& residual)
{
    cv::Mat shifted, clipped;
    residual.convertTo(shifted, CV_32F);
    shifted += 128.0f;
    shifted = cv::max(shifted, 0.0f);
    shifted = cv::min(shifted, 255.0f);
    shifted.convertTo(clipped, CV_8U);
    return clipped;
}

// ══════════════════════════════════════════════════════════════════════════════
// SECTION 9 — SYNTHETIC FRAME GENERATION
// ══════════════════════════════════════════════════════════════════════════════

cv::Mat make_synthetic_frame(int W, int H,
                             int shift_x, int shift_y,
                             float noise_std, uint64_t seed)
{
    cv::Mat frame(H, W, CV_32F, cv::Scalar(80.0f));

    // bright rectangle
    frame(cv::Rect(15, 20, 30, 30)).setTo(200.0f);

    // dark circle
    cv::circle(frame,
               cv::Point(70 + shift_x, 55 + shift_y),
               14, cv::Scalar(40.0f), cv::FILLED);

    // gaussian noise
    cv::setRNGSeed(static_cast<uint64_t>(seed));
    cv::Mat noise(H, W, CV_32F);
    cv::randn(noise, 0.0f, noise_std);
    frame += noise;

    frame = cv::max(frame, 0.0f);
    frame = cv::min(frame, 255.0f);

    cv::Mat out;
    frame.convertTo(out, CV_8U);
    return out;
}

// ══════════════════════════════════════════════════════════════════════════════
// SECTION 10 — DEMO
// ══════════════════════════════════════════════════════════════════════════════

void run_demo(int quality)
{
    const int W = 128, H = 128;
    const int motion_x = 5, motion_y = 3;

    cv::Mat ref = make_synthetic_frame(W, H, 0,        0,        8.0f, 1);
    cv::Mat cur = make_synthetic_frame(W, H, motion_x, motion_y, 8.0f, 2);

    // encode
    MvField mv;
    cv::Mat residual;
    motion_estimate_frame(ref, cur, mv, residual);

    cv::Mat quant_frame;
    std::vector<std::vector<int16_t>> zz_blocks;
    Stats stats = encode_frame(residual, quality, quant_frame, zz_blocks);

    // decode
    cv::Mat recon_residual;
    decode_frame(quant_frame, quality, recon_residual);

    cv::Mat recon_frame;
    reconstruct_frame(ref, mv, recon_residual, recon_frame);

    // metrics
    cv::Mat ref_f, cur_f;
    ref.convertTo(ref_f, CV_32F);
    cur.convertTo(cur_f, CV_32F);
    cv::Mat naive_diff;
    cv::subtract(cur_f, ref_f, naive_diff);

    auto rms = [](const cv::Mat& m) {
        cv::Mat sq = m.mul(m);
        return std::sqrt(static_cast<float>(cv::mean(sq)[0]));
    };

    float rms_naive = rms(naive_diff);
    float rms_mc    = rms(residual);

    std::cout << "\n" << std::string(55, '=') << "\n";
    std::cout << "  Motion compensation + DCT pipeline  |  quality="
              << quality << "\n";
    std::cout << std::string(55, '=') << "\n\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Residual RMS  naive diff : " << rms_naive << "\n";
    std::cout << "  Residual RMS  MC         : " << rms_mc
              << "   (" << std::setprecision(0)
              << 100.0f * (1.0f - rms_mc / rms_naive) << "% reduction)\n";
    std::cout << std::setprecision(2);
    std::cout << "\n  DCT coefs total          : " << stats.total_coefs << "\n";
    std::cout << "  Non-zero after quant     : " << stats.nonzero << "\n";
    std::cout << "  Sparsity                 : "
              << stats.sparsity_pct << "%\n";
    std::cout << "\n  Reconstruction PSNR      : "
              << psnr(cur, recon_frame) << " dB\n";

    // quality sweep
    std::cout << "\n  Quality sweep:\n";
    std::cout << "  " << std::setw(8) << "Quality"
              << std::setw(12) << "Non-zero"
              << std::setw(11) << "Sparsity"
              << std::setw(10) << "PSNR" << "\n";
    std::cout << "  " << std::string(43, '-') << "\n";

    for (int q : {10, 25, 50, 75, 90, 95}) {
        cv::Mat qf, rd, rf;
        std::vector<std::vector<int16_t>> zz;
        Stats st = encode_frame(residual, q, qf, zz);
        decode_frame(qf, q, rd);
        reconstruct_frame(ref, mv, rd, rf);
        std::cout << "  " << std::setw(8) << q
                  << std::setw(12) << st.nonzero
                  << std::setw(10) << std::setprecision(1)
                  << st.sparsity_pct << "%"
                  << std::setw(9) << std::setprecision(2)
                  << psnr(cur, rf) << " dB\n";
    }

    // first block zigzag
    std::cout << "\n  First block zigzag (DC first):\n  ";
    for (int k = 0; k < 64; k++) {
        std::cout << std::setw(5) << zz_blocks[0][k];
        if ((k + 1) % 16 == 0) std::cout << "\n  ";
    }
    std::cout << "\n";

    // display
    auto to_bgr = [](const cv::Mat& m) {
        cv::Mat out;
        if (m.channels() == 1) cv::cvtColor(m, out, cv::COLOR_GRAY2BGR);
        else out = m.clone();
        return out;
    };

    cv::Mat mv_vis   = visualize_mv(cur, mv);
    cv::Mat res_vis  = visualize_residual(residual);
    cv::Mat nres_vis = visualize_residual(naive_diff);

    cv::Mat row1, row2, display;
    cv::hconcat(std::vector<cv::Mat>{to_bgr(ref), to_bgr(cur), mv_vis}, row1);
    cv::hconcat(std::vector<cv::Mat>{to_bgr(nres_vis),
                                     to_bgr(res_vis),
                                     to_bgr(recon_frame)}, row2);
    cv::vconcat(row1, row2, display);

    cv::resize(display, display,
               cv::Size(display.cols * 3, display.rows * 3),
               0, 0, cv::INTER_NEAREST);

    // labels
    std::vector<std::string> labels = {
        "reference", "current", "motion vectors",
        "naive residual", "MC residual", "reconstructed"
    };
    for (int i = 0; i < 6; i++) {
        int col_i = i % 3, row_i = i / 3;
        cv::putText(display, labels[i],
                    cv::Point(col_i * W * 3 + 4, row_i * H * 3 + 14),
                    cv::FONT_HERSHEY_SIMPLEX, 0.38,
                    cv::Scalar(200, 200, 200), 1);
    }

    cv::imshow("Pipeline stages", display);
    std::cout << "\n  Press any key in the image window to exit.\n";
    cv::waitKey(0);
    cv::destroyAllWindows();
}

// ══════════════════════════════════════════════════════════════════════════════
// MAIN
// ══════════════════════════════════════════════════════════════════════════════

int main(int argc, char** argv)
{
    int quality = 50;
    for (int i = 1; i < argc - 1; i++)
        if (std::string(argv[i]) == "--quality")
            quality = std::stoi(argv[i + 1]);

    run_demo(quality);
    return 0;
}
