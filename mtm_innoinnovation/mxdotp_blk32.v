//==============================================================================
// mxdotp_blk32.v -- Self-contained block-level fused MXDOTP ("MXDOTP-B32")
// Includes:
//   1. mxdotp_dot8     (8-element FP8 E5M2/E4M3 exact fixed-point dot product)
//   2. mxdotp_addround (2-stage FP32 align, normalize, and round-to-nearest-even)
//   3. mxdotp_blk32    (Top-level 32-element block accumulator)
//==============================================================================
`timescale 1ns/1ps

//------------------------------------------------------------------------------
// Sub-module 1: 8-Element FP8 Dot Product to Exact Fixed-Point Accumulator
//------------------------------------------------------------------------------
module mxdotp_dot8 #(
    parameter TW = 72
)(
    input  wire [63:0]          a,
    input  wire [63:0]          b,
    input  wire [7:0]           xa,
    input  wire [7:0]           xb,
    input  wire                 fmt,      // 0: E5M2, 1: E4M3
    output wire signed [TW-1:0] t,
    output wire signed [9:0]    s,
    output wire                 nan_f,
    output wire                 pinf_f,
    output wire                 ninf_f,
    output wire                 allnegz
);
    // Shared scale exponent: Xa + Xb - 254 (two E8M0 biases of 127)
    assign s = $signed({2'b00, xa}) + $signed({2'b00, xb}) - 10'sd254;

    wire [7:0] prod_nan, prod_pinf, prod_ninf, prod_negz;
    wire signed [TW-1:0] term [0:7];

    genvar i;
    generate
        for (i = 0; i < 8; i = i + 1) begin : gen_elem
            wire [7:0] ai = a[8*i +: 8];
            wire [7:0] bi = b[8*i +: 8];

            wire sa = ai[7];
            wire sb = bi[7];
            wire sp = sa ^ sb;

            // Unpack E5M2 (fmt=0) vs E4M3 (fmt=1)
            wire [4:0] ea = fmt ? {1'b0, ai[6:3]} : ai[6:2];
            wire [4:0] eb = fmt ? {1'b0, bi[6:3]} : bi[6:2];

            wire [3:0] ma_raw = fmt ? {1'b0, ai[2:0]} : {1'b0, ai[1:0], 1'b0};
            wire [3:0] mb_raw = fmt ? {1'b0, bi[2:0]} : {1'b0, bi[1:0], 1'b0};

            wire a_zero = (ai[6:0] == 7'd0);
            wire b_zero = (bi[6:0] == 7'd0);

            // Special value detection
            wire a_inf = (~fmt) & (ai[6:2] == 5'h1F) & (ai[1:0] == 2'b00);
            wire b_inf = (~fmt) & (bi[6:2] == 5'h1F) & (bi[1:0] == 2'b00);

            wire a_nan = fmt ? (ai[6:0] == 7'h7F) : ((ai[6:2] == 5'h1F) & (ai[1:0] != 2'b00));
            wire b_nan = fmt ? (bi[6:0] == 7'h7F) : ((bi[6:2] == 5'h1F) & (bi[1:0] != 2'b00));

            assign prod_nan[i]  = a_nan | b_nan | (a_inf & b_zero) | (b_inf & a_zero);
            wire   p_inf        = (a_inf | b_inf) & ~prod_nan[i];
            assign prod_pinf[i] = p_inf & ~sp;
            assign prod_ninf[i] = p_inf &  sp;
            assign prod_negz[i] = (a_zero | b_zero) & sp;

            // Normalized mantissas (with hidden bit, aligned to 3 fractional bits)
            wire [3:0] ma = a_zero ? 4'd0 :
                            (ea == 5'd0) ? ma_raw : {1'b1, ma_raw[2:0]};
            wire [3:0] mb = b_zero ? 4'd0 :
                            (eb == 5'd0) ? mb_raw : {1'b1, mb_raw[2:0]};

            // Effective biased exponents (subnormals have eff exp = 1)
            wire [5:0] ea_eff = (ea == 5'd0) ? 6'd1 : {1'b0, ea};
            wire [5:0] eb_eff = (eb == 5'd0) ? 6'd1 : {1'b0, eb};

            // Relative shift into the 72-bit Kulisch-like window
            // E5M2 min exp_sum = 2 (bias 30), E4M3 min exp_sum = 2 (bias 14 -> offset +16)
            wire [5:0] exp_shift = (ea_eff + eb_eff - 6'd2) + (fmt ? 6'd16 : 6'd0);

            wire [7:0] m_prod = (a_zero | b_zero | p_inf | prod_nan[i]) ? 8'd0 : (ma * mb);
            wire [TW-1:0] shifted_mag = {{(TW-8){1'b0}}, m_prod} << exp_shift;

            assign term[i] = sp ? -$signed(shifted_mag) : $signed(shifted_mag);
        end
    endgenerate

    assign nan_f   = (|prod_nan) | ((|prod_pinf) & (|prod_ninf));
    assign pinf_f  = (~nan_f) & (|prod_pinf);
    assign ninf_f  = (~nan_f) & (|prod_ninf);
    assign allnegz = &prod_negz;

    assign t = term[0] + term[1] + term[2] + term[3] +
               term[4] + term[5] + term[6] + term[7];
endmodule

//------------------------------------------------------------------------------
// Sub-module 2: 2-Stage FP32 Add, Normalize, and Round Unit
//------------------------------------------------------------------------------
module mxdotp_addround #(
    parameter TW = 72
)(
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 in_valid,
    input  wire signed [TW-1:0] t,
    input  wire signed [9:0]    s,
    input  wire [31:0]          acc,
    input  wire                 nan_f,
    input  wire                 pinf_f,
    input  wire                 ninf_f,
    input  wire                 allnegz,
    output reg                  out_valid = 1'b0,
    output reg  [31:0]          result    = 32'd0
);
    // ---------------- Stage 1: Unpack Acc & Convert T to Sign-Magnitude ------
    reg               s1_valid   = 1'b0;
    reg               s1_nan     = 1'b0;
    reg               s1_pinf    = 1'b0;
    reg               s1_ninf    = 1'b0;
    reg               s1_allnegz = 1'b0;
    reg [31:0]        s1_acc     = 32'd0;
    reg               s1_t_sign  = 1'b0;
    reg [TW-1:0]      s1_t_mag   = {TW{1'b0}};
    reg signed [9:0]  s1_s       = 10'sd0;

    wire t_neg = t[TW-1];
    wire [TW-1:0] t_abs = t_neg ? (~t + 1'b1) : t;

    always @(posedge clk) begin
        if (!rst_n) begin
            s1_valid   <= 1'b0;
            s1_nan     <= 1'b0;
            s1_pinf    <= 1'b0;
            s1_ninf    <= 1'b0;
            s1_allnegz <= 1'b0;
            s1_acc     <= 32'd0;
            s1_t_sign  <= 1'b0;
            s1_t_mag   <= {TW{1'b0}};
            s1_s       <= 10'sd0;
        end else begin
            s1_valid   <= in_valid;
            s1_nan     <= nan_f;
            s1_pinf    <= pinf_f;
            s1_ninf    <= ninf_f;
            s1_allnegz <= allnegz;
            s1_acc     <= acc;
            s1_t_sign  <= t_neg;
            s1_t_mag   <= t_abs;
            s1_s       <= s;
        end
    end

    // ---------------- Stage 2: Align, Add with FP32 Acc, Normalize & Round ---
    wire        acc_s   = s1_acc[31];
    wire [7:0]  acc_e   = s1_acc[30:23];
    wire [22:0] acc_m   = s1_acc[22:0];
    wire        acc_z   = (s1_acc[30:0] == 31'd0);
    wire        acc_inf = (acc_e == 8'hFF) & (acc_m == 23'd0);
    wire        acc_nan = (acc_e == 8'hFF) & (acc_m != 23'd0);

    // Find MSB of s1_t_mag (Leading Zero Detector)
    integer k;
    reg [6:0] msb_idx;
    reg       t_is_zero;
    always @(*) begin
        msb_idx   = 7'd0;
        t_is_zero = (s1_t_mag == {TW{1'b0}});
        for (k = 0; k < TW; k = k + 1) begin
            if (s1_t_mag[k]) msb_idx = k[6:0];
        end
    end

    // T represents: s1_t_mag * 2^(s1_s - 34)
    // Unbiased exponent of T = s1_s - 34 + msb_idx; FP32 biased exp = unbiased + 127 = s1_s + msb_idx + 93
    wire signed [10:0] t_exp_biased = $signed({1'b0, s1_s}) + $signed({4'b0, msb_idx}) + 11'sd93;

    // Shift s1_t_mag so MSB sits at bit 71
    wire [6:0]    lshift = 7'd71 - msb_idx;
    wire [TW-1:0] t_norm = s1_t_mag << lshift;

    // Combine with acc in extended precision
    wire signed [10:0] acc_exp_eff = acc_z ? 11'sd0 :
                                     (acc_e == 8'd0) ? 11'sd1 : $signed({3'b0, acc_e});
    wire [23:0] acc_mant = acc_z ? 24'd0 :
                           (acc_e == 8'd0) ? {1'b0, acc_m} : {1'b1, acc_m};

    // Convert T alone to FP32 with Round-to-Nearest-Even (when acc == 0)
    wire [23:0] t_mant24 = t_norm[71:48];
    wire        t_guard  = t_norm[47];
    wire        t_round  = t_norm[46];
    wire        t_sticky = |t_norm[45:0];
    wire        t_rup    = t_guard & (t_round | t_sticky | t_mant24[0]);
    wire [24:0] t_m_rnd  = {1'b0, t_mant24} + t_rup;
    wire signed [10:0] t_e_rnd = t_exp_biased + (t_m_rnd[24] ? 11'sd1 : 11'sd0);
    wire [22:0] t_f_rnd  = t_m_rnd[24] ? t_m_rnd[23:1] : t_m_rnd[22:0];

    // Dual-operand alignment (when both T and acc are non-zero)
    wire signed [10:0] exp_diff = t_exp_biased - acc_exp_eff;
    wire t_larger = (exp_diff >= 0);
    wire signed [10:0] exp_max = t_larger ? t_exp_biased : acc_exp_eff;
    wire [10:0] sh_amt = t_larger ? exp_diff[10:0] : (-exp_diff);
    wire [6:0]  sh_sat = (sh_amt > 11'd74) ? 7'd74 : sh_amt[6:0];

    wire [74:0] op_t_raw   = {1'b0, t_norm, 2'b00};
    wire [74:0] op_acc_raw = {1'b0, acc_mant, 50'd0};

    wire [74:0] op_big     = t_larger ? op_t_raw   : op_acc_raw;
    wire [74:0] op_small   = t_larger ? op_acc_raw : op_t_raw;
    wire        sign_big   = t_larger ? s1_t_sign  : acc_s;
    wire        sign_small = t_larger ? acc_s      : s1_t_sign;

    wire [149:0] small_shifted_full = {op_small, 75'd0} >> sh_sat;
    wire [74:0]  op_small_aligned   = {small_shifted_full[149:76], |small_shifted_full[75:0]};

    wire eff_sub = sign_big ^ sign_small;
    wire [75:0] sum_mag = eff_sub ?
                          ((op_big >= op_small_aligned) ? (op_big - op_small_aligned) : (op_small_aligned - op_big)) :
                          (op_big + op_small_aligned);
    wire sum_sign = (eff_sub && (op_small_aligned > op_big)) ? sign_small : sign_big;

    integer m;
    reg [6:0] sum_msb;
    reg       sum_zero;
    always @(*) begin
        sum_msb  = 7'd0;
        sum_zero = (sum_mag == 76'd0);
        for (m = 0; m < 76; m = m + 1) begin
            if (sum_mag[m]) sum_msb = m[6:0];
        end
    end

    wire [75:0] sum_norm = (sum_msb >= 7'd73) ? (sum_mag >> (sum_msb - 7'd73)) :
                                                (sum_mag << (7'd73 - sum_msb));
    wire signed [10:0] sum_exp = exp_max + $signed({4'b0, sum_msb}) - 11'sd73;

    wire [23:0] s_mant24 = sum_norm[73:50];
    wire        s_guard  = sum_norm[49];
    wire        s_round  = sum_norm[48];
    wire        s_sticky = |sum_norm[47:0];
    wire        s_rup    = s_guard & (s_round | s_sticky | s_mant24[0]);
    wire [24:0] s_m_rnd  = {1'b0, s_mant24} + s_rup;
    wire signed [10:0] s_e_rnd = sum_exp + (s_m_rnd[24] ? 11'sd1 : 11'sd0);
    wire [22:0] s_f_rnd  = s_m_rnd[24] ? s_m_rnd[23:1] : s_m_rnd[22:0];

    reg [31:0] comb_res;
    always @(*) begin
        // 1. NaN cases
        if (s1_nan || acc_nan || (s1_pinf && acc_inf && acc_s) || (s1_ninf && acc_inf && !acc_s)) begin
            comb_res = 32'h7FC00000; // Canonical QNaN
        end
        // 2. Infinity cases
        else if (s1_pinf || (acc_inf && !acc_s)) begin
            comb_res = 32'h7F800000;
        end
        else if (s1_ninf || (acc_inf && acc_s)) begin
            comb_res = 32'hFF800000;
        end
        // 3. Both T and Acc are zero
        else if (t_is_zero && acc_z) begin
            comb_res = {(s1_allnegz & acc_s), 31'd0};
        end
        // 4. T is zero -> return Acc
        else if (t_is_zero) begin
            comb_res = s1_acc;
        end
        // 5. Acc is zero -> return rounded T
        else if (acc_z) begin
            if (t_e_rnd >= 11'sd255)      comb_res = {s1_t_sign, 8'hFF, 23'd0};
            else if (t_e_rnd <= 11'sd0)   comb_res = {s1_t_sign, 31'd0};
            else                          comb_res = {s1_t_sign, t_e_rnd[7:0], t_f_rnd};
        end
        // 6. General sum of T + Acc
        else if (sum_zero) begin
            comb_res = 32'h00000000;
        end
        else begin
            if (s_e_rnd >= 11'sd255)      comb_res = {sum_sign, 8'hFF, 23'd0};
            else if (s_e_rnd <= 11'sd0)   comb_res = {sum_sign, 31'd0};
            else                          comb_res = {sum_sign, s_e_rnd[7:0], s_f_rnd};
        end
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
            result    <= 32'd0;
        end else begin
            out_valid <= s1_valid;
            result    <= comb_res;
        end
    end
endmodule

//------------------------------------------------------------------------------
// Top-Level Module: mxdotp_blk32
//------------------------------------------------------------------------------
module mxdotp_blk32 (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        in_valid,
    input  wire        first,        // first beat of a block
    input  wire        last,         // last beat of a block (acc_in sampled here)
    input  wire        fmt,          // 0: E5M2  1: E4M3
    input  wire [63:0] a,
    input  wire [63:0] b,
    input  wire [7:0]  xa,
    input  wire [7:0]  xb,
    input  wire [31:0] acc_in,
    output wire        out_valid,
    output wire [31:0] result,
    output wire        blk_err       // aligned with out_valid
);
    localparam TW = 72;

    wire signed [TW-1:0] t8;
    wire signed [9:0]    s8;
    wire                 nan8, pinf8, ninf8, nz8;

    mxdotp_dot8 #(.TW(TW)) u_dot8 (
        .a(a), .b(b), .xa(xa), .xb(xb), .fmt(fmt),
        .t(t8), .s(s8), .nan_f(nan8), .pinf_f(pinf8), .ninf_f(ninf8), .allnegz(nz8)
    );

    // ---------------- exact block accumulator ----------------
    reg signed [TW-1:0] t_run  = {TW{1'b0}};
    reg signed [9:0]    s_run  = 10'sd0;
    reg                 nan_r  = 1'b0;
    reg                 pinf_r = 1'b0;
    reg                 ninf_r = 1'b0;
    reg                 nz_r   = 1'b0;
    reg                 err_r  = 1'b0;
    reg [2:0]           cnt_r  = 3'd0;

    wire signed [TW-1:0] t_nxt    = first ? t8 : (t_run + t8);
    wire [2:0]           cnt_nxt  = first ? 3'd1 : (cnt_r + 3'd1);
    wire                 err_nxt  = (first ? 1'b0 : err_r)
                                  | (cnt_nxt > 3'd4)
                                  | (~first & (s8 != s_run));
    wire                 nan_nxt  = (first ? 1'b0 : nan_r)  | nan8;
    wire                 pinf_nxt = (first ? 1'b0 : pinf_r) | pinf8;
    wire                 ninf_nxt = (first ? 1'b0 : ninf_r) | ninf8;
    wire                 nz_nxt   = (first ? 1'b1 : nz_r)   & nz8;

    always @(posedge clk) begin
        if (!rst_n) begin
            t_run  <= {TW{1'b0}};
            s_run  <= 10'sd0;
            cnt_r  <= 3'd0;
            err_r  <= 1'b0;
            nan_r  <= 1'b0;
            pinf_r <= 1'b0;
            ninf_r <= 1'b0;
            nz_r   <= 1'b0;
        end else if (in_valid) begin
            t_run  <= t_nxt;
            s_run  <= first ? s8 : s_run;
            cnt_r  <= cnt_nxt;
            err_r  <= err_nxt;
            nan_r  <= nan_nxt;
            pinf_r <= pinf_nxt;
            ninf_r <= ninf_nxt;
            nz_r   <= nz_nxt;
        end
    end

    // ---------------- launch one add/round per block ----------------
    reg                 l_valid = 1'b0;
    reg                 l_nan   = 1'b0;
    reg                 l_pinf  = 1'b0;
    reg                 l_ninf  = 1'b0;
    reg                 l_nz    = 1'b0;
    reg                 l_err   = 1'b0;
    reg signed [TW-1:0] l_t     = {TW{1'b0}};
    reg signed [9:0]    l_s     = 10'sd0;
    reg [31:0]          l_acc   = 32'd0;

    always @(posedge clk) begin
        if (!rst_n) begin
            l_valid <= 1'b0;
            l_t     <= {TW{1'b0}};
            l_s     <= 10'sd0;
            l_acc   <= 32'd0;
            l_nan   <= 1'b0;
            l_pinf  <= 1'b0;
            l_ninf  <= 1'b0;
            l_nz    <= 1'b0;
            l_err   <= 1'b0;
        end else begin
            l_valid <= in_valid & last;
            l_t     <= t_nxt;
            l_s     <= first ? s8 : s_run;
            l_acc   <= acc_in;
            l_nan   <= nan_nxt;
            l_pinf  <= pinf_nxt;
            l_ninf  <= ninf_nxt;
            l_nz    <= nz_nxt;
            l_err   <= err_nxt;
        end
    end

    mxdotp_addround #(.TW(TW)) u_addround (
        .clk(clk), .rst_n(rst_n), .in_valid(l_valid),
        .t(l_t), .s(l_s), .acc(l_acc),
        .nan_f(l_nan), .pinf_f(l_pinf), .ninf_f(l_ninf), .allnegz(l_nz),
        .out_valid(out_valid), .result(result)
    );

    // error flag delayed to match the 2-stage addround latency
    reg err_d1 = 1'b0, err_d2 = 1'b0;
    always @(posedge clk) begin
        if (!rst_n) begin
            err_d1 <= 1'b0;
            err_d2 <= 1'b0;
        end else begin
            err_d1 <= l_err;
            err_d2 <= err_d1;
        end
    end
    assign blk_err = err_d2;
endmodule
