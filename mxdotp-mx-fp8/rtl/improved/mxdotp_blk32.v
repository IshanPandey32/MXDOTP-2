//==============================================================================
// mxdotp_blk32.v  --  IMPROVED: block-level fused MXDOTP ("MXDOTP-B32")
//
// Improvements over the paper-style baseline (mxdotp_unit.v):
//
//  (1) DEFERRED NORMALISATION / SINGLE ROUNDING PER MX BLOCK
//      The baseline rounds to FP32 after every 8-element beat (4x per 32-element
//      block).  Here the exact fixed-point sums T of up to 4 beats are accumulated
//      in a 72-bit register (an exact "mini Kulisch accumulator", 1 adder delay),
//      and the align / normalise / round hardware is used ONCE per block.
//        -> 1 rounding instead of 4 (lower numerical error, no double rounding)
//        -> alignment shifter + LZD + rounder activity cut by 4x (energy)
//
//  (2) NO LOOP-CARRIED FP32 DEPENDENCY
//      Beats of a block are chained through the narrow T register, not through the
//      3-stage FP32 add/round pipeline.  One block-beat can issue every cycle with
//      a single accumulator stream, no software unrolling by 8 outputs is needed
//      to hide the pipeline latency.
//
//  (3) HARDWARE BLOCK CONTROL (first/last) + ERROR FLAG
//      first/last tags replace the per-beat software loop and scale re-issue.
//      `blk_err` flags a protocol violation (scale changed inside a block, or more
//      than 4 beats = 32 elements, which would exceed the 72-bit T range).
//
//  Usage per MX block (same Xa/Xb for all beats):
//      beat0: first=1 ; beat1,2: first=0,last=0 ; beat3: last=1, acc = running FP32 sum
//      (acc_in is sampled on the `last` beat; shorter blocks (1..4 beats) are fine)
//  Latency: 3 cycles from the last beat to the result, throughput 1 beat/cycle.
//==============================================================================
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

    wire [TW-1:0]     t8;
    wire signed [9:0] s8;
    wire              nan8, pinf8, ninf8, nz8;

    mxdotp_dot8 #(.TW(TW)) u_dot8 (
        .a(a), .b(b), .xa(xa), .xb(xb), .fmt(fmt),
        .t(t8), .s(s8), .nan_f(nan8), .pinf_f(pinf8), .ninf_f(ninf8), .allnegz(nz8));

    // ---------------- exact block accumulator ----------------
    reg [TW-1:0]      t_run;
    reg signed [9:0]  s_run;
    reg               nan_r, pinf_r, ninf_r, nz_r, err_r;
    reg [2:0]         cnt_r;

    wire [TW-1:0] t_nxt    = first ? t8 : (t_run + t8);
    wire [2:0]    cnt_nxt  = first ? 3'd1 : (cnt_r + 3'd1);
    wire          err_nxt  = (first ? 1'b0 : err_r)
                           | (cnt_nxt > 3'd4)
                           | (~first & (s8 != s_run));
    wire          nan_nxt  = (first ? 1'b0 : nan_r)  | nan8;
    wire          pinf_nxt = (first ? 1'b0 : pinf_r) | pinf8;
    wire          ninf_nxt = (first ? 1'b0 : ninf_r) | ninf8;
    wire          nz_nxt   = (first ? 1'b1 : nz_r)   & nz8;

    always @(posedge clk) begin
        if (in_valid) begin
            t_run  <= t_nxt;
            s_run  <= first ? s8 : s_run;
            cnt_r  <= cnt_nxt;
            err_r  <= err_nxt;
            nan_r  <= nan_nxt;  pinf_r <= pinf_nxt;  ninf_r <= ninf_nxt;  nz_r <= nz_nxt;
        end
    end

    // ---------------- launch one add/round per block ----------------
    reg                l_valid, l_nan, l_pinf, l_ninf, l_nz, l_err;
    reg [TW-1:0]       l_t;
    reg signed [9:0]   l_s;
    reg [31:0]         l_acc;

    always @(posedge clk) begin
        if (!rst_n) l_valid <= 1'b0;
        else        l_valid <= in_valid & last;
        l_t   <= t_nxt;      l_s    <= first ? s8 : s_run;   l_acc  <= acc_in;
        l_nan <= nan_nxt;    l_pinf <= pinf_nxt;  l_ninf <= ninf_nxt;  l_nz <= nz_nxt;
        l_err <= err_nxt;
    end

    mxdotp_addround #(.TW(TW)) u_addround (
        .clk(clk), .rst_n(rst_n), .in_valid(l_valid),
        .t(l_t), .s(l_s), .acc(l_acc),
        .nan_f(l_nan), .pinf_f(l_pinf), .ninf_f(l_ninf), .allnegz(l_nz),
        .out_valid(out_valid), .result(result));

    // error flag delayed to match the 2-stage addround latency
    reg err_d1, err_d2;
    always @(posedge clk) begin
        err_d1 <= l_err;
        err_d2 <= err_d1;
    end
    assign blk_err = err_d2;
endmodule
