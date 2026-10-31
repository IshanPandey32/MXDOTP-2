//==============================================================================
// mxdotp_unit.v  --  BASELINE: re-implementation of the paper's MXDOTP datapath
//
//   result = acc + (Xa * Xb) * sum_{i=0..7} A_i * B_i       (MXFP8, FP32 accumulate)
//
//   - 8 MXFP8 elements per operand packed in 64 bits (E5M2 or E4M3, `fmt`)
//   - two independent E8M0 block scales (xa, xb)
//   - FP32 accumulator, exact early accumulation, single RNE rounding
//   - 3 pipeline stages, throughput 1 result / cycle, latency 3 cycles
//
//   One MXDOTP instruction == one beat here.  A 32-element MX block needs four
//   dependent beats (acc of beat n = result of beat n-1), i.e. the result is
//   rounded to FP32 four times per block (exactly like the paper's software loop).
//==============================================================================
module mxdotp_unit (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        in_valid,
    input  wire        fmt,          // 0: E5M2  1: E4M3
    input  wire [63:0] a,
    input  wire [63:0] b,
    input  wire [7:0]  xa,
    input  wire [7:0]  xb,
    input  wire [31:0] acc,
    output wire        out_valid,
    output wire [31:0] result
);
    localparam TW = 72;

    wire [TW-1:0]     t_c;
    wire signed [9:0] s_c;
    wire              nan_c, pinf_c, ninf_c, nz_c;

    mxdotp_dot8 #(.TW(TW)) u_dot8 (
        .a(a), .b(b), .xa(xa), .xb(xb), .fmt(fmt),
        .t(t_c), .s(s_c), .nan_f(nan_c), .pinf_f(pinf_c), .ninf_f(ninf_c), .allnegz(nz_c));

    // ---- pipeline register 1 ----
    reg                s1_valid, s1_nan, s1_pinf, s1_ninf, s1_nz;
    reg [TW-1:0]       s1_t;
    reg signed [9:0]   s1_s;
    reg [31:0]         s1_acc;

    always @(posedge clk) begin
        if (!rst_n) s1_valid <= 1'b0;
        else        s1_valid <= in_valid;
        s1_t    <= t_c;   s1_s   <= s_c;   s1_acc <= acc;
        s1_nan  <= nan_c; s1_pinf <= pinf_c; s1_ninf <= ninf_c; s1_nz <= nz_c;
    end

    mxdotp_addround #(.TW(TW)) u_addround (
        .clk(clk), .rst_n(rst_n), .in_valid(s1_valid),
        .t(s1_t), .s(s1_s), .acc(s1_acc),
        .nan_f(s1_nan), .pinf_f(s1_pinf), .ninf_f(s1_ninf), .allnegz(s1_nz),
        .out_valid(out_valid), .result(result));
endmodule
