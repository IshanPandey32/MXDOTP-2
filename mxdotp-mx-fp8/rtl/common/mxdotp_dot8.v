//==============================================================================
// mxdotp_dot8.v   (combinational)  -- STAGE 1 of MXDOTP
// 8 x (FP8 * FP8) products, each exactly expanded into a signed fixed-point
// number whose LSB has weight 2^-34, summed exactly:
//        T = sum_i (-1)^s_i * (Ma_i*Mb_i) << (eba_i + ebb_i - 2)
// plus the combined scale exponent S = Xa + Xb - 254 (E8M0 bias 127 each).
//   true dot product = T * 2^-34 * 2^S           (no rounding anywhere)
// TW = 72 bits is enough for up to 32 products (one whole MX block).
//==============================================================================
module mxdotp_dot8 #(
    parameter TW = 72
) (
    input  wire [63:0]          a,        // 8 x FP8
    input  wire [63:0]          b,        // 8 x FP8
    input  wire [7:0]           xa,       // E8M0 scale of A block
    input  wire [7:0]           xb,       // E8M0 scale of B block
    input  wire                 fmt,      // 0: E5M2, 1: E4M3
    output reg  [TW-1:0]        t,        // two's complement fixed-point sum
    output wire signed [9:0]    s,        // Xa + Xb - 254
    output wire                 nan_f,
    output wire                 pinf_f,
    output wire                 ninf_f,
    output wire                 allnegz   // every lane is a zero product with negative sign
);
    wire [8*TW-1:0] terms;
    wire [7:0] l_nan, l_inf, l_neg, l_zero;

    genvar i;
    generate
        for (i = 0; i < 8; i = i + 1) begin : lane
            wire       sa, sb, za, zb, na, nb, ia, ib;
            wire [3:0] ma, mb;
            wire [4:0] ea, eb;

            fp8_decode u_da (.x(a[8*i +: 8]), .fmt(fmt), .sign(sa), .mant(ma),
                             .ebias(ea), .is_zero(za), .is_nan(na), .is_inf(ia));
            fp8_decode u_db (.x(b[8*i +: 8]), .fmt(fmt), .sign(sb), .mant(mb),
                             .ebias(eb), .is_zero(zb), .is_nan(nb), .is_inf(ib));

            wire         fin = ~(na | nb | ia | ib);
            wire [7:0]   p   = ma * mb;                              // 4x4 -> 8 bit
            wire [5:0]   pos = {1'b0, ea} + {1'b0, eb} - 6'd2;       // 0..58
            wire [TW-1:0] pm = fin ? ({{(TW-8){1'b0}}, p} << pos) : {TW{1'b0}};

            assign terms[i*TW +: TW] = (sa ^ sb) ? (~pm + 1'b1) : pm;

            assign l_nan[i]  = na | nb | (ia & zb) | (ib & za);
            assign l_inf[i]  = (ia | ib) & ~l_nan[i];
            assign l_neg[i]  = sa ^ sb;
            assign l_zero[i] = fin & (za | zb);
        end
    endgenerate

    integer k;
    always @* begin
        t = {TW{1'b0}};
        for (k = 0; k < 8; k = k + 1)
            t = t + terms[k*TW +: TW];
    end

    assign s       = $signed({2'b00, xa}) + $signed({2'b00, xb}) - 10'sd254;
    assign nan_f   = (|l_nan) | (xa == 8'hFF) | (xb == 8'hFF);
    assign pinf_f  = |(l_inf & ~l_neg);
    assign ninf_f  = |(l_inf &  l_neg);
    assign allnegz = &(l_zero & l_neg);
endmodule
