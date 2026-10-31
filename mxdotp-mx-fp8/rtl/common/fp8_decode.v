//==============================================================================
// fp8_decode.v
// Decodes one FP8 element (E5M2 or E4M3) into a common integer form:
//      value = (-1)^sign * mant * 2^(ebias - 18)
//   mant  : 4-bit significand incl. hidden bit (0 for zero/subnormal-hidden)
//   ebias : 5-bit "effective biased exponent" shared by both formats
//             E5M2 : e_eff            (e_eff = (e==0) ? 1 : e)
//             E4M3 : e_eff + 8        (re-biased so both formats share 2^(eb-18))
// fmt = 0 -> E5M2 (Inf/NaN as IEEE)       fmt = 1 -> E4M3 (OCP: no Inf, NaN=S.1111.111)
//==============================================================================
module fp8_decode (
    input  wire [7:0] x,
    input  wire       fmt,
    output wire       sign,
    output wire [3:0] mant,
    output wire [4:0] ebias,
    output wire       is_zero,   // finite zero (either sign)
    output wire       is_nan,
    output wire       is_inf
);
    wire [4:0] e5 = x[6:2];
    wire [1:0] m5 = x[1:0];
    wire [3:0] e4 = x[6:3];
    wire [2:0] m4 = x[2:0];

    wire nan5 = (e5 == 5'd31) & (m5 != 2'd0);
    wire inf5 = (e5 == 5'd31) & (m5 == 2'd0);
    wire nan4 = (e4 == 4'd15) & (m4 == 3'd7);

    wire [3:0] mant5 = {(e5 != 5'd0), m5, 1'b0};
    wire [3:0] mant4 = {(e4 != 4'd0), m4};
    wire [4:0] eb5   = (e5 == 5'd0) ? 5'd1 : e5;
    wire [4:0] eb4   = ((e4 == 4'd0) ? 5'd1 : {1'b0, e4}) + 5'd8;

    assign sign    = x[7];
    assign mant    = fmt ? mant4 : mant5;
    assign ebias   = fmt ? eb4   : eb5;
    assign is_nan  = fmt ? nan4  : nan5;
    assign is_inf  = fmt ? 1'b0  : inf5;
    assign is_zero = (mant == 4'd0) & ~is_nan & ~is_inf;
endmodule
