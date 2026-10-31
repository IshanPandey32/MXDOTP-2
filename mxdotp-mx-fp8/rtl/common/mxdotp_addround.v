//==============================================================================
// mxdotp_addround.v  -- STAGES 2 and 3 of MXDOTP (2 pipeline registers)
//
//  result = RNE_fp32( acc + 2^S * T * 2^-34 )     -- one single rounding
//
//  Stage 2 (align + add): the FP32 accumulator is shifted into a wide fixed-point
//          window ("early accumulation", as in the paper) and added to T.
//          - window LSB weight = 2^(S-34-OT), OT = 32 guard/extension bits
//          - if the accumulator is far larger than the product sum (d > DMAX) the
//            product sum collapses into a +/-1 "sticky" (it can only affect rounding)
//          - if the accumulator is far smaller it is shifted right with sticky
//  Stage 3 (normalise + round): leading-one detect, gradual-underflow aware
//          round-to-nearest-even, overflow -> Inf, special values.
//
//  Bit-exact mirror of scripts/rtl_model.py (validated against exact arithmetic).
//==============================================================================
module mxdotp_addround #(
    parameter TW = 72
) (
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 in_valid,
    input  wire [TW-1:0]        t,
    input  wire signed [9:0]    s,
    input  wire [31:0]          acc,
    input  wire                 nan_f,
    input  wire                 pinf_f,
    input  wire                 ninf_f,
    input  wire                 allnegz,
    output reg                  out_valid,
    output reg  [31:0]          result
);
    localparam OT   = 32;      // extension bits below the product LSB
    localparam DMAX = 73;      // above this, product sum is only a sticky bit
    localparam SW   = 132;     // window width

    //----------------------------------------------------------------------
    // STAGE 2 : classify, align, add
    //----------------------------------------------------------------------
    wire        acc_s    = acc[31];
    wire [7:0]  acc_e    = acc[30:23];
    wire [22:0] acc_m    = acc[22:0];
    wire        acc_nan  = (acc_e == 8'hFF) & (acc_m != 23'd0);
    wire        acc_inf  = (acc_e == 8'hFF) & (acc_m == 23'd0);
    wire        inf_pos  = pinf_f | (acc_inf & ~acc_s);
    wire        inf_neg  = ninf_f | (acc_inf &  acc_s);
    wire        nan_all  = nan_f | acc_nan | (inf_pos & inf_neg);
    wire        special  = nan_all | inf_pos | inf_neg;
    wire [31:0] spec_res = nan_all ? 32'h7FC00000 : (inf_pos ? 32'h7F800000 : 32'hFF800000);
    wire [23:0] A        = {(acc_e != 8'd0), acc_m};
    wire        t_zero   = (t == {TW{1'b0}});

    reg  [SW-1:0]       Au, Tw, Wsum, mag_c;
    reg  signed [11:0]  wexp_c;
    reg                 sign_c;
    integer             ea_eff, s_i, d, pa, kk;

    always @* begin
        ea_eff = (acc_e == 8'd0) ? 1 : acc_e;
        s_i    = s;                                   // sign-extended
        d      = ea_eff - s_i - 116;                  // acc exponent - product exponent (LSB units)
        pa     = OT + d;                              // bit position of acc LSB in the window
        Tw     = {{(SW-TW){t[TW-1]}}, t};
        Tw     = Tw << OT;
        Au     = {{(SW-24){1'b0}}, A};
        wexp_c = s_i - 34 - OT;
        if (A == 24'd0) begin
            Au = {SW{1'b0}};
        end else if (d > DMAX) begin                  // accumulator dominates
            Au     = Au << (OT + DMAX);
            Tw     = t[TW-1] ? {SW{1'b1}} : {{(SW-1){1'b0}}, 1'b1};
            wexp_c = ea_eff - 150 - OT - DMAX;
        end else if (pa >= 0) begin                   // exact placement
            Au = Au << pa;
        end else begin                                // accumulator far below: shift right + sticky
            kk = -pa;
            if (kk >= 24) begin
                Au = {{(SW-1){1'b0}}, 1'b1};
            end else begin
                Au = Au >> kk;
                if ((A & ((24'd1 << kk) - 24'd1)) != 24'd0) Au[0] = 1'b1;
            end
        end
        Wsum   = (acc_s ? (~Au + 1'b1) : Au) + Tw;
        sign_c = Wsum[SW-1];
        mag_c  = sign_c ? (~Wsum + 1'b1) : Wsum;
    end

    reg                s2_valid, s2_sign, s2_bypass, s2_special, s2_allnegz;
    reg  [SW-1:0]      s2_mag;
    reg  signed [11:0] s2_wexp;
    reg  [31:0]        s2_acc, s2_spec;

    always @(posedge clk) begin
        if (!rst_n) s2_valid <= 1'b0;
        else        s2_valid <= in_valid;
        s2_mag     <= mag_c;
        s2_sign    <= sign_c;
        s2_wexp    <= wexp_c;
        s2_bypass  <= t_zero;                         // acc + 0 == acc
        s2_special <= special;
        s2_spec    <= spec_res;
        s2_acc     <= acc;
        s2_allnegz <= allnegz;
    end

    //----------------------------------------------------------------------
    // STAGE 3 : normalise, round (RNE), pack
    //----------------------------------------------------------------------
    reg  [31:0]        res_c;
    integer            p, e_b, rA, rB, r, sh, jj;
    reg  [SW+27:0]     ext, kq, lostv;
    reg  [23:0]        keep;
    reg                guard, sticky, inc;
    reg  [24:0]        rounded;
    reg  [8:0]         base;
    reg  [31:0]        packed;

    always @* begin
        p = 0;
        for (jj = 0; jj < SW; jj = jj + 1)
            if (s2_mag[jj]) p = jj;                   // leading-one position
        e_b = p + s2_wexp + 127;                      // biased exponent of the result
        rA  = p - 23;                                 // LSB position, normal result
        rB  = -149 - s2_wexp;                         // LSB position, subnormal result
        r   = (rA > rB) ? rA : rB;
        sh  = r + 26;
        if (sh > SW + 28) sh = SW + 28;
        ext    = {s2_mag, 28'd0};
        kq     = ext >> sh;                           // {keep[23:0], guard, sticky_lsb}
        lostv  = ext << (SW + 28 - sh);               // bits shifted out -> sticky
        keep   = kq[25:2];
        guard  = kq[1];
        sticky = kq[0] | (|lostv);
        inc    = guard & (sticky | keep[0]);          // round-to-nearest-even
        rounded = {1'b0, keep} + {24'd0, inc};
        if (e_b >= 1) base = e_b - 1; else base = 9'd0;
        packed = {base, 23'd0} + {7'd0, rounded};     // carry bumps the exponent automatically

        if (s2_special)                res_c = s2_spec;
        else if (s2_bypass)            res_c = (s2_acc[30:0] == 31'd0) ? {s2_acc[31] & s2_allnegz, 31'd0} : s2_acc;
        else if (s2_mag == {SW{1'b0}}) res_c = 32'd0;
        else if (packed[31:23] >= 9'd255) res_c = {s2_sign, 31'h7F800000};
        else                           res_c = {s2_sign, packed[30:0]};
    end

    always @(posedge clk) begin
        if (!rst_n) out_valid <= 1'b0;
        else        out_valid <= s2_valid;
        result <= res_c;
    end
endmodule
