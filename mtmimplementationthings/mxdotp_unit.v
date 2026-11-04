//==============================================================================
// mxdotp_unit.v — Synthesizable 3-Stage Pipelined MXFP8 Dot-Product Unit
// Ready for Cadence ncsim/Xcelium, Genus (Synthesis), and Innovus (PnR)
//==============================================================================
`timescale 1ns / 1ps

//------------------------------------------------------------------------------
// 1. fp8_decode: Decodes one FP8 element (E5M2 or E4M3)
//------------------------------------------------------------------------------
module fp8_decode (
    input  wire [7:0] x,
    input  wire       fmt,       // 0: E5M2, 1: E4M3
    output wire       sign,
    output wire [3:0] mant,
    output wire [4:0] ebias,
    output wire       is_zero,
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

//------------------------------------------------------------------------------
// 2. mxdotp_dot8: Stage 1 Combinational 8-Element Dot Product
//------------------------------------------------------------------------------
module mxdotp_dot8 #(
    parameter TW = 72
) (
    input  wire [63:0]          a,
    input  wire [63:0]          b,
    input  wire [7:0]           xa,
    input  wire [7:0]           xb,
    input  wire                 fmt,
    output reg  [TW-1:0]        t,
    output wire signed [9:0]    s,
    output wire                 nan_f,
    output wire                 pinf_f,
    output wire                 ninf_f,
    output wire                 allnegz
);
    wire [8*TW-1:0] terms;
    wire [7:0] l_nan, l_inf, l_neg, l_zero;

    genvar i;
    generate
        for (i = 0; i < 8; i = i + 1) begin : lane
            wire       sa, sb, za, zb, na, nb, ia, ib;
            wire [3:0] ma, mb;
            wire [4:0] ea, eb;

            fp8_decode u_da (
                .x(a[8*i +: 8]), .fmt(fmt), .sign(sa), .mant(ma),
                .ebias(ea), .is_zero(za), .is_nan(na), .is_inf(ia)
            );
            fp8_decode u_db (
                .x(b[8*i +: 8]), .fmt(fmt), .sign(sb), .mant(mb),
                .ebias(eb), .is_zero(zb), .is_nan(nb), .is_inf(ib)
            );

            wire          fin = ~(na | nb | ia | ib);
            wire [7:0]    p   = ma * mb;
            wire [5:0]    pos = {1'b0, ea} + {1'b0, eb} - 6'd2;
            wire [TW-1:0] pm  = fin ? ({{(TW-8){1'b0}}, p} << pos) : {TW{1'b0}};

            assign terms[i*TW +: TW] = (sa ^ sb) ? (~pm + 1'b1) : pm;

            assign l_nan[i]  = na | nb | (ia & zb) | (ib & za);
            assign l_inf[i]  = (ia | ib) & ~l_nan[i];
            assign l_neg[i]  = sa ^ sb;
            assign l_zero[i] = fin & (za | zb);
        end
    endgenerate

    // Balanced adder tree for synthesis timing
    wire [TW-1:0] sum_l1_0 = terms[0*TW +: TW] + terms[1*TW +: TW];
    wire [TW-1:0] sum_l1_1 = terms[2*TW +: TW] + terms[3*TW +: TW];
    wire [TW-1:0] sum_l1_2 = terms[4*TW +: TW] + terms[5*TW +: TW];
    wire [TW-1:0] sum_l1_3 = terms[6*TW +: TW] + terms[7*TW +: TW];

    wire [TW-1:0] sum_l2_0 = sum_l1_0 + sum_l1_1;
    wire [TW-1:0] sum_l2_1 = sum_l1_2 + sum_l1_3;

    always @* begin
        t = sum_l2_0 + sum_l2_1;
    end

    assign s       = $signed({2'b00, xa}) + $signed({2'b00, xb}) - 10'sd254;
    assign nan_f   = (|l_nan) | (xa == 8'hFF) | (xb == 8'hFF);
    assign pinf_f  = |(l_inf & ~l_neg);
    assign ninf_f  = |(l_inf &  l_neg);
    assign allnegz = &(l_zero & l_neg);
endmodule

//------------------------------------------------------------------------------
// 3. mxdotp_addround: Stages 2 (Align/Add) and 3 (Normalize/Round/Pack)
//------------------------------------------------------------------------------
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
    localparam OT   = 32;
    localparam DMAX = 73;
    localparam SW   = 132;

    //----------------------------------------------------------------------
    // STAGE 2 : Classify, Align, Add
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
    reg  signed [11:0]  ea_eff, s_i, d, pa, kk;

    always @* begin
        // Default initializations to prevent latch inference in Genus
        ea_eff = (acc_e == 8'd0) ? 12'sd1 : $signed({4'd0, acc_e});
        s_i    = {{2{s[9]}}, s};
        d      = ea_eff - s_i - 12'sd116;
        pa     = 12'sd32 + d;
        kk     = 12'sd0;
        Tw     = {{(SW-TW){t[TW-1]}}, t} << OT;
        Au     = {{(SW-24){1'b0}}, A};
        wexp_c = s_i - 12'sd34 - 12'sd32;

        if (A == 24'd0) begin
            Au = {SW{1'b0}};
        end else if (d > 12'sd73) begin
            Au     = Au << (OT + DMAX);
            Tw     = t[TW-1] ? {SW{1'b1}} : {{(SW-1){1'b0}}, 1'b1};
            wexp_c = ea_eff - 12'sd150 - 12'sd32 - 12'sd73;
        end else if (pa >= 12'sd0) begin
            Au = Au << pa[7:0];
        end else begin
            kk = -pa;
            if (kk >= 12'sd24) begin
                Au = {{(SW-1){1'b0}}, 1'b1};
            end else begin
                Au = Au >> kk[4:0];
                if ((A & ((24'd1 << kk[4:0]) - 24'd1)) != 24'd0)
                    Au[0] = 1'b1;
            end
        end

        Wsum   = (acc_s ? (~Au + 1'b1) : Au) + Tw;
        sign_c = Wsum[SW-1];
        mag_c  = sign_c ? (~Wsum + 1'b1) : Wsum;
    end

    // Stage 2 Pipeline Registers (Async Reset + Clock Gating Friendly)
    reg                s2_valid, s2_sign, s2_bypass, s2_special, s2_allnegz;
    reg  [SW-1:0]      s2_mag;
    reg  signed [11:0] s2_wexp;
    reg  [31:0]        s2_acc, s2_spec;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s2_valid   <= 1'b0;
            s2_mag     <= {SW{1'b0}};
            s2_sign    <= 1'b0;
            s2_wexp    <= 12'sd0;
            s2_bypass  <= 1'b0;
            s2_special <= 1'b0;
            s2_spec    <= 32'd0;
            s2_acc     <= 32'd0;
            s2_allnegz <= 1'b0;
        end else begin
            s2_valid <= in_valid;
            if (in_valid) begin
                s2_mag     <= mag_c;
                s2_sign    <= sign_c;
                s2_wexp    <= wexp_c;
                s2_bypass  <= t_zero;
                s2_special <= special;
                s2_spec    <= spec_res;
                s2_acc     <= acc;
                s2_allnegz <= allnegz;
            end
        end
    end

    //----------------------------------------------------------------------
    // STAGE 3 : Normalize, Round (RNE), Pack
    //----------------------------------------------------------------------
    reg  [31:0]        res_c;
    reg  signed [11:0] p, e_b, rA, rB, r, sh;
    integer            jj;
    reg  [SW+27:0]     ext, kq, lostv;
    reg  [23:0]        keep;
    reg                guard, sticky, inc;
    reg  [24:0]        rounded;
    reg  [8:0]         base;
    reg  [31:0]        packed;
    reg  [7:0]         sh_amt, lost_sh;

    always @* begin
        p = 12'sd0;
        for (jj = 0; jj < SW; jj = jj + 1) begin
            if (s2_mag[jj]) p = jj[11:0];
        end

        e_b = p + s2_wexp + 12'sd127;
        rA  = p - 12'sd23;
        rB  = -12'sd149 - s2_wexp;
        r   = (rA > rB) ? rA : rB;
        sh  = r + 12'sd26;
        if (sh > 12'sd160) sh = 12'sd160;
        if (sh < 12'sd0)   sh = 12'sd0;

        sh_amt  = sh[7:0];
        lost_sh = 8'd160 - sh_amt;

        ext     = {s2_mag, 28'd0};
        kq      = ext >> sh_amt;
        lostv   = (sh_amt == 8'd0) ? {160{1'b0}} : (ext << lost_sh);
        keep    = kq[25:2];
        guard   = kq[1];
        sticky  = kq[0] | (|lostv);
        inc     = guard & (sticky | keep[0]);
        rounded = {1'b0, keep} + {24'd0, inc};

        if (e_b >= 12'sd1) base = e_b[8:0] - 9'd1;
        else               base = 9'd0;

        packed = {base, 23'd0} + {7'd0, rounded};

        if (s2_special)
            res_c = s2_spec;
        else if (s2_bypass)
            res_c = (s2_acc[30:0] == 31'd0) ? {s2_acc[31] & s2_allnegz, 31'd0} : s2_acc;
        else if (s2_mag == {SW{1'b0}})
            res_c = 32'd0;
        else if (packed[31:23] >= 9'd255)
            res_c = {s2_sign, 31'h7F800000};
        else
            res_c = {s2_sign, packed[30:0]};
    end

    // Stage 3 Output Registers
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
            result    <= 32'd0;
        end else begin
            out_valid <= s2_valid;
            if (s2_valid) begin
                result <= res_c;
            end
        end
    end
endmodule

//------------------------------------------------------------------------------
// 4. mxdotp_unit: Top-Level Module
//------------------------------------------------------------------------------
module mxdotp_unit (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        in_valid,
    input  wire        fmt,          // 0: E5M2, 1: E4M3
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
        .t(t_c), .s(s_c), .nan_f(nan_c), .pinf_f(pinf_c), .ninf_f(ninf_c), .allnegz(nz_c)
    );

    // Stage 1 Pipeline Registers (Async Reset + Clock Gating Friendly)
    reg                s1_valid, s1_nan, s1_pinf, s1_ninf, s1_nz;
    reg [TW-1:0]       s1_t;
    reg signed [9:0]   s1_s;
    reg [31:0]         s1_acc;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s1_valid <= 1'b0;
            s1_t     <= {TW{1'b0}};
            s1_s     <= 10'sd0;
            s1_acc   <= 32'd0;
            s1_nan   <= 1'b0;
            s1_pinf  <= 1'b0;
            s1_ninf  <= 1'b0;
            s1_nz    <= 1'b0;
        end else begin
            s1_valid <= in_valid;
            if (in_valid) begin
                s1_t    <= t_c;
                s1_s    <= s_c;
                s1_acc  <= acc;
                s1_nan  <= nan_c;
                s1_pinf <= pinf_c;
                s1_ninf <= ninf_c;
                s1_nz   <= nz_c;
            end
        end
    end

    mxdotp_addround #(.TW(TW)) u_addround (
        .clk(clk), .rst_n(rst_n), .in_valid(s1_valid),
        .t(s1_t), .s(s1_s), .acc(s1_acc),
        .nan_f(s1_nan), .pinf_f(s1_pinf), .ninf_f(s1_ninf), .allnegz(s1_nz),
        .out_valid(out_valid), .result(result)
    );
endmodule
