//==============================================================================
// tb_mxdotp_blk32.v - self-checking testbench for the IMPROVED mxdotp_blk32
//   Each vector is one 32-element MX block = 4 beats of 8 elements.
//   vector layout (hex, 596 bit):
//     {Aall[255:0], Ball[255:0], xa, xb, acc, 4'fmt, exp[31:0]}
//   Aall = {a_beat3, a_beat2, a_beat1, a_beat0}, beat0 = bits [63:0], same for B.
//   Golden = acc + 2^(Xa+Xb-254) * sum_{32} A*B  with ONE rounding (exact reference).
//==============================================================================
`timescale 1ns/1ps
module tb_mxdotp_blk32;
    localparam MAXV = 16384;

    reg clk = 1'b0;
    always #5 clk = ~clk;

    reg          rst_n = 1'b0;
    reg          in_valid = 1'b0, first = 1'b0, last = 1'b0, fmt = 1'b0;
    reg  [63:0]  a = 64'd0, b = 64'd0;
    reg  [7:0]   xa = 8'd0, xb = 8'd0;
    reg  [31:0]  acc_in = 32'd0;
    wire         out_valid, blk_err;
    wire [31:0]  result;

    mxdotp_blk32 dut (.clk(clk), .rst_n(rst_n), .in_valid(in_valid), .first(first), .last(last),
                      .fmt(fmt), .a(a), .b(b), .xa(xa), .xb(xb), .acc_in(acc_in),
                      .out_valid(out_valid), .result(result), .blk_err(blk_err));

    reg [595:0]  vec  [0:MAXV-1];
    reg [31:0]   expq [0:MAXV-1];
    reg [8*256-1:0] vecfile;
    integer nvec, i, j, o, errors, checked;
    reg [255:0] Aall, Ball;

    always @(posedge clk) begin
        if (out_valid) begin
            if (result !== expq[o] || blk_err !== 1'b0) begin
                errors = errors + 1;
                if (errors <= 20)
                    $display("[FAIL] block %0d : got %08h expected %08h err=%b", o, result, expq[o], blk_err);
            end
            o = o + 1;
            checked = checked + 1;
        end
    end

    initial begin
        errors = 0; o = 0; checked = 0;
        if (!$value$plusargs("VEC=%s", vecfile)) vecfile = "blk_vectors.mem";
        for (i = 0; i < MAXV; i = i + 1) vec[i] = {596{1'bx}};
        $readmemh(vecfile, vec);
        nvec = 0;
        while (nvec < MAXV && ((^vec[nvec]) !== 1'bx)) nvec = nvec + 1;
        if (nvec == 0) begin
            $display("[ERROR] no vectors loaded from %0s (add .mem as a simulation source or use +VEC=)", vecfile);
            $finish;
        end
        $display("Loaded %0d blocks", nvec);

        repeat (4) @(posedge clk);
        #1 rst_n = 1'b1;

        for (i = 0; i < nvec; i = i + 1) begin
            Aall = vec[i][595:340];
            Ball = vec[i][339:84];
            for (j = 0; j < 4; j = j + 1) begin
                @(posedge clk); #1;
                a        = Aall[64*j +: 64];
                b        = Ball[64*j +: 64];
                xa       = vec[i][83:76];
                xb       = vec[i][75:68];
                fmt      = vec[i][32];
                acc_in   = (j == 3) ? vec[i][67:36] : 32'hDEADBEEF;   // only sampled on last beat
                first    = (j == 0);
                last     = (j == 3);
                in_valid = 1'b1;
                if (j == 3) expq[i] = vec[i][31:0];
            end
            if (($random & 1) == 0) begin                              // optional gap between blocks
                @(posedge clk); #1 in_valid = 1'b0; first = 1'b0; last = 1'b0;
            end
        end
        @(posedge clk); #1 in_valid = 1'b0; first = 1'b0; last = 1'b0;
        repeat (10) @(posedge clk);

        if (checked != nvec) begin
            errors = errors + 1;
            $display("[FAIL] checked %0d blocks but sent %0d", checked, nvec);
        end
        if (errors == 0) $display("PASS : %0d / %0d blocks bit-exact", checked, nvec);
        else             $display("FAIL : %0d errors in %0d blocks", errors, nvec);
        $finish;
    end
endmodule
