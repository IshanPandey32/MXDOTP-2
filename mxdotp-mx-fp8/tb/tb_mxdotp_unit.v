//==============================================================================
// tb_mxdotp_unit.v - self-checking testbench for the BASELINE mxdotp_unit
//   * reads golden vectors produced by scripts/gen_vectors.py (exact arithmetic)
//   * drives one vector per cycle (with random idle bubbles) and checks every output
//   * vector line layout (hex, 212 bit): {a[63:0], b[63:0], xa, xb, acc, 4'fmt, exp[31:0]}
//
// Vector file lookup: +VEC=<path>  (plusarg)  else "unit_vectors.mem" in the sim directory.
//==============================================================================
`timescale 1ns/1ps
module tb_mxdotp_unit;
    localparam MAXV = 65536;

    reg clk = 1'b0;
    always #5 clk = ~clk;

    reg          rst_n = 1'b0;
    reg          in_valid = 1'b0;
    reg          fmt = 1'b0;
    reg  [63:0]  a = 64'd0, b = 64'd0;
    reg  [7:0]   xa = 8'd0, xb = 8'd0;
    reg  [31:0]  acc = 32'd0;
    wire         out_valid;
    wire [31:0]  result;

    mxdotp_unit dut (.clk(clk), .rst_n(rst_n), .in_valid(in_valid), .fmt(fmt),
                     .a(a), .b(b), .xa(xa), .xb(xb), .acc(acc),
                     .out_valid(out_valid), .result(result));

    reg [211:0]  vec  [0:MAXV-1];
    reg [31:0]   expq [0:MAXV-1];
    reg [8*256-1:0] vecfile;
    integer nvec, i, o, errors, checked;

    // ------------------------------------------------ checker
    always @(posedge clk) begin
        if (out_valid) begin
            if (result !== expq[o]) begin
                errors = errors + 1;
                if (errors <= 20)
                    $display("[FAIL] vec %0d : got %08h expected %08h", o, result, expq[o]);
            end
            o = o + 1;
            checked = checked + 1;
        end
    end

    // ------------------------------------------------ stimulus
    initial begin
        errors = 0; o = 0; checked = 0;
        if (!$value$plusargs("VEC=%s", vecfile)) vecfile = "unit_vectors.mem";
        for (i = 0; i < MAXV; i = i + 1) vec[i] = {212{1'bx}};
        $readmemh(vecfile, vec);
        nvec = 0;
        while (nvec < MAXV && ((^vec[nvec]) !== 1'bx)) nvec = nvec + 1;
        if (nvec == 0) begin
            $display("[ERROR] no vectors loaded from %0s (add .mem as a simulation source or use +VEC=)", vecfile);
            $finish;
        end
        $display("Loaded %0d vectors", nvec);

        repeat (4) @(posedge clk);
        #1 rst_n = 1'b1;

        i = 0;
        while (i < nvec) begin
            @(posedge clk); #1;
            if (($random & 3) == 0) begin            // random bubble
                in_valid = 1'b0;
            end else begin
                {a, b, xa, xb, acc} = vec[i][211:36];
                fmt          = vec[i][32];
                expq[i]      = vec[i][31:0];
                in_valid     = 1'b1;
                i = i + 1;
            end
        end
        @(posedge clk); #1 in_valid = 1'b0;
        repeat (10) @(posedge clk);

        if (checked != nvec) begin
            errors = errors + 1;
            $display("[FAIL] checked %0d results but sent %0d vectors", checked, nvec);
        end
        if (errors == 0) $display("PASS : %0d / %0d vectors bit-exact", checked, nvec);
        else             $display("FAIL : %0d errors in %0d vectors", errors, nvec);
        $finish;
    end
endmodule
