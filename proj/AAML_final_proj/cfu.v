module Cfu (
  input               cmd_valid,
  output              cmd_ready,
  input      [9:0]    cmd_payload_function_id,
  input      [31:0]   cmd_payload_inputs_0,
  input      [31:0]   cmd_payload_inputs_1,
  output reg          rsp_valid,
  input               rsp_ready,
  output reg [31:0]   rsp_payload_outputs_0,
  input               reset,
  input               clk
);
  reg [15:0] InputOffset;

  // SIMD multiply step:
  wire signed [16:0] prod_0, prod_1, prod_2, prod_3;
  assign prod_0 =  ($signed(cmd_payload_inputs_0[7 : 0]) + $signed(InputOffset))
                  * $signed(cmd_payload_inputs_1[7 : 0]);
  assign prod_1 =  ($signed(cmd_payload_inputs_0[15: 8]) + $signed(InputOffset))
                  * $signed(cmd_payload_inputs_1[15: 8]);
  assign prod_2 =  ($signed(cmd_payload_inputs_0[23:16]) + $signed(InputOffset))
                  * $signed(cmd_payload_inputs_1[23:16]);
  assign prod_3 =  ($signed(cmd_payload_inputs_0[31:24]) + $signed(InputOffset))
                  * $signed(cmd_payload_inputs_1[31:24]);

  wire signed [31:0] sum_prods;
  assign sum_prods = prod_0 + prod_1 + prod_2 + prod_3;

// ------------------------------------------------------------------------------
    wire signed [31:0] value;
    reg signed [31:0] minValue;
    reg signed [31:0] maxValue;
    reg signed [31:0] clampedValue;
    assign value = cmd_payload_inputs_0;

    always @* begin
        if (value <= minValue)
            clampedValue = minValue;
        else if (value >= maxValue)
            clampedValue = maxValue;
        else
            clampedValue = value;
    end
//-------------------------------------------------------------------------------
  reg [15:0] filter_offset, input_offset;

  // SIMD multiply step:
  wire signed [16:0] prod_fc_0, prod_fc_1, prod_fc_2, prod_fc_3;
  assign prod_fc_0 =  ($signed(cmd_payload_inputs_0[7 : 0]) + $signed(filter_offset))
                    * ($signed(cmd_payload_inputs_1[7 : 0]) + $signed(input_offset));
  assign prod_fc_1 =  ($signed(cmd_payload_inputs_0[15: 8]) + $signed(filter_offset))
                    * ($signed(cmd_payload_inputs_1[15: 8]) + $signed(input_offset));
  assign prod_fc_2 =  ($signed(cmd_payload_inputs_0[23:16]) + $signed(filter_offset))
                    * ($signed(cmd_payload_inputs_1[23:16]) + $signed(input_offset));
  assign prod_fc_3 =  ($signed(cmd_payload_inputs_0[31:24]) + $signed(filter_offset))
                    * ($signed(cmd_payload_inputs_1[31:24]) + $signed(input_offset));

  wire signed [31:0] sum_prods_fc;
  assign sum_prods_fc = prod_fc_0 + prod_fc_1 + prod_fc_2 + prod_fc_3;

  // Only not ready for a command when we have a response.
  assign cmd_ready = ~rsp_valid;

  always @(posedge clk) begin
    if (reset) begin
      rsp_payload_outputs_0 <= 32'b0;
      rsp_valid <= 1'b0;
      InputOffset <= 0;
    end else if (rsp_valid) begin
      // Waiting to hand off response to CPU.
      rsp_valid <= ~rsp_ready;
    end else if (cmd_valid) begin
      rsp_valid <= 1'b1;
      // Accumulate step:
      case (cmd_payload_function_id[9:3])
        2'b000_0000: begin
          InputOffset <= InputOffset;
          rsp_payload_outputs_0 <= rsp_payload_outputs_0 + sum_prods;
        end
        2'b000_0001: begin
          InputOffset <= cmd_payload_inputs_0[15:0];
          rsp_payload_outputs_0 <= 0'b0;
        end
        7'd3: begin
          filter_offset <= cmd_payload_inputs_0;
          input_offset <= cmd_payload_inputs_1;
          rsp_payload_outputs_0 <= 0'b0;
        end
        7'd4: begin
          filter_offset <= filter_offset;
          input_offset <= input_offset;
          rsp_payload_outputs_0 <= rsp_payload_outputs_0 + sum_prods_fc;
        end
        7'd6: begin
          minValue = cmd_payload_inputs_0;
          maxValue = cmd_payload_inputs_1;
        end
        7'd7: begin
          rsp_payload_outputs_0 <= clampedValue;
        end
        default: begin
          InputOffset <= InputOffset;
          rsp_payload_outputs_0 <= 0'b0;
          rsp_payload_outputs_0 <= rsp_payload_outputs_0 <= 0'b0 + sum_prods_fc;

        end
      endcase
    end
  end
endmodule