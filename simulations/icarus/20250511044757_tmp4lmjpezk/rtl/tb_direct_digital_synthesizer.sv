module tb_direct_digital_synthesizer #(
    parameter g_Rs = 100000000.0,
    parameter g_accWidth = 32,
    parameter g_lutBits = 10,
    parameter g_outWidth = 12,
    parameter g_file_io_A = "/home/kkaivola/ICDesignProd/thesdk_template/Entities/direct_digital_synthesizer/simulations/icarus/20250511044757_tmp4lmjpezk/io_A_kaK4RuVEQn7TjCf5.txt",
    parameter g_file_io_B = "/home/kkaivola/ICDesignProd/thesdk_template/Entities/direct_digital_synthesizer/simulations/icarus/20250511044757_tmp4lmjpezk/io_B_YlVZLi4Tw58pyJg7.txt",
    parameter g_file_control_write = "/home/kkaivola/ICDesignProd/thesdk_template/Entities/direct_digital_synthesizer/simulations/icarus/20250511044757_tmp4lmjpezk/control_write_FtvdAgvwo7qTZ2iS.txt"
);
//timescale 1ps this should probably be a global model parameter
//Parameter definitions
parameter integer c_Ts=1/(g_Rs*1e-12);
//Register definitions
reg clock;
reg reset;
reg [31:0] io_A;
reg initdone;

//Wire definitions
wire [11:0] io_B;

//Assignments
//Variables for the io_files
integer status_io_A, f_io_A;
initial f_io_A = $fopen(g_file_io_A,"r");
integer status_io_B, f_io_B;
initial f_io_B = $fopen(g_file_io_B,"w");
integer status_control_write, f_control_write;
time ctstamp_control_write, ptstamp_control_write, tdiff_control_write;
initial ctstamp_control_write=0;
initial ptstamp_control_write=0;
integer buffer_reset;
integer buffer_initdone;
initial f_control_write = $fopen(g_file_control_write,"r");

// Manual commands

// Generates dumpfile
initial begin
  $dumpfile("/home/kkaivola/ICDesignProd/thesdk_template/Entities/direct_digital_synthesizer/simulations/icarus/20250511044757_tmp4lmjpezk/rtl/direct_digital_synthesizer_dump.vcd");
  $dumpvars(0, tb_direct_digital_synthesizer);
end 

//DUT definition
direct_digital_synthesizer direct_digital_synthesizer (
    .clock(clock),
    .reset(reset),
    .io_A(io_A),
    .io_B(io_B)
);
//Master clock is omnipresent
always #(c_Ts/2.0) clock = !clock;

//io_out
always @(negedge clock)begin
    if ( ~$isunknown(io_B) ) begin
        $fwrite(f_io_B, 
        "%d\n",
        io_B
        );
    end
end


//Execution with parallel fork-join and sequential begin-end sections
initial #0 begin
fork
    clock = 'b0;
    io_A = 'b0;


    // Sequences enabled by initdone
    $display("Ready to read");
    while (!$feof(f_io_A)) begin
    @(negedge clock)
        if ( initdone ) begin
            status_io_A = $fscanf(f_io_A, 
            "%d\n",
            io_A
            );
        end
    end
    begin
    while(!$feof(f_control_write)) begin
        tdiff_control_write = ctstamp_control_write-ptstamp_control_write;
        #tdiff_control_write begin
            ptstamp_control_write = ctstamp_control_write;
            reset = buffer_reset;
            initdone = buffer_initdone;
            status_control_write = $fscanf(f_control_write, "%d\t%d\t%d\n",
                ctstamp_control_write,
                buffer_reset,
                buffer_initdone
            );
        end
    end
    tdiff_control_write = ctstamp_control_write-ptstamp_control_write;
    #tdiff_control_write begin
        ptstamp_control_write = ctstamp_control_write;
        reset = buffer_reset;
        initdone = buffer_initdone;
    end
    end

join

//Close the io_files
$fclose(f_io_A);
$fclose(f_io_B);
$fclose(f_control_write);


$finish;
end

endmodule