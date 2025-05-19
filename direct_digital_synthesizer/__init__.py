"""
=========
direct_digital_synthesizer
=========

direct_digital_synthesizer model template The System Development Kit
Used as a template for all TheSyDeKick Entities.

Current docstring documentation style is Numpy
https://numpydoc.readthedocs.io/en/latest/format.html

This text here is to remind you that documentation is important.
However, youu may find it out the even the documentation of this 
entity may be outdated and incomplete. Regardless of that, every day 
and in every way we are getting better and better :).

Initially written by Marko Kosunen, marko.kosunen@aalto.fi, 2017.

"""

import os
import sys
if not (os.path.abspath('../../thesdk') in sys.path):
    sys.path.append(os.path.abspath('../../thesdk'))

from thesdk import *
from rtl import *
from spice import *
import plot_format
plot_format.set_style('isscc')

import numpy as np

class direct_digital_synthesizer(rtl,thesdk):

    def __init__(self,*arg): 
        self.print_log(type='I', msg='Initializing %s' %(__name__)) 
        self.proplist = [ 'Rs' ];    # Properties that can be propagated from parent
        self.Rs =  100e6;            # Sampling frequency
        self.IOS=Bundle()            # Pointer for input data
        self.IOS.Members['io_A']=IO()   # Pointer for input data
        self.IOS.Members['io_B']= IO()
        self.IOS.Members['control_write']= IO() 
        self.model='py';             # Can be set externally, but is not propagated
        self.models=['py','icarus']
        self.lang = 'sv'
        self.shape = 'sine'
        #self.controller = DDS_controller(self)
        self.par= False              # By default, no parallel processing
        self.queue= []               # By default, no parallel processing

        # DDS parameters:
        self.acc_width = 32      # bits in the phase accumulator
        self.lut_bits  = 10      # log2(# entries in the sine table)
        self.out_width = 12      # output amplitude bits

        # state
        self.phase0 = 0          # initial phase word
        self.lut    = None       # will hold our sine‑table

        if len(arg)>=1:
            parent=arg[0]
            self.copy_propval(parent,self.proplist)
            self.parent = parent

        self.init()

    def init(self):
        # build a sine LUT with signed integer outputs
        #N = 1 << self.lut_bits
        #angles = 2 * np.pi * np.arange(N) / N
        #angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        # scale to full‑scale of out_width:
        #amp = (2**(self.out_width - 1) - 1)
        #table = np.round(amp * np.sin(angles)).astype(np.int32)
        #self.lut = table
        self.build_lut(self.shape)

    def build_lut(self, waveform='sine'):
        N = 1 << self.lut_bits
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        amp = (2**(self.out_width - 1) - 1)

        if waveform == 'sine':
            table = np.round(amp * np.sin(angles))
        elif waveform == 'square':
            table = amp * np.sign(np.sin(angles))
        elif waveform == 'triangle':
            table = amp * (2/np.pi) * np.arcsin(np.sin(angles))
        else:
            raise ValueError('Unsupported waveform')

        self.lut = table.astype(np.int32)

    def main(self):
        '''Guideline. Isolate python processing to main method.
        
        To isolate the internal processing from IO connection assigments, 
        The procedure to follow is
        1) Assign input data from input to local variable
        2) Do the processing
        3) Assign local variable to output

        '''
        #inval=self.IOS.Members['io_A'].Data
        #out=-inval
        #if self.par:
        #    self.queue.put(out)
        #self.IOS.Members['io_B'].Data=out
        
        # DDS
        # 1) take the tuning words as a 1‑D array of unsigned ints
        tw = self.IOS.Members['io_A'].Data.flatten().astype(np.uint64)

        # 2) cumulative‐sum to simulate a running phase accumulator
        # add the carry‑in from the previous run, then wrap
        phases = (np.cumsum(tw, dtype=np.uint64) + self.phase0) & ((1 << self.acc_width) - 1)

        # 3) save the last phase for next invocation (if you re‑call main)
        self.phase0 = int(phases[-1])

        # 4) drop the top bits to index the LUT
        addrs = phases >> (self.acc_width - self.lut_bits)

        # 5) look up the sine value
        samples = self.lut[addrs]    # shape = (num_samples,)

        # 6) reshape to a column vector and drive the output IO
        self.IOS.Members['io_B'].Data = samples.reshape(-1, 1)

    # Fix unsigned interpretation issue for signed 12-bit data
    def unsigned_to_signed(self, x, bits):
        mask = (1 << bits)
        return ((x + mask//2) % mask) - mask//2

    def run(self,*args):
        '''Guideline: Define model depencies of executions in `run` method.

        '''
        """
        Dispatch between Python ('py') and RTL ('sv','icarus','verilator') models.
        """
        
        # allow passing a data queue for Python mode
        if args:
            self.par   = True
            self.queue = args[0]

        # 1) Python simulation
        if self.model == 'py':
            return self.main()

        elif self.model in [ 'sv', 'icarus']:
            inputlist     = ["io_A"]
            #output_phase  = [f"io_B_phase_{i}" for i in range(32)]
            output_ampl   = ["io_B"]
            #for i in output_ampl:
            #    output_ampl[i] = self.unsigned_to_signed(output_ampl[i], self.out_width)

            # tuning word input
            f1 = rtl_iofile(self, name='io_A', dir='in', iotype='sample',
                    ionames=inputlist, datatype='uint')

            # phase output
            #f2 = rtl_iofile(self, name='io_B_phase', dir='out', iotype='sample',
            #        ionames=output_phase, datatype='int')

             # waveform / amplitude output
            f3 = rtl_iofile(self, name='io_B', dir='out', iotype='sample',
                    ionames=output_ampl, datatype='int')

            # sync type depending on HDL
            if self.lang == 'sv':
                f1.rtl_io_sync = '@(negedge clock)'
                #f2.rtl_io_sync = '@(negedge clock)'
                f3.rtl_io_sync = '@(negedge clock)'
            elif self.lang == 'vhdl':
                f1.rtl_io_sync = 'falling_edge(clock)'
                #f2.rtl_io_sync = 'falling_edge(clock)'
                f3.rtl_io_sync = 'falling_edge(clock)'

            # RTL parameters to Verilog (if your testbench expects them)
            self.rtlparameters = dict([
                    ('g_Rs',       ('real', self.Rs)),
                    ('g_accWidth', ('int',  self.acc_width)),
                    ('g_lutBits',  ('int',  self.lut_bits)),
                    ('g_outWidth', ('int',  self.out_width)),
            ])

            self.run_rtl()

        # 3) unsupported model
        else:
            self.print_log(type='F', msg=f"Model '{self.model}' not supported")
        #elif self.model == 'rtl':
        #    self.controller.reset()
        #    self.controller.start_datafeed()
        #    self.controller.dut.run()

    def define_io_conditions(self):
        '''This overloads the method called by run_rtl method. It defines the read/write conditions for the files

        '''
        if self.lang == 'vhdl':
            # Input A is read to verilog simulation after 'initdone' is set to 1 by controller
            self.iofile_bundle.Members['io_A'].rtl_io_condition='(initdone=\'1\')'
            # Output is read to verilog simulation when all of the outputs are valid, 
            # and after 'initdone' is set to 1 by controller
            self.iofile_bundle.Members['io_B'].rtl_io_condition_append(cond='and (initdone=\'1\')')
        if self.lang == 'sv':
            # Input A is read to verilog simulation after 'initdone' is set to 1 by controller
            self.iofile_bundle.Members['io_A'].rtl_io_condition='initdone'
            # Output is read to verilog simulation when all of the outputs are valid, 
            # and after 'initdone' is set to 1 by controller
            # causes AttributeError no attribute '_rtl_io_condition'
            #self.iofile_bundle.Members['io_B'].rtl_io_condition_append(cond='&& initdone')

if __name__=="__main__":
    import argparse
    import matplotlib.pyplot as plt
    from scipy.fft import fft, fftfreq
    from  direct_digital_synthesizer import *
    from  direct_digital_synthesizer.controller import controller as direct_digital_synthesizer_controller
    import pdb
    import math
    # Implement argument parser
    parser = argparse.ArgumentParser(description='Parse selectors')
    parser.add_argument('--show', dest='show', type=bool, nargs='?', const = True, 
            default=False,help='Show figures on screen')
    args=parser.parse_args()

    numSamples=1024
    rs=100e6
    acc_width = 32
    lut_bits = 10
    out_width = 12
    controller=direct_digital_synthesizer_controller()
    # Tuning word / accumulator width
    # DDS parameters:
    controller.acc_width = acc_width     # bits in the phase accumulator
    controller.lut_bits  = lut_bits      # log2(# entries in the sine table)
    controller.out_width = out_width     # output amplitude bits
    controller.Rs=rs               # Sampling frequency
    # Calculate tuning word for exactly one cycle in `length` steps:
    tw = int((1 << controller.acc_width) / numSamples)
    print("Tuning word: "+str(tw))
    
    #indata=np.cos(2*math.pi/length*np.arange(length)).reshape(-1,1)

    # Build a column vector of that constant tuning word:
    indata = np.full((numSamples,1), tw, dtype=np.uint64)
    print("Indata: "+ str(indata))

    controller.reset()
    controller.step_time()
    controller.start_datafeed()
    models=['py','icarus']
    # Enables VHDL testbench
    #lang='vhdl'
    lang='sv'
    duts=[]
    plotters=[]
    for model in models:
        d=direct_digital_synthesizer()
        duts.append(d) 
        d.model=model
        print("Model: ", d.model)
        d.lang = lang
        d.Rs=rs
        d.acc_width = acc_width
        d.lut_bits  = lut_bits
        d.out_width = out_width
        d.interactive_rtl = True # Enable interactive RTL simulation (GTKWave)
        d.IOS.Members['io_A'].Data=indata
        # Datafield of control_write IO is a type iofile, 
        # Method rtl.create_connectors adopts it to be iofile of dut.  
        d.IOS.Members['control_write']=controller.IOS.Members['control_write']
        d.init()
        d.run()

    for k in range(len(duts)):
        #hfont = {'fontname':'Sans'}
        #figure,axes=plt.subplots(2,1,sharex=True)
        #x = np.arange(length).reshape(-1,1)
        #axes[0].plot(x,indata)
        #axes[0].set_ylim(-1.1, 1.1)
        #axes[0].set_xlim((np.amin(x), np.amax(x)))
        #axes[0].set_ylabel('Input', **hfont,fontsize=18)
        #axes[0].grid(True)
        #axes[1].plot(x, duts[k].IOS.Members['io_B'].Data)
        #axes[1].set_ylim(-1.1, 1.1)
        #axes[1].set_xlim((np.amin(x), np.amax(x)))
        #axes[1].set_ylabel('Output', **hfont,fontsize=18)
        #axes[1].set_xlabel('Sample (n)', **hfont,fontsize=18)
        #axes[1].grid(True)
        #titlestr = "direct_digital_synthesizer model %s" %(duts[k].model) 
        #plt.suptitle(titlestr,fontsize=20)
        #plt.grid(True)
        #printstr="./inv_%s.eps" %(duts[k].model)
        #plt.show(block=False)

        outdata = duts[k].IOS.Members['io_B'].Data
        outdata = np.reshape(outdata, (-1,))
        outdata = duts[k].unsigned_to_signed(outdata.astype(np.int32), duts[k].out_width)
        print(type(outdata))
        print("Outdata: " + str(outdata))
        print("Output min/max:", np.min(outdata), np.max(outdata))
        print(outdata.shape)
        
        # Recompute some internals for visualization
        phases = (np.cumsum(indata, dtype=np.uint64) + d.phase0) & ((1 << d.acc_width) - 1)
        indices = phases >> (d.acc_width - d.lut_bits)

        # Spectrum
        #Y_db = np.abs(fft(outdata))[:len(outdata)//2]
        #Y_db = 20 * np.log10(Y / np.max(Y))  # in dB
        #f = fftfreq(len(outdata), 1/d.Rs)[:len(outdata)//2] / 1e6  # MHz

        # Compute FFT
        Y = np.abs(fft(outdata.astype(float)))[:len(outdata)//2]  # cast to float
        f = fftfreq(len(outdata), 1/d.Rs)[:len(outdata)//2] / 1e6  # MHz

        # Normalize safely
        Y_max = np.max(Y)
        if Y_max == 0:
            print("⚠️ FFT input was all-zero.")
            Y_db = np.zeros_like(Y)
        else:
            Y_db = 20 * np.log10(Y / Y_max + 1e-12)  # +eps avoids log(0)

        # Estimated output freq
        fout = (d.Rs * int(indata[0])) / (1 << d.acc_width)
        #fout = list(map(lambda fout :str(fout) + '%',fout.round(2)))
        print(f"Output frequency: {fout/1e6:.3f} MHz")

        # Plot layout: 4 stacked subplots
        hfont = {'fontname': 'Sans'}
        figure, axes = plt.subplots(4, 1, sharex=False, figsize=(10, 10))

        # 1. Input tuning word
        axes[0].plot(indata)
        axes[0].set_ylabel("Tuning Word", **hfont, fontsize=12)
        axes[0].set_title("DDS Simulation Results, Model:"+duts[k].model, fontsize=14)
        axes[0].grid(True)

        # 2. DDS output waveform
        axes[1].plot(outdata, label='DDS Output')
        # Optional: overlay ideal sine wave
        expected = (2**(d.out_width - 1) - 1) * np.sin(2 * np.pi * np.arange(len(outdata)) / len(outdata))
        axes[1].plot(expected, '--', label='Ideal Sine', alpha=0.6)
        axes[1].set_ylabel("Amplitude", **hfont, fontsize=12)
        axes[1].legend()
        axes[1].grid(True)

        # 3. LUT indices (optional debug view)
        axes[2].plot(indices)
        axes[2].set_ylabel("LUT Index", **hfont, fontsize=12)
        axes[2].grid(True)

        # 4. Spectrum
        axes[3].plot(f, Y_db)
        axes[3].set_ylabel("Magnitude (dB)", **hfont, fontsize=12)
        axes[3].set_xlabel("Frequency (MHz)", **hfont, fontsize=12)
        axes[3].grid(True)

        plt.tight_layout()
        plt.show()
        #figure.savefig(printstr, format='eps', dpi=300);

    #This is here to keep the images visible
    #For batch execution, you should comment the following line 
    input()
    #This is to have exit status for successful execution
    sys.exit(0)

