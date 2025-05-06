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
import plot_format
plot_format.set_style('isscc')

import numpy as np

class direct_digital_synthesizer(rtl,thesdk):

    def __init__(self,*arg): 
        self.print_log(type='I', msg='Inititalizing %s' %(__name__)) 
        self.proplist = [ 'Rs' ];    # Properties that can be propagated from parent
        self.Rs =  100e6;            # Sampling frequency
        self.IOS=Bundle()            # Pointer for input data
        self.IOS.Members['A']=IO()   # Pointer for input data
        self.IOS.Members['Z']= IO()
        self.model='py';             # Can be set externally, but is not propagated
        self.models=['py','rtl']
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
            self.parent =parent;

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
        self.build_lut('square')

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
        
        To isolate the interna processing from IO connection assigments, 
        The procedure to follow is
        1) Assign input data from input to local variable
        2) Do the processing
        3) Assign local variable to output

        '''
        #inval=self.IOS.Members['A'].Data
        #out=-inval
        #if self.par:
        #    self.queue.put(out)
        #self.IOS.Members['Z'].Data=out
        
        # DDS
        # 1) take the tuning words as a 1‑D array of unsigned ints
        tw = self.IOS.Members['A'].Data.flatten().astype(np.uint64)

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
        self.IOS.Members['Z'].Data = samples.reshape(-1, 1)

    def run(self,*arg):
        '''Guideline: Define model depencies of executions in `run` method.

        '''
        if len(arg)>0:
            self.par=True      #flag for parallel processing
            self.queue=arg[0]  #multiprocessing.queue as the first argument
        if self.model=='py':
            self.main()
        #elif self.model == 'rtl':
        #    self.controller.reset()
        #    self.controller.start_datafeed()
        #    self.controller.dut.run()

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

    length=1024
    rs=100e6
    acc_w = 32
    # Calculate tuning word for exactly one cycle in `length` steps:
    tw = int((1 << acc_w) / length)
    print("Tuning word: "+str(tw))
    
    #indata=np.cos(2*math.pi/length*np.arange(length)).reshape(-1,1)

    # Build a column vector of that constant tuning word:
    indata = np.full((length,1), tw, dtype=np.uint64)
    print("Indata: "+ str(indata))

    models=['py','rtl']
    duts=[]
    plotters=[]
    for model in models:
        d=direct_digital_synthesizer()
        duts.append(d) 
        d.model=model
        d.Rs=rs
        d.acc_width = acc_w
        d.lut_bits  = 10
        d.out_width = 12
        d.IOS.Members['A'].Data=indata
        d.init()
        d.run()

    for k in range(len(duts)):
        hfont = {'fontname':'Sans'}
        #figure,axes=plt.subplots(2,1,sharex=True)
        #x = np.arange(length).reshape(-1,1)
        #axes[0].plot(x,indata)
        #axes[0].set_ylim(-1.1, 1.1);
        #axes[0].set_xlim((np.amin(x), np.amax(x)));
        #axes[0].set_ylabel('Input', **hfont,fontsize=18);
        #axes[0].grid(True)
        #axes[1].plot(x, duts[k].IOS.Members['Z'].Data)
        #axes[1].set_ylim(-1.1, 1.1);
        #axes[1].set_xlim((np.amin(x), np.amax(x)));
        #axes[1].set_ylabel('Output', **hfont,fontsize=18);
        #axes[1].set_xlabel('Sample (n)', **hfont,fontsize=18);
        #axes[1].grid(True)
        #titlestr = "Myentity model %s" %(duts[k].model) 
        #plt.suptitle(titlestr,fontsize=20);
        #plt.grid(True);
        #printstr="./inv_%s.eps" %(duts[k].model)
        #plt.show(block=False);

        outdata = duts[k].IOS.Members['Z'].Data

        print("Outdata: " + str(outdata))
        print("Output min/max:", np.min(outdata), np.max(outdata))
        
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
        axes[0].set_title("DDS Simulation Results", fontsize=14)
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
        figure.savefig(printstr, format='eps', dpi=300);

    #This is here to keep the images visible
    #For batch execution, you should comment the following line 
    input()
    #This is to have exit status for successful execution
    sys.exit(0)

