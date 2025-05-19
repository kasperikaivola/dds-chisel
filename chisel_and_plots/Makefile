#Directories
VERILOGPATH= /home/kkaivola/ICDesignProd/thesdk_template/Entities/direct_digital_synthesizer/chisel/verilog
SCALAPATH= /home/kkaivola/ICDesignProd/thesdk_template/Entities/direct_digital_synthesizer/chisel/src/main/scala
#DEPDIR :=.depdir
#$(shell mkdir -p $(DEPDIR) >/dev/null)
$(shell mkdir -p $(VERILOGPATH) >/dev/null)
MODULES= direct_digital_synthesizer
PACKAGE=direct_digital_synthesizer

TARGETS = $(foreach name,$(MODULES), $(VERILOGPATH)/$(name).v)
TEST_TARGETS = $(foreach name,$(MODULES), test_$(name))

#Commands
SBT=sbt -J-Xmx16G -J-Xss8M

TOUCH=touch -r
vpath %.scala $(SCALAPATH)/direct_digital_synthesizer
.PHONY: all help doc clean $(MODULES)


all: $(TARGETS) $(TEST_TARGETS)


#Recipes for individual modules
direct_digital_synthesizer: $(VERILOGPATH)/direct_digital_synthesizer.v

#Test recipes for in
.PHONY: test_direct_digital_synthesizer
test_direct_digital_synthesizer:
	$(SBT) 'testOnly direct_digital_synthesizer.direct_digital_synthesizerSpec'


#Template for conditional makes
#Figure out the naming conventions later. Package is lowercase, class equals main method
#Does not track dependencies from scala source
$(VERILOGPATH)/%.v : %.scala  
	$(eval package:=$(basename $(notdir $@)))
	$(eval class:=$(basename $(notdir $@)))
	$(eval testbenchfile:=$(dir $<)tb_$(notdir $<))
	$(SBT) -Dwaveform=$(WAVEFORM) 'runMain $(PACKAGE).$(class) -td $(VERILOGPATH)'
	@#Test if testbech generator exists and compile it
	@if [ -f $(testbenchfile) ]; then \
		$(SBT) -Dwaveform=$(WAVEFORM) 'runMain $(package).tb_$(class) -td $(VERILOGPATH)'; \
	fi
clean:
	rm -f $(VERILOGPATH)/*.v
	rm -f $(VERILOGPATH)/*.anno
	rm -f $(VERILOGPATH)/*.fir
	rm -rf $(VERILOGPATH)
	#rm -rf $(DEPDIR)

MMOD ?=
memmap:
	cp $(VERILOGPATH)/$(MMOD).v  $(VERILOGPATH)/$(MMOD)_unmapped.orig.v
	$(SBT) 'runMain $(MMOD).$(MMOD) -td $(VERILOGPATH) --infer-rw $(MMOD) --repl-seq-mem -c:$(MMOD):-o:$(VERILOGPATH)/$(MMOD).conf'

#Generate cleanup recipes for individual modules
.PHONY: clean_direct_digital_synthesizer
clean_direct_digital_synthesizer:
	rm -f $(VERILOGPATH)/direct_digital_synthesizer.v
	rm -f $(VERILOGPATH)/direct_digital_synthesizer.anno
	rm -f $(VERILOGPATH)/direct_digital_synthesizer.fir
	rm -f $(VERILOGPATH)/direct_digital_synthesizer_memmapped.conf
	rm -f $(VERILOGPATH)/direct_digital_synthesizer_memmapped.v

help:
	@echo "configured modules are:";
	@for i in $(MODULES) ; do \
	   echo $$i; \
	done

doc:
	sbt doc

