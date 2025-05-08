# Default model
MODEL ?= sv

#.PHONY: all sim chisel clean
.PHONY: all sim doc clean

#all: chisel sim
all: sim

#sim: chisel
sim:
	cd /home/kkaivola/ICDesignProd/thesdk_template/Entities/direct_digital_synthesizer/direct_digital_synthesizer && \
	python3 __init__.py || (echo "make sim failed 13427?"; exit 1)

show:
	cd /home/kkaivola/ICDesignProd/thesdk_template/Entities/direct_digital_synthesizer/direct_digital_synthesizer && \
	python3 __init__.py --show || (echo "make sim failed 13427?"; exit 1)

chisel:
	cd /home/kkaivola/ICDesignProd/thesdk_template/Entities/direct_digital_synthesizer/chisel && \
	make

doc:
	cd /home/kkaivola/ICDesignProd/thesdk_template/Entities/direct_digital_synthesizer/doc && \
	make html || (echo "make sim failed"; exit 1)

clean:
	#cd /home/kkaivola/ICDesignProd/thesdk_template/Entities/direct_digital_synthesizer/chisel && \
	#make clean && \
	rm -rf /home/kkaivola/ICDesignProd/thesdk_template/Entities/direct_digital_synthesizer/simulations/* 
   
