launch: main
	sudo singularity exec --writable main sh

main: main.sif
	sudo singularity build --sandbox main main.sif

main.sif: main.def
	sudo singularity build main.sif main.def 

clear:
	sudo rm -rf main main.sif