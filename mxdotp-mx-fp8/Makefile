.PHONY: model vectors study sim clean
model:
	python3 scripts/rtl_model.py
vectors:
	python3 scripts/gen_vectors.py
study:
	python3 scripts/accuracy_study.py
sim: vectors
	./sim/run_iverilog.sh
clean:
	rm -rf build vivado/mxdotp_proj vivado/reports sim/vectors/*.mem *.jou *.log
