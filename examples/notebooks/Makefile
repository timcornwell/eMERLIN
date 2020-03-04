
NBs = eMERLIN_3C277_1_pipeline.ipynb

RSTs=  $(NBs:.ipynb=.rst)

JUPYTER ?= jupyter
TIMEOUT = 3600

.PHONY: rst
rst: $(RSTs)

%.rst: %.ipynb
	jupyter nbconvert --execute --ExecutePreprocessor.timeout=$(TIMEOUT) --to rst $<

%.html: %.rst
	${RST2HTML} $(<:.rstw=.rst)  > $@

.PHONY: clean
clean:
	rm -f ${RSTs}
