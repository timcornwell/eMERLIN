

# Disable built in rules
.SUFFIXES:

SCs = eMERLIN_RASCIL_pipeline.py

RSTs=  $(SCs:.py=.rst)

.PHONY: rst
rst: $(RSTs)

%.rst: %.py
	cat $@.header > $@
	sed  's/^/     /'  $< >> $@

%.html: %.rst
	${RST2HTML} $(<:.rstw=.rst)  > $@

.PHONY: clean
clean:
	rm -f ${RSTs}
