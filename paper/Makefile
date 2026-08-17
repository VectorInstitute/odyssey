# Build the manuscript PDF. Requires a TeX distribution with latexmk.
#   make            -> main.pdf
#   make clean      -> remove build artifacts (keeps main.pdf)
#   make distclean  -> also remove main.pdf
.PHONY: all clean distclean
all: main.pdf
main.pdf: main.tex references.bib sn-jnl.cls sn-nature.bst
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
clean:
	latexmk -c
	rm -f main.bbl main.run.xml
distclean: clean
	rm -f main.pdf
