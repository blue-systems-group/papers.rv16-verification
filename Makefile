OUT=out

all:
	mkdir -p $(OUT)
	pdflatex -aux-directory=$(OUT) -output-directory=$(OUT) paper.tex
	bibtex $(OUT)/paper.aux
	pdflatex -aux-directory=$(OUT) -output-directory=$(OUT) paper.tex
	pdflatex -aux-directory=$(OUT) -output-directory=$(OUT) paper.tex

clean:
	rm -rfv $(OUT)
