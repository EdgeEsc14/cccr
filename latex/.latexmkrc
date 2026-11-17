# Configuración de latexmk: PDF con pdflatex, archivos auxiliares en build/
$pdf_mode = 1;
$out_dir  = 'build';
$pdflatex = 'pdflatex -interaction=nonstopmode -file-line-error -synctex=1 %O %S';
@default_files = ('main.tex');
