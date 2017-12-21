SPHINXOPTS  =
SPHINXBUILD = sphinx-build
SPHINXPROJ  = pytorch_scatter
SOURCEDIR   = source
BUILDDIR    = build

# Put it first so that `make` without argument is like `make help`.
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
