# Main target at the top
all: slimmer

# Make sure to add all header files here (wildcards work too)
feedpanda.h: slimmer.cpp flatfile.h
	./scripts/feedpanda.pl ${CMSSW_BASE}/src/PandaTree/defs/nanoaod.def ${PWD} slimmer.cpp feedpanda.h

flatfile.h: flatfile.txt
	scripts/maketree.py flatfile.txt

slimmer: feedpanda.h flatfile.h slimmer.cpp ${CMSSW_BASE}/src/PandaTree/Objects/interface/Event.h
	g++ -I${CMSSW_BASE}/src -L${CMSSW_BASE}/lib/${SCRAM_ARCH} \
	-lPandaTreeFramework -lPandaTreeObjects \
	`root-config --glibs` `root-config --cflags` \
	-o slimmer slimmer.cpp

.PHONY: test

# Can put arbirary number of input files. Just output file name needs to be last
test: slimmer
	test ! -f test.root || rm test.root
	./slimmer /data/t3home000/dabercro/scratch/nano/99031B88-51E4-6845-A025-AD32A333C178.root test.root
