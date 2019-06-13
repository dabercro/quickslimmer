# Main target at the top
all: slimmer

# Make sure to add all header files here (wildcards work too)
feedpanda.h: slimmer.cpp flatfile.h
	scripts/feedpanda.pl ${CMSSW_BASE}/src/PandaTree/defs/panda.def flatfile.h slimmer.cpp feedpanda.h

flatfile.h: flatfile.txt
	scripts/maketree.py flatfile.txt

slimmer: feedpanda.h flatfile.h slimmer.cpp
	g++ -I${CMSSW_BASE}/src -L${CMSSW_BASE}/lib/${SCRAM_ARCH} \
	-lPandaTreeFramework -lPandaTreeObjects -lPandaTreeUtils \
	`root-config --glibs` `root-config --cflags` \
	-o slimmer slimmer.cpp

.PHONY: test

# Can put arbirary number of input files. Just output file name needs to be last
test: slimmer
	test ! -f test.root || rm test.root
	./slimmer /mnt/hadoop/cms/store/user/paus/pandaf/013/WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8+RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v1+MINIAODSIM/A3* test.root
