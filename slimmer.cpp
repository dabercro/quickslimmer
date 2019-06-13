#include "feedpanda.h"
#include "flatfile.h"

#include "TH1F.h"

int main(int argc, char** argv) {

  if (argc < 3) {
    std::cout << argv[0] << " INPUTS OUTPUT" << std::endl;
    return 1;
  }

  flatfile output {argv[argc - 1]};
  TH1F all_hist {"htotal", "htotal", 1, -1, 1};

  // Loop over all input files
  for (int i_file = 1; i_file < argc - 1; i_file++) {

    std::cout << "Running over file " << argv[i_file]
              << " (" << i_file << "/" << (argc - 2) << ")" << std::endl;

    // Get the PandaTree
    TFile input {argv[i_file]};
    auto* events_tree = static_cast<TTree*>(input.Get("events"));
    panda::Event event;
    // This sets the branch statuses
    feedpanda(event, events_tree);
    auto nentries = events_tree->GetEntries();

    // Loop over tree
    for(decltype(nentries) entry = 0; entry != nentries; ++entry) {

      event.getEntry(*events_tree, entry);
      all_hist.Fill(0.0, event.weight);

      // Puts things back to default values (0 for now)
      output.reset();

      for (auto& mu : event.muons)
        output.set_muons(mu);

      for (auto& jet : event.chsAK4Jets)
        output.set_jets(jet);

      for (auto& cand : event.pfCandidates)
        output.set_pfcands(cand);

      // Writes to tree
      output.fill();

    }
  }

  output.write(&all_hist);
  return 0;

}
