#include "feedpanda.h"
#include "flatfile.h"

int main(int argc, char** argv) {

  if (argc < 3) {
    std::cout << argv[0] << " INPUTS OUTPUT" << std::endl;
    return 1;
  }

  flatfile output {argv[argc - 1]};

  // Loop over all input files
  for (int i_file = 1; i_file < argc - 1; i_file++) {

    std::cout << "Running over file " << argv[i_file]
              << " (" << i_file << "/" << (argc - 2) << ")" << std::endl;

    // Get the PandaTree
    TFile input {argv[i_file]};
    auto* events_tree = static_cast<TTree*>(input.Get("Events"));
    panda::Event event;
    // This sets the branch statuses
    crombie::feedpanda(event, events_tree);
    auto nentries = events_tree->GetEntries();

    // Loop over tree
    for(decltype(nentries) entry = 0; entry != nentries; ++entry) {
      event.getEntry(*events_tree, entry);
      // Puts things back to default values (0 for now)
      output.reset(event);

      for (auto& ele : event.Electron)
        output.set_electrons(ele);

      // Writes to tree
      output.fill();

    }
  }

  return 0;

}
