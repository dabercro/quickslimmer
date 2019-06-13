#! /usr/bin/env perl

use strict;
use warnings;

# Check arguments and print help function if needed
sub print_use {
    print "Usage: $0 PANDADEF SOURCE SOURCE ... OUTPUTHEAD\n\n";
    print "PANDADEF    Must be a PandaTree def file\n";
    print "SOURCE      Source file(s) that you would like to analyze\n";
    print "OUTPUTHEAD  Place where you would like to put the output header\n";
    return "\nYou did something wrong with arguments\n";
}
# Check presence of arguments
my $out_file = pop @ARGV or die print_use;
if (not @ARGV) {
    die print_use;
}

# Check that each source file exists
foreach my $fname (@ARGV) {
    if (not -e $fname) {
        print "File $fname does not exist!\n\n";
        die print_use;
    }
}

# Check first line of header
if (-e $out_file) {
    open (my $handle, '<', $out_file);
    chomp (my $first = <$handle>);
    close $handle;
    if ($first ne '#ifndef CROMBIE_FEEDPANDA_H') {
        print "First line of $out_file looks suspicious! I don't want to overwrite:\n";
        print "$first\n";
        die print_use;
    }
}

# Location of Panda definitions
my $def_file = shift @ARGV;
my @source;
my @branches = ('triggers');

foreach my $infile (@ARGV) {
    open (my $handle, '<', $infile);
    push @source, <$handle>;
    close $handle;
}

# Filter to get the members called
chomp(@source = grep { /\.|(->)/ } @source);

for (@source) {
    # Don't match with function members of event
    if (/\be(vent)?\.(\w+)(?!\w*\()/) {
        push @branches, $2;
    }
}

sub uniq_sort {
    my %seen;
    return sort(grep {! $seen{$_}++ } @_);
}

# Get unique branches from first pass
@branches = uniq_sort @branches;

# Now check def file for references to load
open (my $handle, '<', $def_file);
chomp (my @pandadef = grep { /->/ } <$handle>);
close $handle;
my %poss_refs;
my @last_branches;
while (@last_branches != @branches) {
    @last_branches = uniq_sort @branches;
    # Fill more branches
    foreach my $branch (@branches) {
        foreach my $line (@pandadef) {
            if ($line =~ /$branch\.(\w+)->(\w+)/) {
                push @{$poss_refs{$1}}, $2;
            }
        }
    }
    foreach my $key (keys %poss_refs) {
        @{$poss_refs{$key}} = uniq_sort @{$poss_refs{$key}};
        my @new_branches = grep { /(\.|->)$key/ } @source;
        if (@new_branches) {
            push @branches, @{$poss_refs{$key}};
        }
    }
    @branches = uniq_sort @branches;
}

# Now we have our branches. Time to write a header file!
open (my $out, '>', $out_file);

print $out <<HEAD;
#ifndef CROMBIE_FEEDPANDA_H
#define CROMBIE_FEEDPANDA_H 1

#include "TTree.h"

#include "PandaTree/Objects/interface/Event.h"

void feedpanda(panda::Event& event, TTree* input) {
  event.setStatus(*input, {"!*"});
  event.setAddress(*input,
HEAD

print $out '    {"' . join("\",\n     \"", @branches) . '"';

print $out <<HEAD;
});
}

#endif
HEAD

close $out;
