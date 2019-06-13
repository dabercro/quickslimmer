#! /usr/bin/env python

import os
import sys
import re
import copy
import subprocess
import logging

from collections import defaultdict

from xml.etree import ElementTree

TYPES = {
    'B': 'Char_t',
    'b': 'UChar_t',
    'S': 'Short_t',
    's': 'UShort_t',
    'I': 'Int_t',
    'i': 'UInt_t',
    'F': 'Float_t',
    'D': 'Double_t',
    'L': 'Long64_t',
    'l': 'ULong64_t',
    'O': 'Bool_t'
}

COMPRESSION_LEVEL = '1'
DEFAULT_TYPE = 'F'
DEFAULT_VAL = '0'
RESET_SIGNATURE = 'reset()'
FILL_SIGNATURE = 'fill()'
FILTER = {'--': [], '++': []}

DEFINITIONS = {}

PREF_HOLDER = '[]'

IN_FUNCTION = None

if os.environ.get('debug'):
    logging.basicConfig(level=logging.DEBUG)

class Prefix:
    prefs = {}
    def __init__(self, pref, parent):
        self.val = '%s%s' % (parent or '', pref)
        self.parent = parent
        self.prefs[self.val] = self

    @staticmethod
    def get(prefix):
        try:
            return Prefix.prefs[prefix]
        except KeyError:
            return Prefix(prefix, None)

    def replace(self, string):
        if not self.val:
            return string
        string = re.sub(r'([^\w]|^)\[\]', r'\1' + self.val, string)
        return re.sub(r'\[\^\]', self.parent or '', string)

    def go_back(self, line):
        if not self.val:
            return line
        fix = lambda pref, l, ins: re.sub(r'(.+?)(?!\.)\b' + re.escape(pref) + r'(\w)',
                                          r'\1' + ins + r'\2', l)

        line = fix(self.val, line, '[]')
        if self.parent:
            line = fix(self.parent, line, '[^]')
        return line


prefixes = []   # Unfortunately, I wrote a lot of code with this global name already, so let's use it
class Prefixes:
    current = None
    def __init__(self, addition, components):
        global prefixes
        self.parent = copy.deepcopy(self.current)
        inputs = [comp.strip() for comp in components.split(',')]
        self.prefs = ['%s%s' % (pref, comp) for pref in prefixes for comp in inputs] or inputs if not addition else prefixes + inputs
        if not addition:
            loop = prefixes or [None]
            for parent in loop:
                for comp in inputs:
                    Prefix(comp, parent)

        prefixes = self.prefs
        Prefixes.current = self

    def get_parent(self):
        global prefixes
        Prefixes.current = self.parent
        prefixes = self.parent.prefs if self.parent else []
        

class Branch:
    branches = {}
    floats = set()
    def __init__(self, pref, name, data_type, default_val, is_saved):
        logging.getLogger('Branch').debug(
            'creating branch: %s %s %s %s %s',
            pref, name, data_type, default_val, is_saved
            )

        self.is_vector = name.endswith('[]')
        self.prefix = pref
        self.name = name.rstrip('[]')
        self.data_type = data_type
        self.default_val = default_val
        self.is_saved = is_saved
        self.branches[self.name] = self

    def name(self):
        return self.name + ('[]'*self.is_vector)

    def declare(self):
        store_type = TYPES.get(self.data_type, self.data_type)
        if self.is_vector:
            store_type = 'std::vector<%s>' % store_type
        return '  %s %s;' % (store_type, self.name)

    def book(self):
        if self.is_vector:
            return 't->Branch("{0}", &{0});'.format(self.name, self.data_type, self.prefix)
        return 't->Branch("{0}", &{0}, "{0}/{1}");'.format(self.name, self.data_type)

    def reset(self):
        if self.is_vector:
            return '{0}.clear();'.format(b.name)
        return '{0} = {1};'.format(b.name, b.default_val)

class Uncertainty:
    uncs = {}
    def __init__(self, name):
        if name in self.uncs:
            raise RuntimeError(name + ' uncertainty already defined once, use update instead')
        self.branches = []
        self.name = name
        Uncertainty.uncs[name] = self

    def update(self, branch):
        if isinstance(branch, list):
            for b in branch:
                self.update(b)
        else:
            self.branches.append(branch.split('_%s' % self.name)[0])
        logging.getLogger('Uncertainty.update').debug(
            'Uncertainty: %s; branches: %s',
            self.name, self.branches
            )


def check_uncertainties(input_line):
    output = [input_line]

    pref = Prefix.get(prefixes[0] if prefixes else '')
    for unc, val in Uncertainty.uncs.items():
        beginning = re.match(r'(^\w*)', input_line.lstrip('# ')).group(1)

        if beginning.endswith('Up') or beginning.endswith('Down') or '%s_%s' % (pref.val, beginning) in val.branches or not beginning:
            continue

        check_line = pref.replace(input_line)
        for branch in val.branches:
            check_line = re.sub(r'\b' + re.escape(branch) + r'\b', '%s_%sUp' % (branch, unc), check_line)

        check_line = pref.go_back(check_line)

        if check_line != input_line:
            if prefixes:
                val.update(['%s_%s_%s' % (p, beginning, unc) for p in prefixes])
            else:
                val.update('%s_%s' % (beginning, unc))
            new_line = check_line.replace(beginning, '%s_%s%s' % (beginning, unc, 'Up'), 1)
            output.append(new_line)
            output.append(new_line.replace('Up', 'Down'))

    return output


def create_branches(var, data_type, val, is_saved):
    branches = [Branch(pref, ('%s_%s' % (pref, var)).rstrip('_'), data_type, val, is_saved) for pref in prefixes] or [Branch('', var, data_type, val, is_saved)]
    # Add uncertainties as we come to them
    for b in branches:
        match = re.match(r'(\w*Up$)', b.name)
        if match:
            for unc in Uncertainty.uncs:
                if match.group(1).endswith('_%sUp' % unc):
                    logging.getLogger('create_branches').debug(
                        'Branch %s matched Uncertainty %s',
                        b.name, unc
                        )
                    Uncertainty.uncs[unc].update(b.name)

    return branches


class Parser:
    def __init__(self, comment='! ', block='*'):
        self.defs = {}
        self.comment = comment
        self.block = block
        self.prev_line = ''           # Holds line continuations
        self.in_comment = False
        self.updown_char = '%'

    def parse(self, raw_line):
        input_line = raw_line.split(self.comment)[0].strip()

        if self.prev_line:
            input_line = self.prev_line + input_line

        if input_line and input_line[-1] == '\\':    # line continuation
            self.prev_line = input_line[:-1]
            return []
        else:
            self.prev_line = ''

        if input_line == self.comment + self.block:
            self.in_comment = True
        elif input_line == self.block + self.comment:
            self.in_comment = False

        # Skip empty lines, comments (!), or block comment contents
        if not input_line or self.in_comment:
            return []

        match = re.match(r'.*(\`(.*)\`).*', input_line)   # Search for shell commands
        if match:
            input_line = input_line.replace(match.group(1), subprocess.check_output(['bash', '-c', match.group(2)]).strip())

        for word, meaning in DEFINITIONS.iteritems():
            input_line = input_line.replace(word, meaning)

        # Expand range operators '..'
        expand = re.search(r'(\d+)\.\.(\d+)', input_line)
        while expand:
            input_line = input_line.replace(expand.group(), ', '.join([str(n) for n in range(int(expand.group(1)), int(expand.group(2)))]))
            expand = re.search(r'(\d+)\.\.(\d+)', input_line)

        start = ''
        pointer_holder = '^&^&^'
        while input_line != start:
            start = input_line
            input_line = input_line.replace('->', pointer_holder)
            for matches in list(re.finditer(r'<([^<>{}]*){(.*?)}([^<>]*)>', input_line)) + list(re.finditer(r'<([^<>{}]*)\|([^\|]*?)\|([^<>\|]*)>', input_line)):
                beg, end = ('<', '>') if '|' in matches.group(1) or '{' in matches.group(3) and '}' in matches.group(3) else ('', '')
                for splitter in [', ', ' + ', ' - ', ' * ', ' / ', ' : ']:
                    if splitter.strip() in matches.group(2):
                        input_line = input_line.replace(
                            matches.group(0),
                            beg + matches.group(1) +
                            ('%s%s%s%s%s' % (matches.group(3), end, splitter, beg, matches.group(1))).join(
                                [suff.strip() for suff in matches.group(2).split(splitter.strip())]) +
                            matches.group(3) + end
                            )
                        break

            input_line = input_line.replace(pointer_holder, '->')


        match = re.match(r'(.*)\|([^\s])?\s+(.*)', input_line)  # Search for substitution
        if match:
            char = match.group(2) or '$'
            subs = match.group(3).split(',')
            lines = [match.group(1).replace('=%s=' % char, str(i)).replace(char * 3, var.strip().upper()).replace(char * 2, var.strip().title()).replace(char, var.strip())
                     for i, var in enumerate(subs)]

            if len(lines) == 2 and char == self.updown_char:
                lines[0] = lines[0].replace('+-', '+')
                lines[1] = lines[1].replace('+-', '-')

            return [line for l in lines for line in self.parse(l)]  # Recursively parse lines in case multiple expansions are present

        match = re.match(r'(\#?\s*)\?\s*([\w\[\]]*)\s*;\s*(\w*)\s*(;\s*([\w\s,]*?)\s*)?\?(((.*(<-|->|=))\s*).*)', input_line) # Parse uncertainties
        if match:
            unc_name = match.group(3)
            if unc_name not in Uncertainty.uncs:
                Uncertainty(unc_name)

            followers = match.group(5) or ''
            branches = [match.group(2)] + [branch.strip() for branch in followers.split(',') if branch]
            new_lines = []
            if len(branches) > 1:
                new_lines.append('~~~ []_%s ~~~' % match.group(2))

            base = match.group(2).rstrip('[]')
            arr_end = '[]' if match.group(2).endswith('[]') else ''

            new_master = '%s_%s%s%s' % (base, unc_name, self.updown_char * 2, arr_end)
            ending = ' |%s up, down' % self.updown_char
            new_lines.append(match.group(1) + new_master + match.group(6).rstrip() + ending)
            for branch in branches[1:]:
                new_lines.append(match.group(1) +
                                 '%s_%s%s%s' % (branch + arr_end, unc_name, self.updown_char * 2, arr_end) +
                                 match.group(7) +
                                 '[]_%s%s * []_%s/[]_%s' % (branch, arr_end,
                                                            new_master,
                                                            match.group(2)) + ending)

            return [line for l in new_lines for line in self.parse(l)]
            
        while re.match(r'.*->.*\w\[\](?!\w).*', input_line):
            input_line = re.sub(r'(->.*\w)\[\](?!\w)', r'\1.back()', input_line)

        # Multi-line for expansion
        if '\\' in input_line:
            return [line for l in input_line.split('\\') for line in self.parse(l)]

        logging.getLogger('Parser.parse').debug('output: %s', input_line)

        return [input_line]


class MyXMLParser:
    def __init__(self, tag, spec_tag, member):
        self.tag = tag
        self.spec_tag = spec_tag
        self.member = member
        self.output = []
        self.spectators = []
    def start(self, tag, attr):
        if tag == self.tag:
            self.output.append(attr[self.member])
        elif tag == self.spec_tag:
            self.spectators.append(attr[self.member])
    def end(self, _):
        pass
    def data(self, _):
        pass
    def close(self):
        return self.output, self.spectators


class Reader:
    readers = []
    holders = set()
    def __init__(self, weights, prefix, output, inputs, specs, subs):
        self.weights = weights
        self.output = ('%s_%s' % (prefix, output)).strip('_')
        sub_pref = lambda x: [(label, Prefix.get(prefix).replace(subs.get(inp, inp)) if prefix else subs.get(label, label))
                              for label, inp in x]
        self.inputs = sub_pref(inputs)
        self.specs = sub_pref(specs)
        self.name = 'reader_%s' % self.output
        self.method = 'method_%s' % self.output
        self.floats = []
        self.readers.append(self)

    def float_inputs(self, mod_fill):
        for index, inp in enumerate(self.inputs):
            address = inp[1]
            if Branch.branches[address].data_type != 'F':
                newaddress = '_tmva_float_%s' % address
                self.inputs[index] = (inp[0], newaddress)
                if address not in Branch.floats:
                    Branch.floats.add(address)
                    mod_fill.insert(0, '%s = %s;' % (newaddress, address))
                    self.floats.append(newaddress)

    def implement_systematics(self):
        output = []
        original_branch = Branch.branches[self.output]

        for sys, val in Uncertainty.uncs.iteritems():
            varying = []
            for inp in self.inputs:
                if inp[1] in val.branches:
                    varying.append(inp[1])

            if not varying:
                continue

            val.update('%s_%s' % (self.output, sys))
            original_branch = Branch.branches[self.output]

            for moved in varying:
                declare = 'decltype({moved}) '.format(moved=moved)
                tmpname = '_syshold_{moved}_{sys}'.format(moved=moved, sys=sys)
                if tmpname in self.holders:
                    declare = ''
                self.holders.add(tmpname)
                output.append(
                    (declare + tmpname + ' = {moved};').format(
                        moved=moved))

            for direction in ['Up', 'Down']:
                for moved in varying:
                    output.append(
                        '{moved} = {moved}_{sys}{direction};'.format(
                            moved=moved, sys=sys, direction=direction))

                # Save to branch
                outputname = '{base}_{sys}{direction}'.format(
                    base=self.output, sys=sys, direction=direction)

                # Call to create branch
                Branch(original_branch.prefix, outputname, original_branch.data_type,
                       original_branch.default_val, original_branch.is_saved)

                # Set systematic branch
                output.append('{output} = {method}->GetMvaValue();'.format(output=outputname, method=self.method))

                # Put everything back
            for moved in varying:
                output.append(
                    '{moved} = _syshold_{moved}_{sys};'.format(
                        moved=moved, sys=sys))

        return output


class Corrector(object):
    correctors = []
    branchcount = defaultdict(lambda: 0)
    def __init__(self, name, expr, infile, hists, cut, histtype):
        self.name = name
        self.expr = expr
        self.cut = cut
        self.corrector = 'corrector_{branch}_{num}'.format(branch=name, num=Corrector.branchcount[branch])
        Corrector.branchcount[branch] += 1
        # Use TH1F or TH2F depending on the number of expressions
        self.init = 'Correction<%s> %s {"%s", "%s"};' % (
            histtype or 'TH%iF' % len(expr.split(',')), self.corrector,
            infile, '", "'.join([h.strip() for h in hists.split(',')]))
        Corrector.correctors.append(self)

    def eval(self, mod_fill):
        if self.cut:
            mod_fill.append('{branch} *= {cut} ? {corrector}.GetCorrection({args}) : 1.0;'.format(
                    branch=self.name, cut=self.cut, corrector=self.corrector, args=self.expr))
        else:
            mod_fill.append('{branch} *= {corrector}.GetCorrection({args});'.format(
                    branch=self.name, corrector=self.corrector, args=self.expr))


class TFReader(object):
    readers = []
    def __init__(self, targets, weights, inputs, input_node, output_node, is_saved, mod_fill):
        self.targets = targets
        self.weights = weights

        pref = lambda l: Branch.branches[l.strip()].prefix
        self.inputs = [(line.replace(pref(line), PREF_HOLDER) if pref(line) in prefixes else line).strip() \
                           for line in open(inputs, 'r')]

        self.input_node = input_node
        self.output_nodes = output_node
        self.is_saved = is_saved

        for branch in targets:
            create_branches(branch, 'F', 0, is_saved)

        # Grab a reference to the current prefixes
        self.prefixes = prefixes or ['']

        self.session = '%s_session' % os.path.basename(weights).split('.')[0]
        TFReader.readers.append(self)
        self.eval(mod_fill)

    def create(self):
        # Just load the graph when the object is created. Simple.
        return """  tensorflow::GraphDef {session}_graph_def;
  check_tf(tensorflow::Read{encoding}Proto(tensorflow::Env::Default(), "{weights}", &{session}_graph_def));
  check_tf({session}->Create({session}_graph_def));
""".format(encoding='Text' if self.weights.endswith('.pbtxt') else 'Binary',
           session=self.session, weights=self.weights)

    def eval(self, mod_fill):
        # Initialize input tensors
        mod_fill.append('std::vector<std::pair<std::string, tensorflow::Tensor>> {session}_inputs {init};'.format(
                session=self.session, b='{', e='}',
                init='{{"%s", {tensorflow::DT_FLOAT, {1, %i}}}}' % (self.input_node, len(self.inputs))))

        # Set up the different prefixes
        for pref in self.prefixes:
            mod_fill.extend(['{session}_inputs[0].second.matrix<float>()(0, {i}) = static_cast<float>({val});'.format(
                        session=self.session, pref=pref, i=i, val=in_branch.replace(PREF_HOLDER, pref)) for i, in_branch in enumerate(self.inputs)])

            mod_fill.extend(['std::vector<tensorflow::Tensor> {session}_outputs_{pref};'.format(session=self.session, pref=pref),
                             'check_tf({session}->Run({session}_inputs, {out_layer}, {target}, &{session}_outputs_{pref}));'.format(
                        session=self.session, out_layer='{"%s"}' % '", "'.join(self.output_nodes), target='{}', pref=pref)] + 
                            ['{pref}_{target} = {session}_outputs_{pref}[{node_index}].matrix<float>()(0, {target_index});'.format(
                        session=self.session, pref=pref, target=target,
                        node_index=min(len(self.output_nodes) - 1, target_index),
                        target_index=target_index - min(len(self.output_nodes) - 1, target_index)
                        ).lstrip('_') for target_index, target in enumerate(self.targets)])


class Function:
    def __init__(self, signature, prefixes):
        logging.getLogger('Function').debug('%s %s', signature, prefixes)
        self.enum_name = re.match(r'(.*)\(.*\)', signature).group(1)
        self.prefixes = prefixes
        if prefixes:
            signature = signature.replace('(', '(const %s base, ' % self.enum_name)
        else:
            signature = signature.replace('(', '(const ')
        self.signature = 'set_' + signature.replace(', ', ', const ').replace(', const )', ')')
        template_args = set()
        for match in re.finditer(r'<(.)>', self.signature):
            template_args.add(match.group(1))
            self.signature = self.signature.replace(match.group(0), match.group(1))

        self.template = 'template <typename %s> ' % ', typename '.join(template_args) if template_args else ''
        self.variables = []
        self.tmp_vars = []

    def add_var(self, variable, value):
        new_var = (variable, value) if '~' in value else (('_%s' % variable).rstrip('_'), value)
        self.variables.append(new_var)

    def add_tmp_var(self, line):
        self.tmp_vars.append(line)

    def declare(self, functions):
        output = '{0}%svoid %s;' % (self.template, self.signature)

        for func in functions:
            # Just use the first occurance of an identical enum class
            if func.enum_name == self.enum_name:
                break
            if func.prefixes == self.prefixes:
                output = output.replace('const %s' % self.enum_name, 'const %s' % func.enum_name)
                self.signature = self.signature.replace('const %s' % self.enum_name, 'const %s' % func.enum_name)
                self.enum_name = func.enum_name

                return output.format('')

        if self.prefixes:
            return output.format("""
  enum class %s {
    %s
  };
  """ % (self.enum_name, ',\n    '.join(self.prefixes)))
        return output.format('')

    def var_line(self, var, val, pref):
        incr = ['++', '--']
        indent = ' ' * (4 if self.prefixes else 2)
        if val in incr:
            return '%s%s%s%s;' % (indent,val, pref, var)
        elif '~' in val:
            return '%sif (!(%s))\n  %s%s;' % (indent, Prefix.get(pref).replace(var),
                                              indent, 'return' if len(val) == 2 else 'throw')
        elif var.endswith('[]'):
            return '{indent}{pref}{var}.push_back({val});'.format(
                indent=indent, pref=pref, var=var.rstrip('[]'), val=Prefix.get(pref).replace(val))

        op = '='
        if val[0] in '+-*/':
            op = val[0] + '='
            val = val[1:]
        return'%s%s%s %s %s;' % (indent, pref, var, op, Prefix.get(pref).replace(val))

    def implement(self, classname):
        if self.prefixes:
            middle = """  switch(base) {
%s
    break;
  default:
    throw;
  }""" % ('\n    break;\n'.join(
                    ['\n'.join(['  case %s::%s::%s:' % (classname, self.enum_name, pref)] +
                               [self.var_line(var, val, pref) for var, val in self.variables])
                     for pref in self.prefixes]))
        else:
            middle = '\n'.join([self.var_line(var.lstrip('_'), val, '') for var, val in self.variables])

        return """%svoid %s::%s {
%s%s
}
""" % (self.template, classname,
       self.signature,
       ''.join(['  auto %s;\n' % tmp for tmp in self.tmp_vars]),
       middle)


if __name__ == '__main__':
    classname = os.path.basename(sys.argv[1]).split('.')[0]

    includes = ['<string>', '<vector>', '"TObject.h"', '"TFile.h"', '"TTree.h"']
    functions = []
    mod_fill = []
    correction = None

    parser = Parser()

    with open(sys.argv[1], 'r') as config_file:
        for raw_line in config_file:
            for line in parser.parse(raw_line):
                line = line.strip()

                # Check for includes
                match = re.match(r'^([<"].*[">])$', line)
                if match:
                    includes.append(match.group(1))
                    continue

                # Compression level
                match = re.match(r'COMPRESSION\s+(\d)', line)
                if match:
                    COMPRESSION_LEVEL = match.group(1)
                    continue

                # Define a "macro" for the config file
                match = re.match(r'DEFINE\s+(\w+)\s+(.*)', line)
                if match:
                    DEFINITIONS[match.group(1)] = match.group(2)
                    continue

                # Check for default values or reset function signature
                match = re.match(r'{(/(\w))?(-?\d+)?(reset\(.*\))?(fill\(.*\))?}', line)
                if match:
                    if match.group(2):
                        DEFAULT_TYPE = match.group(2)
                    elif match.group(3):
                        DEFAULT_VAL = match.group(3)
                    elif match.group(4):
                        RESET_SIGNATURE = match.group(4).replace('(', '(const ').replace(', ', ', const ')
                    elif match.group(5):
                        FILL_SIGNATURE = match.group(5).replace('(', '(const ').replace(', ', ', const ')
                    continue

                # Get "class" enums and/or functions for setting
                match = re.match(r'(\+)?([\w,\s]*){', line)
                if match:
                    # Get the different classes that are being added
                    Prefixes(bool(match.group(1)), match.group(2))

                    continue

                # End previous enum scope
                match = re.match(r'}', line)
                if match:
                    if IN_FUNCTION:
                        functions.append(IN_FUNCTION)
                    IN_FUNCTION = None
                    Prefixes.current.get_parent()

                    continue

                # Get functions for setting values
                match = re.match('^(\w+\(.*\))$', line)
                if match:
                    logging.getLogger('match-function').debug('%s', match.groups())
                    # Pass off previous function as quickly as possible to prevent prefix changing
                    if IN_FUNCTION:
                        functions.append(copy.deepcopy(IN_FUNCTION))
                        IN_FUNCTION = None

                    IN_FUNCTION = Function(match.group(1), prefixes)
                    continue

                # TensorFlow applicator
                match = re.match(r'^(\#\s*)?\[{2}([^;]*);\s*([^;]*);\s*([^;]*);\s*([^;]*);\s*([^;]*)\]{2}', line)
                if match:
                    includes.extend(['"tensorflow/core/public/session.h"',
                                     '<memory>'])

                    if not match.group(1):
                        TFReader(targets=[target.strip() for target in match.group(2).split(',')],
                                 weights=match.group(3), inputs=match.group(4),
                                 input_node=match.group(5),
                                 output_node=[out.strip() for out in match.group(6).split(',')],
                                 is_saved=(not match.group(1)), mod_fill=mod_fill)
                    continue

                # Get TMVA information to evaluate
                match = re.match(r'^(\#\s*)?\[([^;]*);\s*([^;]*)(;\s*(.*))?\](\s?=\s?(.*))?', line)
                if match:
                    includes.extend(['"TMVA/Reader.h"', '"TMVA/IMethod.h"'])

                    is_saved = not bool(match.group(1))
                    var = match.group(2)
                    weights = match.group(3)
                    trained_with = match.group(5)
                    subs = {}
                    if trained_with and os.path.exists(trained_with):  # In this case, trained_with actually points to a file that lists substitutions
                        with open(trained_with, 'r') as sub_file:
                            for line in sub_file:
                                info = line.split()
                                subs[info[0]] = info[1]

                    default = match.group(7) or DEFAULT_VAL
                    branches = create_branches(var, 'F', default, is_saved)
                    if is_saved:
                        xml_vars, xml_specs = ElementTree.parse(weights, ElementTree.XMLParser(target=MyXMLParser('Variable', 'Spectator', 'Expression'))).getroot()
                        rep_pref = lambda x: [(v, v.replace(trained_with or (Branch.branches[v].prefix if Branch.branches[v].prefix in prefixes
                                                                             else PREF_HOLDER), PREF_HOLDER))
                                              for v in x]
                        inputs = rep_pref(xml_vars)
                        specs = rep_pref(xml_specs)
                        for reader in [Reader(weights, b.prefix, var, inputs, specs, subs) for b in branches]:
                            mod_fill.extend(reader.implement_systematics())
                            mod_fill.append('%s = %s->GetMvaValue();' % (reader.output, reader.method))

                    continue

                # Event filter
                match = re.match(r'(\+\+|--)(.*)(\+\+|--)', line)
                if match:
                    FILTER[match.group(1)] = ['if (!(%s))' % match.group(2).strip(),       # Filter at front (--) or back (++) of fill
                                              '  return;']
                    continue

                # Master corrections branch
                match = re.match(r'~\{((\w*).*)\}~', line)
                if match:
                    includes.append('"SkimmingTools/interface/Correction.h"')
                    correction = match.group(2)
                    # Continue to let the branch mechanism pick this up
                    line = match.group(1)

                # Corrector applicator
                match = re.match(r'(#|\?)?~([^;]*);([^;]*);([^;]*);([^;]*)(;(.*?)(TH\d.)?)?~', line)
                if match:
                    includes.append('"SkimmingTools/interface/Correction.h"')
                    branch = match.group(2).strip()
                    Corrector(branch,                 # Branch name
                              match.group(3).strip(), # Expression to evaluate histogram
                              match.group(4).strip(), # File name
                              match.group(5).strip(), # Histogram name(s)
                              (match.group(7) or '').strip(), # Cut to pass
                              match.group(8)          # Type of histogram to expect
                              ).eval(mod_fill)

                    if (branch not in Branch.branches):
                        create_branches(branch, 'F', 1.0, not (match.group(1) == '#'))

                    # If master branch defined, merge
                    if correction and match.group(1) != '?':
                        mod_fill.append('{corr} *= {branch};'.format(corr=correction, branch=branch))
                    continue

                # A cut inside a function
                match = re.match(r'(~{2,3})\s*(.*?)\s*(~{2,})', line)
                if match:
                    # Only valid if in a function
                    IN_FUNCTION.add_var(match.group(2), match.group(1))
                    continue

                # Add variable for beginning of function
                match = re.match(r'^\((.*)\)$', line)
                if match:
                    if IN_FUNCTION is not None:
                        IN_FUNCTION.add_tmp_var(match.group(1))
                    continue

                # Probably a branch line
                for branch_line in check_uncertainties(line):
                    # Add branch names individually
                    match = re.match(r'(\#\s*)?([\w\[\]]*)(/([\w\:\<\>]*))?(\s=\s(.*?))?(\s->\s(.*?))?(\s<-\s(.*?))?$', branch_line)
                    if match:
                        var = match.group(2)
                        data_type = match.group(4) or DEFAULT_TYPE
                        val = (match.group(6) or DEFAULT_VAL).strip()
                        is_saved = not bool(match.group(1))

                        logging.getLogger('matched-branch').debug(
                            'Branch line: %s %s %s %s',
                            var, data_type, val, is_saved
                            )

                        in_func = IN_FUNCTION is not None and match.group(7)

                        branches = create_branches(var, data_type, val, is_saved)

                        if in_func:
                            IN_FUNCTION.add_var(var, match.group(8))

                        if match.group(9):
                            for b in branches:
                                mod_fill.append('%s = %s;' % (b.name, Prefix.get(b.prefix).replace(match.group(10))))

    ## Finished parsing the configuration file
    if IN_FUNCTION:
        functions.append(IN_FUNCTION)
        IN_FUNCTION = None

    # Check that all of the TMVA inputs are floats
    for reader in Reader.readers:
        reader.float_inputs(mod_fill)

    # Let's put the branches in some sort of order
    all_branches = [Branch.branches[key] for key in sorted(Branch.branches)]

    with open('%s.h' % classname, 'w') as output:
        write = lambda x: output.write('%s\n' % x)

        # Write the beginning of the file
        write("""#ifndef CROMBIE_{0}_H
#define CROMBIE_{0}_H
""".format(classname.upper()))
        done = set()
        for inc in includes:
            if inc in done:
                continue
            write('#include %s' % inc)
            done.add(inc)

        # If using Tensorflow, write this error handling function
        if TFReader.readers:
            write("""
void check_tf (const tensorflow::Status& status) {
  if (not status.ok()) {
    std::cerr << status.error_message() << std::endl;
    throw;
  }
}""")

        # Start the class definition
        write('\nclass %s {' % classname)
        write("""
 public:
  {0}(const char* outfile_name, const char* name = "events");
  ~{0}() {1}
""".format(classname, '{ write(t); f->Close(); }'))

        # Write the elements for the memory of each branch
        for branch in all_branches:
            write(branch.declare())

        # Some functions, and define the private members
        write("""
  void %s;
  void %s;
  void write(TObject* obj) { f->WriteTObject(obj, obj->GetName()); }
  %s

 private:
  TFile* f;
  TTree* t;
%s%s%s%s
};
""" % (RESET_SIGNATURE, FILL_SIGNATURE, '\n  '.join([f.declare(functions) for f in functions]),
       ''.join(['\n  %s' % corr.init for corr in Corrector.correctors]),
       ''.join(['\n  Float_t %s;' % address for reader in Reader.readers for address in reader.floats]),
       ''.join(['\n  TMVA::Reader %s {"Silent"};\n  TMVA::IMethod* %s {};' % (reader.name, reader.method) for reader in Reader.readers]),
       ''.join(['\n  std::unique_ptr<tensorflow::Session> %s {tensorflow::NewSession({})};' % reader.session for reader in TFReader.readers]),
       ))

        # Initialize the class
        write("""%s::%s(const char* outfile_name, const char* name)
: f {new TFile(outfile_name, "CREATE", "", %s)},
  t {new TTree(name, name)}
{
  %s
%s%s%s%s}""" % (classname, classname, COMPRESSION_LEVEL,
              '\n  '.join([b.book() for b in all_branches if b.is_saved]),
              ''.join(['  %s.AddVariable("%s", &%s);\n' % (reader.name, label, var) for reader in Reader.readers for label, var in reader.inputs]),
              ''.join(['  %s.AddSpectator("%s", &%s);\n' % (reader.name, label, var) for reader in Reader.readers for label, var in reader.specs]),
              ''.join(['  %s = %s.BookMVA("%s", "%s");\n' % (reader.method, reader.name, reader.output, reader.weights) for reader in Reader.readers]),
              ''.join([reader.create() for reader in TFReader.readers])))

        # reset function
        write("""
void %s::%s {
  %s
}""" % (classname, RESET_SIGNATURE, '\n  '.join([b.reset() for b in all_branches])))

        # fill function
        mod_fill = FILTER['--'] + mod_fill + FILTER['++']
        write("""
void %s::%s {
  %s
}
""" % (classname, FILL_SIGNATURE, '\n  '.join(mod_fill + ['t->Fill();'])))

        for func in functions:
            write(func.implement(classname))

        # Template enum conversion
        def write_convert(first, second):
            write(("""{0}::{1} to_{1}({0}::{2} e_cls) {begin}
  switch (e_cls) {begin}
  case %s
  default:
    throw;
  {end}
{end}
""" % '\n  case '.join(['{0}::{2}::%s:\n    return {0}::{1}::%s;' %
                       (pref, pref) for pref in second.prefixes if pref in first.prefixes])).format(
                           classname, first.enum_name, second.enum_name, first.prefixes[0], begin='{', end='}'))
            

        prev_func = None
        for func in functions:
            if prev_func and func.prefixes != prev_func.prefixes and prev_func.prefixes:
                if False not in [prev in func.prefixes for prev in prev_func.prefixes]:
                    write_convert(prev_func, func)
                    write_convert(func, prev_func)

            prev_func = func

        write('#endif')

    print 'Created %s class with %i branches' % (classname, len(all_branches))
