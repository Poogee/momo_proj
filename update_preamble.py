import re

with open('project_proposal.tex', 'r') as f:
    proposal = f.read()
    
with open('report_v2.tex', 'r') as f:
    report = f.read()

# Extract from \documentclass to \maketitle from proposal
prop_header_match = re.search(r'(\\documentclass.*?\\maketitle)', proposal, re.DOTALL)
prop_header = prop_header_match.group(1)

# Make sure we add extra packages needed by report (booktabs, multirow, etc)
extra_packages = r"""
\usepackage{booktabs}
\usepackage{array}
\usepackage{tabularx}
\usepackage{longtable}
\usepackage{multirow}
\usepackage{hyperref}
\usepackage{microtype}
\usepackage{caption}
\definecolor{darkred}{rgb}{0.7,0.12,0.12}
\definecolor{darkgreen}{rgb}{0.08,0.47,0.16}
"""

prop_header = prop_header.replace('\\begin{document}', extra_packages + '\n\\begin{document}')

# In report, find the part after the custom title block, keeping the abstract
# The abstract in report is under \paragraph{Аннотация.}
report_abstract_match = re.search(r'\\paragraph{Аннотация\.}(.*?)(?=\\section)', report, re.DOTALL)
report_abstract = report_abstract_match.group(1).strip()

abstract_block = f"\\begin{{abstract}}\n{report_abstract}\n\\end{{abstract}}\n\n"

# In report, find everything after \section{Введение}
report_rest_match = re.search(r'(\\section{Введение}.*)', report, re.DOTALL)
report_rest = report_rest_match.group(1)

# Now, we also have the TL;DR figure before the abstract, maybe put it right after \maketitle or \begin{abstract}
tldr_match = re.search(r'(\\begin{figure}\[h\].*?\\end{figure})', report, re.DOTALL)
tldr_fig = tldr_match.group(1) + '\n\n' if tldr_match else ''

# Since IEEEtran is two-column, longtable might break or need adjustments (table*) but we can fix that if pdflatex fails.
# Actually, the user asked to change the format, so we replace it.

final_tex = prop_header + "\n\n" + abstract_block + tldr_fig + report_rest

with open('report_v2_formatted.tex', 'w') as f:
    f.write(final_tex)
