# Third-party notices

The root MIT license covers author-written code and exported DelePref heads,
not independently obtained models, databases or dependencies.

## ESM and ESMC

The release uses [ESM](https://github.com/Biohub/esm) and the exact pinned
[ESMC-600M snapshot](https://huggingface.co/biohub/ESMC-600M). The current
upstream model card declares MIT plus third-party notices. Consult the
snapshot's license and upstream `THIRD_PARTY_NOTICE.md`; no backbone weights
are bundled. The tested pip dependency `esm==3.2.1.post1` contains its older
Cambrian license notices. Installing it does not replace those dependency
terms with this repository's MIT license.

`encoder_compat.py` adapts the character-BPE constructor also present in
the current MIT-licensed Biohub ESM `esm/tokenization/sequence_tokenizer.py`.
It removes incompatible read-only special-token properties without changing
the vocabulary or token IDs. The applicable source notice is reproduced here:

Copyright 2026 Chan Zuckerberg Biohub, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

## Other scientific software

- [MMseqs2](https://github.com/soedinglab/MMseqs2): GPLv3, external executable;
  no binary bundled.
- [SCISOR](https://github.com/baronet2/SCISOR): MIT, official code/weights
  obtained separately. SCISOR-L K1 remains labelled K1.
- [Raygun](https://github.com/rohitsinghlab/raygun): official LICENSE is
  CC BY-NC 4.0; README.v1 includes additional/noncommercial wording. Follow
  the exact revision's terms; code/weights are not relicensed here.
- USalign, ESMFold, Boltz and PyMOL: independently obtained tools under their
  own licenses. Source/predictor identities remain in the scientific archive.

## Data

PDB coordinates are CC0; retain structure-author citations through PDB IDs.
CATH domains, UniProt/UniRef sequences and M-CSA annotations are CC BY 4.0;
source identifiers and provider links are retained in the Data Card/profiles.
Historical experimental-assay data retain their assay-specific references
and licenses. They are not newly measured or relicensed as author-owned data.
Python dependencies retain their own licenses. Cached embeddings and derived
predictions do not grant a new license to an upstream database.
