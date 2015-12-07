environment setting:
    Python 2.7

file usage:
    filter_trimming.py:
        input file need trimming (file name is writtten in
        filter_trimming.py), output file that is trimmed

    output48_39.py:
        mapping 48 phonemes to 39 phonemes

    hmm.py:
        Load transition probability calculating by train data, then generate predict result
command:
    python hmm.py: generate the predict result
    python filter_trimming.py: trimming the file