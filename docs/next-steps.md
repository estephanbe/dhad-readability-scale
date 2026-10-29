# Next steps, and transfer to Arabic


- **Arabic:**
  - swap the backbone for an Arabic encoder (CAMeLBERT-MSA, AraBERTv2) or XLM-R;
  - measure lexical load from root/pattern rarity and MSA vs dialect;
  - measure syntax from clitic and iḍāfa depth with CAMeL Tools;
  - the held-out-vocabulary CV carries over directly as a held-out-root CV.
- **With more time:**
  - real human-labelled data, with ordinal difficulty levels rather than binary flags;
  - word-frequency and concreteness features as a strong interpretable baseline;
  - extend the behavioural tests (`scripts/behavioural_tests.py`) into pass/fail unit tests, and
    retrain without the template cue words to remove the template → lexical shortcut;
  - an encoder seed ensemble.
