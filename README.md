# Performance of neuralcoref / data exploration

Tools used:
- GUM Corpus: https://github.com/amir-zeldes/gum
- neuralcoref: https://github.com/huggingface/neuralcoref
- Reference Coreference Scorer: https://github.com/conll/reference-coreference-scorers
- Spacy: https://spacy.io/models/en

This projects investigates how well neuralcoref (a coreference system using neural networks) fed by spacy language models for English (en_core_web_sm and en_core_web_lg) performs on the GUM corpus among different POS categories(Speech pronouns, third person pronouns, demonstratives, definite expressions, proper names and others).The performance is assessed via Reference Coreference Scorer. This project was inspired by Pascal Denis & Jason Baldridge [paper](https://www.aclweb.org/anthology/D08-1069.pdf).

It was found that neuralcoref fed with en_core_web_sm had the best performance on third person pronouns having a F1 score of 96.09 for the bio section of the GUM corpus, compare this to the F1 score of 2.63 of speech pronouns for the whow section.
