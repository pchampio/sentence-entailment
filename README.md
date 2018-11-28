## Sentence entailment

### Introduction
Given two sentences of text, s1 and s2, the systems need to compute how similar s1 and s2 are.

### Dataset
The SICK data set consists of 10,000 English sentence pairs, each annotated for relatedness in meaning.  
File Structure: tab-separated text file

Fields:
 - sentence pair ID
 - sentence A
 - sentence B
 - semantic relatedness gold label (on a 1-5 continuous scale)
 - textual entailment gold label (NEUTRAL, ENTAILMENT, or CONTRADICTION)

The SICK data set is released under a Creative Commons Attribution-NonCommercial-ShareAlike 3.0 
Unported License (http://creativecommons.org/licenses/by-nc-sa/3.0/deed.en_US)

### Evaluation

Systems are evaluated on classification accuracy (the percent of labels that are
predicted correctly) for every sentence pairs. We are also interested in the precision/recall scores
for each class as well as a confusion matrix.
