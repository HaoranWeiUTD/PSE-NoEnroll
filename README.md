# Personalized-Speech-Enhancement-without-User-Enrollment
This repository is for Personalized Speech Enhancement without User Enrollment for Real-world Scenarios

# Project Highlight
Many speech enhancement (SE) approaches have been proposed to deal with cocktail party problem. 
By utilizing speaker's enrollment speech,  personalized speech enhancement(PSE) approaches further improve the performance of SE approaches. 
But PSE approaches require users to record additional clean audio for registration, which can be an redundant works for many real-world scenarios. 
For personal device scenario and Vloggers scenarios, this paper propose a novel speech enhancement approach integrating the PSE method without requiring users to provide pre-registered speech. 
The proposed approach automatically selects high-quality speech clips to help denoising low-quality speech clips, with speech quality assessment module and noise adaptation module. 
This paper also collected simulated test sets and real-world scenarios test sets to prove the performance. 
Experiment result shows the proposed approach outperform the corresponding SE approach for both objective and subjective evaluation metrics.

# Testing Sets
The testing set for this paper can be divided into 2 sets. 
The first test set is a simulated sets created from the test-clean part of librispeech dataset and noise datasets, this test set aims to simulate audio/video stocks from a personal device, this set has clean audio as a reference. 
The second test set is real-word scenario sets which is collected from Youtube, this set contains real-word speech and noise, but does not have clean speech as a reference.
