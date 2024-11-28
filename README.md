# Personalized Speech Enhancement without User Enrollment
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

## Simulated Test Sets
This test set aims to simulate audio recorded from a personal device. Created from the test-clean part of librispeech dataset and noise datasets, this test set aims to simulate audio/video stocks from a personal device. 
As this set is a simulated data set, a reference audio will be provided to measure the quality of enhanced speech.
Test-clean set of librispeech has 2620 audios from 40 speakers, with a total duration of 324 minutes. These 40 speakers from the test set do not overlap with the training and dev set.

## Real-world Scenarios Test Sets
Real-world Scenarios Test Sets are collected to test the speech enhancement performance in real world.  
This test sets have two scenarios, the first is street food review scenario, and the second is commencement speech scenario. Each scenario has 10 audios collected from Youtube.

### Youtube Street Food Review
Street food review videos is one of the hot scenarios for Vloggers. 
To simulate this scenario, ten videos with the keywords "street food vlog English" are collected from YouTube. The search results are sorted by view count on Aug 12, 2024. 
Videos shorter than 4 minutes, having background music, or containing more than one target speaker get removed. 
Then the top 10 videos are collected, the total duration of this dataset is 140 minutes. 

### Youtube Commencement Speeches
Another real-world scenario is recorded public speech. These speeches may not be clear due to cheers and applause from the audience. 
To simulate this scenario, 10 audios of the most notable commencement speeches of years 2023 & 2024 are collected. 
Five commencement speeches are selected from each year, and these speeches only have one major speaker.  The total duration of this test set is 201 minutes. 
