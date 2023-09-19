# Multilingual Speaker Recognition

## Overview
This repository contains the code and documentation for a project on Multilingual Speaker Recognition. The primary goal of this project is to distinguish between ten bilingual speakers speaking two different languages, Hindi and English, using various techniques, including MFCC feature extraction and machine learning models such as KNN, CNN, and LSTM.

## Project Description
In this project, we explore the fascinating field of Multilingual Speaker Recognition. Here's an in-depth look at our approach, methods, and findings:

### Data Collection and Preprocessing
- We collected speech data ourselves, recording sentences in both Hindi and English by ten different bilingual speakers.
- Speech signals were recorded at a 16 KHz sampling rate with a mono channel.
- Preprocessing was performed to remove noise and silence between words in the recorded sentences.

### Feature Extraction with MFCC
- We employed Mel-Frequency Cepstral Coefficients (MFCC) feature extraction to capture pertinent acoustic characteristics from the raw speech signals.
- MFCC is a widely used technique in speech processing to represent the spectral characteristics of audio signals.

### Model Selection
We experimented with various machine learning models to accomplish the speaker recognition task:

1. **K-Nearest Neighbors (KNN)**:
   - KNN is a simple yet effective classification algorithm that relies on the similarity between data points.

2. **Convolutional Neural Network (CNN)**:
   - CNNs have proven to be powerful for various signal processing tasks.
   - We employed CNNs to explore how deep learning techniques could enhance speaker recognition.

3. **Long Short-Term Memory (LSTM)**:
   - LSTMs are a type of recurrent neural network (RNN) known for their ability to capture sequential patterns.
   - We investigated how LSTMs could be applied to speaker recognition.

### Experiments and Results
- We conducted extensive experiments to evaluate the performance of our models.
- We analyzed the impact of noise on audio data and its effect on speaker recognition accuracy.
- We compared the suitability of VCV (Vowel-Consonant-Vowel) and non-VCV sentences for the speaker recognition task.
- Finally, we compared the performance of KNN, CNN, and LSTMs to determine which model performed better and why.

### Achievements
- We achieved a peak accuracy of 78% in distinguishing between the ten bilingual speakers in both Hindi and English.
- This result demonstrates the effectiveness of our approach in Multilingual Speaker Recognition.

## Dependencies
- Python 3.x
- Libraries: NumPy, Pandas, Scikit-Learn, TensorFlow, Keras (for CNN and LSTM), and any other specific dependencies mentioned in the notebooks.

## Contributions
Feel free to contribute to this project by opening issues or creating pull requests. Your feedback and suggestions are highly appreciated.

## Acknowledgments
We would like to express our gratitude to Prof. Prasantha Kumar Ghosh for their guidance and support throughout this project.
