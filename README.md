# Stress Detection System

**By Leon Motaung**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange) ![NLP](https://img.shields.io/badge/NLP-NLTK-green)

## 📖 Overview

This project is a Machine Learning system designed to detect signs of psychological stress in social media posts. By analyzing text data from platforms like Reddit, the system classifies content into "Stress" or "No Stress" categories. This tool aims to assist in the early detection of mental health issues by leveraging Natural Language Processing (NLP) and binary classification algorithms.

## 🚀 Features

* **Text Preprocessing:** Robust cleaning pipeline that removes stopwords, URLs, HTML tags, and punctuation.
* **Stemming:** Utilizes the Snowball Stemmer to normalize words to their root forms.
* **Visualization:** Generates Word Clouds to visualize the most frequent terms associated with stress.
* **Binary Classification:** Uses the Bernoulli Naive Bayes algorithm for efficient text classification.
* **Real-time Prediction:** Allows users to input raw text and receive an immediate stress assessment.

## 🛠️ Tech Stack

* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Natural Language Processing:** NLTK (Natural Language Toolkit)
* **Machine Learning:** Scikit-Learn
* **Visualization:** Matplotlib, WordCloud

## 📂 Dataset

The model is trained on a dataset containing posts from mental health-related subreddits.
* **Source:** Kaggle
* **Labels:** `0` (No Stress) / `1` (Stress)
* **Columns Used:** `text`, `label`

## ⚙️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/leonmotaung/stress-detection.git](https://github.com/leonmotaung/stress-detection.git)
    cd stress-detection
    ```

2.  **Install dependencies:**
    You will need the following Python libraries. You can install them via pip:
    ```bash
    pip install pandas numpy nltk scikit-learn matplotlib wordcloud
    ```

## 🧠 Methodology

### 1. Data Cleaning
Raw text data from social media is noisy. The `clean()` function performs the following operations:
* Converts text to lowercase.
* Removes URLs, HTML tags, and brackets.
* Removes punctuation and special characters.
* Removes stopwords (common words like "the", "is", "at").
* Applies Stemming (reducing words like "stressed" to "stress").

### 2. Feature Extraction
We use `CountVectorizer` to convert the cleaned text into a matrix of token counts, enabling the machine learning model to process the text mathematically.

### 3. Model Training
The project utilizes the **Bernoulli Naive Bayes** classifier (`BernoulliNB`). This algorithm is particularly well-suited for binary classification tasks where features are binary occurrences (word presence/absence), making it highly effective for short text classification.

## 💻 Usage

To run the project locally, execute the main script. The system will train the model and then prompt you for input.

```python
# Run the script
python stress_detector.py# Stress Detection System

**By Leon Motaung**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange) ![NLP](https://img.shields.io/badge/NLP-NLTK-green)

## 📖 Overview

This project is a Machine Learning system designed to detect signs of psychological stress in social media posts. By analyzing text data from platforms like Reddit, the system classifies content into "Stress" or "No Stress" categories. This tool aims to assist in the early detection of mental health issues by leveraging Natural Language Processing (NLP) and binary classification algorithms.

## 🚀 Features

* **Text Preprocessing:** Robust cleaning pipeline that removes stopwords, URLs, HTML tags, and punctuation.
* **Stemming:** Utilizes the Snowball Stemmer to normalize words to their root forms.
* **Visualization:** Generates Word Clouds to visualize the most frequent terms associated with stress.
* **Binary Classification:** Uses the Bernoulli Naive Bayes algorithm for efficient text classification.
* **Real-time Prediction:** Allows users to input raw text and receive an immediate stress assessment.

## 🛠️ Tech Stack

* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Natural Language Processing:** NLTK (Natural Language Toolkit)
* **Machine Learning:** Scikit-Learn
* **Visualization:** Matplotlib, WordCloud

## 📂 Dataset

The model is trained on a dataset containing posts from mental health-related subreddits.
* **Source:** Kaggle
* **Labels:** `0` (No Stress) / `1` (Stress)
* **Columns Used:** `text`, `label`

## ⚙️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/leonmotaung/stress-detection.git](https://github.com/leonmotaung/stress-detection.git)
    cd stress-detection
    ```

2.  **Install dependencies:**
    You will need the following Python libraries. You can install them via pip:
    ```bash
    pip install pandas numpy nltk scikit-learn matplotlib wordcloud
    ```

## 🧠 Methodology

### 1. Data Cleaning
Raw text data from social media is noisy. The `clean()` function performs the following operations:
* Converts text to lowercase.
* Removes URLs, HTML tags, and brackets.
* Removes punctuation and special characters.
* Removes stopwords (common words like "the", "is", "at").
* Applies Stemming (reducing words like "stressed" to "stress").

### 2. Feature Extraction
We use `CountVectorizer` to convert the cleaned text into a matrix of token counts, enabling the machine learning model to process the text mathematically.

### 3. Model Training
The project utilizes the **Bernoulli Naive Bayes** classifier (`BernoulliNB`). This algorithm is particularly well-suited for binary classification tasks where features are binary occurrences (word presence/absence), making it highly effective for short text classification.

## 💻 Usage

To run the project locally, execute the main script. The system will train the model and then prompt you for input.

```python
# Run the script
python stress_detector.py
