# Twitter Financial News Sentiment Analysis

This project aims to classify financial news tweets into predefined categories using various machine learning and deep learning models. The notebook explores the data, cleans the text, trains models like Logistic Regression and a Convolutional Neural Network (CNN), and provides a simple web interface for real-time predictions using Streamlit.

![Word Cloud of Financial Tweets](https://i.imgur.com/KzQO6mE.png)

---

## **Table of Contents**

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [File Structure](#file-structure)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)

---

## **Project Overview**

The goal is to perform multi-class text classification on a dataset of financial tweets. The project covers the entire machine learning pipeline:
1.  **Data Loading and Preprocessing:** Cleaning text data to remove noise like URLs, punctuation, and stopwords.
2.  **Exploratory Data Analysis (EDA):** Visualizing the distribution of tweet categories and the most frequent words using a word cloud.
3.  **Feature Engineering:** Converting text into numerical vectors using TF-IDF.
4.  **Model Training:** Building and training several models, including Logistic Regression, Linear SVC, and a TensorFlow-based CNN.
5.  **Evaluation:** Assessing model performance using classification reports and confusion matrices.
6.  **Deployment:** Creating a simple web application with Streamlit to predict the sentiment of new tweets.

---

## **Dataset**

The dataset consists of two CSV files:
- `train_data.csv`: Used for training the models.
- `valid_data.csv`: Used for validation and testing.

Each file contains two columns:
- `text`: The raw text of the financial tweet.
- `label`: A numerical category from 0 to 19 representing the tweet's topic.

### **Label Mapping**

The numerical labels correspond to the following categories:

| Label | Category |
| :--- | :--- |
| 0 | Analyst Update |
| 1 | Fed / Central Banks |
| 2 | Company / Product News |
| 3 | Treasuries / Corporate Debt |
| 4 | Dividend |
| 5 | Earnings |
| 6 | Energy / Oil |
| 7 | Financials |
| 8 | Currencies |
| 9 | General News / Opinion |
| 10 | Gold / Metals / Materials |
| 11 | IPO |
| 12 | Legal / Regulation |
| 13 | M&A / Investments |
| 14 | Macro |
| 15 | Markets |
| 16 | Politics |
| 17 | Personnel Change |
| 18 | Stock Commentary |
| 19 | Stock Movement |

---

## **Methodology**

The text data is first cleaned by converting it to lowercase, removing URLs, mentions, hashtags, punctuation, and stopwords. The cleaned text is then vectorized using `TfidfVectorizer`.

Several models were trained and evaluated:
- **Logistic Regression:** A baseline model that achieved ~77% accuracy.
- **Linear SVC:** Another strong baseline model.
- **Convolutional Neural Network (CNN):** A deep learning model using TensorFlow/Keras that also achieved ~77% accuracy, showing strong performance in capturing local patterns in text.

---

## **Results**

Both the Logistic Regression and the CNN models performed well, achieving a validation accuracy of approximately **77%**. The CNN model's performance suggests that deep learning is a viable approach for this task.

![Training and Validation Accuracy of CNN Model](https://i.imgur.com/v8tT1y5.png)

---

## **File Structure**

```
twitter-financial-sentiment/
├── twitter.ipynb             # The original Jupyter Notebook with all experiments.
├── app.py                    # The Streamlit web application script.
├── requirements.txt          # A list of required Python packages.
├── model.joblib              # The saved Logistic Regression model.
├── vectorizer.joblib         # The saved TF-IDF vectorizer.
├── train_data.csv            # Training data.
├── valid_data.csv            # Validation data.
└── README.md                 # This file.
```

---

## **Setup and Installation**

**1. Clone the repository:**
```bash
git clone [https://github.com/your-username/twitter-financial-sentiment.git](https://github.com/your-username/twitter-financial-sentiment.git)
cd twitter-financial-sentiment
```

**2. Create a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

**3. Install the required packages:**
```bash
pip install -r requirements.txt
```
**4. Download NLTK data:**
Run the following commands in a Python interpreter:
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
```

---

## **Usage**

### **1. Saving the Model and Vectorizer**

Before running the Streamlit app, you need to train and save the model. Add the following code to the end of your `twitter.ipynb` notebook and run it to save the Logistic Regression model and the TF-IDF vectorizer.

```python
# Add this to your Jupyter Notebook to save the model
import joblib

# ... (your existing code for training the Logistic Regression model) ...

# Save the vectorizer
joblib.dump(vectorizer, 'vectorizer.joblib')

# Save the trained model
joblib.dump(model, 'model.joblib')
```

### **2. Running the Streamlit App**

Once the `model.joblib` and `vectorizer.joblib` files are saved, you can run the web application.

```bash
streamlit run app.py
```

This will open a new tab in your browser where you can enter a financial tweet and see the model's prediction.
