# **Sentiment Analysis on Movie Reviews with PySpark** 🎞️

## **📖 1. Project Description**

This project aims to develop and evaluate Machine Learning models for sentiment classification (positive or negative) based on the text of movie reviews. The dataset used is the "IMDb reviews in Portuguese do Brasil," which contains reviews in Portuguese and their English translations. Here was used the reviews in English, however the idea is to try the same approach using it in portuguese later.

The analysis and modeling were performed using the Apache Spark ecosystem, specifically the PySpark library, which is ideal for large-scale data processing. Three different classification algorithms were explored, trained, and optimized: **Decision Tree**, **Random Forest**, and **XGBoost**.

---

## **📂 2. Project Structure**

The project workflow, as executed in the notebook, follows these steps:

### **⚙️ 2.1. Environment Setup**
- Installation of the necessary libraries for distributed processing and modeling, including:
  - `pyspark`: To use Apache Spark with Python.
  - `xgboost`: For the Gradient Boosting model.
  - Other libraries like `wordcloud`, `matplotlib`, and `nltk`.

### **📊 2.2. Data Loading and Exploratory Data Analysis (EDA)**
- **Loading:** Data was loaded from the `imdb-reviews-pt-br.csv` file into a Spark DataFrame.
- **Initial Exploration:**
  - Checking the total number of records (49,459).
  - Analyzing the distribution of sentiment classes (`pos` vs. `neg`), which was found to be well-balanced.
- **Visualization**: A **Word Cloud** was generated to visually identify the most frequent terms in the review corpus.

### **🛠️ 2.3. Data Pre-processing and Feature Engineering**
To prepare the text data for the Machine Learning models, a pipeline was built with the following stages:
1.  **Text Cleaning:** Removal of punctuation and special characters using regular expressions.
2.  **Label Creation:** Conversion of the categorical sentiment column ('pos', 'neg') into a numeric label column using `StringIndexer` ('neg' -> 0.0, 'pos' -> 1.0).
3.  **Tokenization:** Splitting the cleaned text into individual words (tokens).
4.  **Stop Words Removal:** Exclusion of common English words that do not add value to sentiment analysis.
5.  **Vectorization (TF-IDF):**
    - **HashingTF / CountVectorizer:** Conversion of tokens into feature vectors by calculating term frequency.
    - **IDF (Inverse Document Frequency):** Application of the IDF weight to the vectors to give more importance to words that are rare and less importance to those that are very frequent.

### **🤖 2.4. Modeling, Optimization, and Evaluation**
- The dataset was split into **training (70%)** and **testing (30%)** sets.
- Three base models were trained and evaluated: Decision Tree, Random Forest, and XGBoost.
- Hyperparameter tuning was performed on all three models using `ParamGridBuilder` and `CrossValidator` with 3-folds to find the optimal settings.
- The models' performance was evaluated using **Accuracy, Precision, Recall, and F1-Score**.

---

## **📈 3. Final Model and Results**

The **XGBoost Classifier** emerged as the best-performing model. A key optimization was switching the vectorization method from **HashingTF** to **CountVectorizer**. While HashingTF is faster, it can cause hash collisions (mapping different words to the same feature), which introduces noise. CountVectorizer creates a more precise representation by assigning a unique index to each word in the vocabulary.

This change resulted in a significant performance boost for the final XGBoost model.

> **Best Hyperparameters Found for Final XGBoost Model:**
> - `max_depth`: 6
> - `n_estimators`: 100

> **Final Performance on Test Set (XGBoost with CountVectorizer):**
> - **Accuracy:** 84.18%
> - **Precision:** 84.25%
> - **Recall:** 84.18%
> - **F1-Score:** 84.18%

The final model was validated on new, unseen sentences and correctly classified both positive and negative sentiments, confirming its effectiveness.

---

## **🚀 4. How to Run This Project**

To run this notebook on your own machine or cloud environment, follow these steps:

1.  **Prerequisites:**
    - Python 3.x installed.
    - An environment to run Jupyter Notebooks, such as Jupyter Lab or Google Colab.

2.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    cd your-repository-name
    ```

3.  **Install Libraries:**
    Open the notebook and run the following cell to install the required Python packages:
    ```python
    !pip install pyspark xgboost wordcloud matplotlib nltk
    ```

4.  **Dataset:**
    Ensure the dataset file `imdb-reviews-pt-br.csv` is located in the correct directory as referenced in the notebook.

5.  **Execute the Notebook:**
    Run the cells of the `sentiment-analysis-NLP.ipynb` notebook sequentially from top to bottom.

---

## **🎓 5. Credits and Acknowledgements**

This project was developed as part of the **"Spark: Processamento de Linguagem Natural com PySpark"** course from **Alura**.

I would like to extend my sincere gratitude to the entire Alura team for creating this excellent course.

A special thanks to the instructor, **Ana Duarte**, for her clear and insightful teaching.

- **Course Link:** [https://cursos.alura.com.br/course/spark-processamento-linguagem-natural](https://cursos.alura.com.br/course/spark-processamento-linguagem-natural)

---

## **👩‍💻 6. Author**

**Sthefanie Otaviano**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](http://linkedin.com/in/sthefanie-ferreira-de-s-d-otaviano-976a59206)
