🎬 Movie Review Sentiment Analysis using Naive Bayes
A machine learning project to classify IMDB movie reviews as either positive or negative sentiment using the classic Naive Bayes family of algorithms.

🌟 Project Overview
This project implements a text classification pipeline to perform binary sentiment analysis on the IMDB 50k Movie Reviews Dataset. The pipeline involves thorough text preprocessing, feature extraction using Bag-of-Words (Count Vectorization), and evaluation of three distinct Naive Bayes classifiers: Multinomial, Bernoulli, and Gaussian.

The best-performing model, Multinomial Naive Bayes, achieved an accuracy of over 85%.


🛠️ Technology and Libraries
Language: Python

Core Libraries:

pandas, numpy: Data manipulation.

re, nltk: Text preprocessing (cleaning, stemming, stopword removal).

scikit-learn: Machine learning (vectorization, model training, evaluation).

joblib: Model persistence.










💾 Dataset
The project utilizes the IMDB 50k Movie Reviews Dataset, which contains 50,000 highly polarized reviews for training and testing.

Source: /kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv

Features (X): review (text of the movie review).

Target (Y): sentiment, which was encoded as 1 for positive and 0 for negative.

Data Split: 80% (40,000 reviews) for training and 20% (10,000 reviews) for testing.








⚙️ Methodology and PipelineThe sentiment analysis process follows these sequential steps:Data Loading: The CSV data is loaded into a pandas DataFrame.Preprocessing: A multi-step cleaning process is applied to the review text:Remove HTML tags (e.g., <br />).Lowercasing all text.Remove special characters and replace them with spaces.Remove English stopwords (e.g., 'the', 'a', 'is').Apply Porter Stemming (e.g., 'running' $\rightarrow$ 'run', 'reviews' $\rightarrow$ 'review').Feature Extraction: The preprocessed, tokenized reviews are converted into a numerical feature matrix using CountVectorizer (Bag-of-Words). This resulted in $\mathbf{73,392}$ unique features.Model Training: Three Naive Bayes models were trained on the count-vectorized data.Evaluation: Model performance was assessed using Accuracy, Precision, Recall, and F1-Score.
















Here is a comprehensive README.md file for your sentiment analysis project, structured for clarity and professionalism on GitHub:🎬 Movie Review Sentiment Analysis using Naive BayesA machine learning project to classify IMDB movie reviews as either positive or negative sentiment using the classic Naive Bayes family of algorithms.


🌟 Project OverviewThis project implements a text classification pipeline to perform binary sentiment analysis on the IMDB 50k Movie Reviews Dataset. The pipeline involves thorough text preprocessing, feature extraction using Bag-of-Words (Count Vectorization), and evaluation of three distinct Naive Bayes classifiers: Multinomial, Bernoulli, and Gaussian.The best-performing model, Multinomial Naive Bayes, achieved an accuracy of over 85%.🛠️ Technology and LibrariesLanguage: PythonCore Libraries:pandas, numpy: Data manipulation.re, nltk: Text preprocessing (cleaning, stemming, stopword removal).scikit-learn: Machine learning (vectorization, model training, evaluation).joblib: Model persistence.


💾 DatasetThe project utilizes the IMDB 50k Movie Reviews Dataset, which contains 50,000 highly polarized reviews for training and testing.Source: /kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csvFeatures (X): review (text of the movie review).Target (Y): sentiment, which was encoded as 1 for positive and 0 for negative.Data Split: 80% (40,000 reviews) for training and 20% (10,000 reviews) for testing.⚙️ Methodology and PipelineThe sentiment analysis process follows these sequential steps:Data Loading: The CSV data is loaded into a pandas DataFrame.Preprocessing: A multi-step cleaning process is applied to the review text:Remove HTML tags (e.g., <br />).Lowercasing all text.Remove special characters and replace them with spaces.Remove English stopwords (e.g., 'the', 'a', 'is').Apply Porter Stemming (e.g., 'running' $\rightarrow$ 'run', 'reviews' $\rightarrow$ 'review').Feature Extraction: The preprocessed, tokenized reviews are converted into a numerical feature matrix using CountVectorizer (Bag-of-Words). This resulted in $\mathbf{73,392}$ unique features.Model Training: Three Naive Bayes models were trained on the count-vectorized data.Evaluation: Model performance was assessed using Accuracy, Precision, Recall, and F1-Score.📊 ResultsThe models were evaluated on the 10,000-review test set.ModelAccuracyPrecision (Weighted)Recall (Weighted)F1-Score (Weighted)MultinomialNB0.85240.85290.85240.8524BernoulliNB0.84830.84950.84830.8482GaussianNB0.6278N/AN/AN/AConclusion: The Multinomial Naive Bayes model demonstrated the strongest performance and is selected as the final production model. The Gaussian Naive Bayes model performed poorly, which is typical when applied to sparse, count-based features.



🚀 How to Use the ModelThe trained models and the fitted CountVectorizer are saved using joblib. To predict the sentiment of a new review, you must load these assets and apply the exact same preprocessing pipeline used during training.Saved Assetsmultinomial_nb_model.pkl (The best model)bernoulli_nb_model.pklcountvectorizer.pkl (Crucial for transforming new text)Steps to Predict on New DataLoad the countvectorizer.pkl and multinomial_nb_model.pkl using joblib.load().Apply the full text preprocessing pipeline (cleaning, lowercasing, stopword removal, stemming) to your new review text.Use the loaded CountVectorizer.transform() method to convert the preprocessed text into a numerical feature vector.Call model.predict() on the resulting feature vector to get the sentiment (1 for positive, 0 for negative).



🔗 Repository Structuresentiment-analysis-project/
├── sentiment-analysis.ipynb    # Full code, pipeline steps, and results
├── multinomial_nb_model.pkl    # Saved MultinomialNB model
├── bernoulli_nb_model.pkl      # Saved BernoulliNB model
├── countvectorizer.pkl         # Saved CountVectorizer (required for prediction)
├── IMDB Dataset.csv            # Original dataset (optional, can link to source)
└── README.md                   # This file




