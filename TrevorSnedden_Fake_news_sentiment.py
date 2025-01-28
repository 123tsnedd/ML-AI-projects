import pandas as pd
import numpy as np
from sklearnex import patch_sklearn
patch_sklearn()
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import nltk
from collections import Counter
import string
from scipy.sparse import hstack

# Download required NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Utility Functions
def get_tokens_tags(row):
    """Tokenize and tag parts of speech"""
    tokens = nltk.word_tokenize(row)
    tagged = nltk.pos_tag(tokens)
    return tokens, tagged

def get_puct_count(row):
    """Count punctuation occurrences"""
    punc_count = Counter()
    for elem in row:
        if elem in string.punctuation:
            punc_count[elem] += 1
    punc_total = sum(punc_count.values())
    return punc_count, punc_total

def get_word_list(row):
    """Extract words by filtering non-alphabetical characters"""
    tokens = nltk.word_tokenize(row)
    filtered = [e for e in tokens if e.isalpha()]
    return filtered

def get_features(row):
    """Extract key features from a single text"""
    row_len = len(row.strip())  # Character count
    word_list = get_word_list(row)  # List of words
    word_count = len(word_list)
    _, punc_total = get_puct_count(row)  # Total punctuation count
    
    features = {
        'row_len': row_len,
        'word_count': word_count,
        'punc_total': punc_total
    }
    return features

def scale_features(feature_df):
    """Scale features using StandardScaler"""
    scaler = preprocessing.StandardScaler()
    return scaler.fit_transform(feature_df)

def transform_article(df, features):
    """Transform articles using TF-IDF"""
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X = vectorizer.fit_transform(df['article'])
    x_combined = hstack([X, features])

    y = df['sentiment'].apply(lambda x: 0 if x.lower() == 'real' else 1)  # Encode labels
    return x_combined, y

def show_hist(df)->None:
    df.hist(bins=20, figsize= (10,6))
    plt.tight_layout
    plt.show()
    return None

def analyse_anomolies(news_df)->None:
    # Separate anomalies and normal points
    anomalies = news_df[news_df['anomaly'] == -1]
    normal_points = news_df[news_df['anomaly'] == 1]

    print(f"Number of Anomalies: {len(anomalies)}")
    print(f"Number of Normal Points: {len(normal_points)}")

    # Display some flagged articles
    print("Sample Anomalies:\n", anomalies[['article', 'sentiment']].head())
    return None

def plot_confusion(y_test, yhat) -> None:
    cm= confusion_matrix(y_test, yhat, labels=[0,1]) #0=real 1 = fake
    #plot
    matrix= ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['real', 'fake'])
    matrix.plot(cmap=plt.cm.Blues)
    plt.title("confusing matrix")
    plt.show()
    return None



# Main Execution
if __name__ == '__main__':
    # Load dataset
    news_df = pd.read_csv('./wk15/News.csv')

    # Ensure the 'article' column exists
    if 'article' not in news_df.columns:
        raise ValueError("Dataset doesn't contain the required 'article' column")
    
    # Extract features
    feature_list = [get_features(row) for row in news_df['article']]
    feature_df = pd.DataFrame(feature_list)
    print("Sample Features:\n", feature_df.head(2))
    
    # Scale the features
    scaled_features = scale_features(feature_df)

    X,y= transform_article(news_df, scaled_features)

    # Split data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=789)

    # Train decision tree
    DT = DecisionTreeClassifier(random_state=40, max_depth=10)
    DT.fit(x_train, y_train)

      # Predict on the test set
    y_pred = DT.predict(x_test)

    # Evaluate model performance
    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['REAL', 'FAKE']))

    # Display feature distributions
    show_hist(feature_df)
    plot_confusion(y_test, y_pred)
