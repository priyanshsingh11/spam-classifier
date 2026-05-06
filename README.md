# Email Spam Classifier

A production-ready machine learning application that classifies emails as Spam or Ham (Not Spam). This project incorporates MLOps best practices by using DVC (Data Version Control) and AWS S3 for data and model versioning.

## Key Features
- Machine Learning: Built using Scikit-learn (Multinomial Naive Bayes).
- Web Interface: User-friendly UI built with Flask, HTML, and CSS.
- Data Versioning: DVC integration to handle large datasets and model artifacts without bloating Git.
- Cloud Storage: AWS S3 integration as a remote storage backend for DVC.

## Tech Stack
- Backend: Python, Flask
- Machine Learning: Scikit-learn, Pandas, Pickle
- Data Versioning: DVC
- Cloud: AWS S3
- Frontend: HTML5, CSS3

## Project Structure
```text
.
├── .dvc/               # DVC configuration
├── templates/          # HTML templates for Flask
│   ├── index.html      # Homepage
│   ├── result.html     # Classification results
│   └── error.html      # Error handling
├── main.py             # Flask application entry point
├── spam.csv.dvc        # DVC tracking for dataset
├── model.pkl.dvc       # DVC tracking for trained model
├── vectorizer.pkl.dvc  # DVC tracking for vectorizer
├── spam detection project.ipynb # Training notebook
└── README.md           # Project documentation
```

## Setup and Installation

### 1. Clone the Repository
```bash
git clone https://github.com/priyanshsingh11/spam-classifier.git
cd spam-classifier
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
pip install dvc[s3]
```

### 3. Pull Data and Models (DVC)
Since the large data files and models are not stored in Git, you need to pull them from the S3 remote:
```bash
# Configure AWS CLI first if not already done
aws configure

# Pull the assets
dvc pull
```

## Running the Application
```bash
python main.py
```
Visit http://127.0.0.1:5000 in your browser to test the classifier.

## MLOps Workflow
To update the data or model:
1. Modify the file (e.g., spam.csv).
2. Run dvc add spam.csv.
3. Commit the updated .dvc file: git add spam.csv.dvc && git commit -m "Update dataset".
4. Push the actual data to S3: dvc push.

---
Created by [Priyansh Singh](https://github.com/priyanshsingh11)
