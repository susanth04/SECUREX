import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import re
from urllib.parse import urlparse
import os
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_training.log'),
        logging.StreamHandler()
    ]
)

class URLFeatureExtractor:
    @staticmethod
    def having_ip_address(url):
        match = re.search(
            r'(([01]?\d\d?|2[0-4]\d|25[0-5])\.([01]?\d\d?|2[0-4]\d|25[0-5])\.([01]?\d\d?|2[0-4]\d|25[0-5])\.)', url)
        return 1 if match else 0

    @staticmethod
    def abnormal_url(url):
        hostname = urlparse(url).hostname
        return 0 if hostname and hostname in url else 1

    @staticmethod
    def count_characters(url, char):
        return url.count(char)

    @staticmethod
    def no_of_dir(url):
        return urlparse(url).path.count('/')

    @staticmethod
    def shortening_service(url):
        match = re.search(r'bit\.ly|goo\.gl|t\.co|tinyurl|is\.gd|ow\.ly|shorte\.st|cli\.gs|x\.co|tr\.im', url)
        return 1 if match else 0

    @staticmethod
    def url_length(url):
        return len(str(url))

    @staticmethod
    def hostname_length(url):
        return len(urlparse(url).netloc)

    @staticmethod
    def suspicious_words(url):
        match = re.search(r'paypal|login|signin|bank|account|update|free|bonus|ebay|secure', url, re.IGNORECASE)
        return 1 if match else 0

    @staticmethod
    def digit_count(url):
        return sum(char.isdigit() for char in url)

    @staticmethod
    def letter_count(url):
        return sum(char.isalpha() for char in url)

    @staticmethod
    def fd_length(url):
        try:
            return len(urlparse(url).path.split('/')[1])
        except IndexError:
            return 0

class URLClassifierTrainer:
    def __init__(self, data_path='malicious_phish.csv'):
        self.data_path = data_path
        self.feature_extractor = URLFeatureExtractor()
        self.model = None
        self.feature_names = [
            'use_of_ip', 'abnormal_url', 'count.', 'count-www', 'count@',
            'count_dir', 'short_url', 'count-https', 'count-http', 'count%',
            'count?', 'count-', 'count=', 'url_length', 'hostname_length',
            'sus_url', 'fd_length', 'count-digits', 'count-letters'
        ]

    def load_data(self):
        """Load and validate the dataset"""
        try:
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f"Dataset file not found: {self.data_path}")
            
            df = pd.read_csv(self.data_path)
            
            required_columns = ['url', 'type']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            return df
            
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            raise

    def extract_features(self, df):
        """Extract features from URLs"""
        try:
            logging.info("Extracting features from URLs...")
            
            # Apply feature engineering
            df['use_of_ip'] = df['url'].apply(self.feature_extractor.having_ip_address)
            df['abnormal_url'] = df['url'].apply(self.feature_extractor.abnormal_url)
            df['count.'] = df['url'].apply(lambda i: self.feature_extractor.count_characters(i, '.'))
            df['count-www'] = df['url'].apply(lambda i: self.feature_extractor.count_characters(i, 'www'))
            df['count@'] = df['url'].apply(lambda i: self.feature_extractor.count_characters(i, '@'))
            df['count_dir'] = df['url'].apply(self.feature_extractor.no_of_dir)
            df['short_url'] = df['url'].apply(self.feature_extractor.shortening_service)
            df['count-https'] = df['url'].apply(lambda i: self.feature_extractor.count_characters(i, 'https'))
            df['count-http'] = df['url'].apply(lambda i: self.feature_extractor.count_characters(i, 'http'))
            df['count%'] = df['url'].apply(lambda i: self.feature_extractor.count_characters(i, '%'))
            df['count?'] = df['url'].apply(lambda i: self.feature_extractor.count_characters(i, '?'))
            df['count-'] = df['url'].apply(lambda i: self.feature_extractor.count_characters(i, '-'))
            df['count='] = df['url'].apply(lambda i: self.feature_extractor.count_characters(i, '='))
            df['url_length'] = df['url'].apply(self.feature_extractor.url_length)
            df['hostname_length'] = df['url'].apply(self.feature_extractor.hostname_length)
            df['sus_url'] = df['url'].apply(self.feature_extractor.suspicious_words)
            df['count-digits'] = df['url'].apply(self.feature_extractor.digit_count)
            df['count-letters'] = df['url'].apply(self.feature_extractor.letter_count)
            df['fd_length'] = df['url'].apply(self.feature_extractor.fd_length)
            
            return df
            
        except Exception as e:
            logging.error(f"Error extracting features: {str(e)}")
            raise

    def prepare_data(self, df):
        """Prepare features and target for training"""
        try:
            # Encode target labels
            df["type_code"] = df["type"].astype('category').cat.codes
            
            # Get features and target
            X = df[self.feature_names]
            y = df['type_code']
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logging.error(f"Error preparing data: {str(e)}")
            raise

    def train_model(self, X_train, y_train):
        """Train the Random Forest model"""
        try:
            logging.info("Training Random Forest model...")
            
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1  # Use all available cores
            )
            
            self.model.fit(X_train, y_train)
            
        except Exception as e:
            logging.error(f"Error training model: {str(e)}")
            raise

    def evaluate_model(self, X_test, y_test):
        """Evaluate the trained model"""
        try:
            logging.info("Evaluating model performance...")
            
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            
            logging.info(f"Model Accuracy: {accuracy:.4f}")
            logging.info("\nClassification Report:\n" + report)
            
            return accuracy, report
            
        except Exception as e:
            logging.error(f"Error evaluating model: {str(e)}")
            raise

    def save_model(self, filename='url_classifier.pkl'):
        """Save the trained model"""
        try:
            if self.model is None:
                raise ValueError("No trained model to save!")
            
            with open(filename, 'wb') as f:
                pickle.dump(self.model, f)
            
            logging.info(f"Model saved successfully as '{filename}'")
            
        except Exception as e:
            logging.error(f"Error saving model: {str(e)}")
            raise

    def train(self):
        """Complete training pipeline"""
        try:
            start_time = datetime.now()
            logging.info("Starting model training pipeline...")
            
            # Load data
            df = self.load_data()
            logging.info(f"Loaded dataset with {len(df)} samples")
            
            # Extract features
            df = self.extract_features(df)
            logging.info("Feature extraction completed")
            
            # Prepare data
            X_train, X_test, y_train, y_test = self.prepare_data(df)
            logging.info(f"Data split: {len(X_train)} training samples, {len(X_test)} test samples")
            
            # Train model
            self.train_model(X_train, y_train)
            
            # Evaluate model
            accuracy, report = self.evaluate_model(X_test, y_test)
            
            # Save model
            self.save_model()
            
            end_time = datetime.now()
            training_time = (end_time - start_time).total_seconds()
            logging.info(f"Training completed in {training_time:.2f} seconds")
            
            # Save feature names with the model for future reference
            with open('feature_names.pkl', 'wb') as f:
                pickle.dump(self.feature_names, f)
            
            return accuracy, report
            
        except Exception as e:
            logging.error(f"Training pipeline failed: {str(e)}")
            raise

def main():
    try:
        trainer = URLClassifierTrainer()
        accuracy, report = trainer.train()
        
        # Print final summary
        print("\nTraining Summary:")
        print("=" * 50)
        print(f"Model Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(report)
        print("\nModel and feature names have been saved successfully!")
        
    except Exception as e:
        print(f"Error in training process: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()