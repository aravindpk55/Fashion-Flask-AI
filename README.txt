README – FashionFlask Clothing Review Application

Project Title:
FashionFlask – Clothing Recommendation and Review System using Flask & Machine Learning

Overview:
FashionFlask is a web application that allows users to browse clothing items, write reviews, and receive personalized recommendations based on their input. It integrates a trained machine learning model to predict whether a user would recommend a product based on review text and rating.

Folder Structure:

s4106389/
│
├── app.py                  # Main Flask application file
├── train_model.py          # Script to train and save ML model
├── assignment3_II.csv      # Dataset used for training and displaying products
├── model.pkl               # Trained Logistic Regression model
├── vectorizer.pkl          # TF-IDF vectorizer for review text
├── presentation.mp4        # Video presentation file for assignment demonstration
│
├── templates/              # Folder for HTML templates
│   ├── base.html
│   ├── home.html
│   ├── categories.html
│   ├── item.html
│   ├── review_form.html
│   └── review_result.html
│
├── static/
│   └── styles.css          # Custom CSS for visual styling

How to Run the Application Locally:

1. Make sure all required Python packages are installed:
   pip install flask pandas scikit-learn scipy joblib

2. Train the machine learning model (only if not already trained):
   python train_model.py

3. Run the Flask app:
   python app.py

4. Open your browser and go to:
   http://127.0.0.1:5000/

Features:

- View clothing items and categories
- Submit reviews with title, text, and rating
- ML model predicts recommendation status
- See “You might also like” items using intelligent filtering
- Modern, clean UI with Bootstrap and custom CSS

Author:
Aravind Paruvathaselvam Kokila (s4106389)
Master of datascience Student, APDS – RMIT University
