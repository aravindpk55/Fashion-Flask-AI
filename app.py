# Import required libraries
from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import os
import joblib
from scipy.sparse import hstack

# Initialize Flask app
app = Flask(__name__)

# Load the clothing data and the pre-trained ML model and vectorizer
items_df = pd.read_csv("assignment3_II.csv")
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Temporary storage for user-submitted reviews (will be lost on app restart)
session_reviews = []

# Homepage: shows the first 20 items
@app.route('/')
def home():
    return render_template('home.html', items=items_df.head(20))

# Category page: shows items that belong to a selected category
@app.route('/category/<string:cat>')
def category(cat):
    filtered = items_df[items_df['Class Name'].str.lower() == cat.lower()]
    return render_template('categories.html', items=filtered, category=cat)

# Item detail page: shows the item, its reviews, and recommendations
@app.route('/item/<int:item_id>')
def item(item_id):
    # Get the data for the selected item
    item_data = items_df[items_df['Clothing ID'] == item_id].iloc[0]

    # Get all original reviews for that item
    original_reviews = items_df[items_df['Clothing ID'] == item_id][['Title', 'Review Text', 'Rating']].copy()
    original_reviews['Review Text'] = original_reviews['Review Text'].fillna('')
    original_reviews['Rating'] = original_reviews['Rating'].fillna(0)
    
    # Predict recommendation for original reviews using ML model
    X_text = vectorizer.transform(original_reviews['Review Text'])
    X_rating = original_reviews[['Rating']].astype(float).values
    X_combined = hstack([X_text, X_rating])
    original_reviews['Recommended'] = model.predict(X_combined)

    # Include temporary user-submitted reviews stored in session
    custom_reviews = [r for r in session_reviews if r['Clothing ID'] == item_id]
    custom_df = pd.DataFrame(custom_reviews)
    
    # Merge both original and session-based reviews
    all_reviews = pd.concat([original_reviews, custom_df], ignore_index=True)

    # Suggest 3 similar recommended items (same category, different ID)
    candidate_items = items_df[(items_df['Class Name'] == item_data['Class Name']) &
                               (items_df['Clothing ID'] != item_id)].copy()
    candidate_items['Review Text'] = candidate_items['Review Text'].fillna('')
    candidate_items['Rating'] = candidate_items['Rating'].fillna(0)
    X_cand_text = vectorizer.transform(candidate_items['Review Text'])
    X_cand_rating = candidate_items[['Rating']].astype(float).values
    X_cand_combined = hstack([X_cand_text, X_cand_rating])
    candidate_items['ML_Recommended'] = model.predict(X_cand_combined)
    recommended_items = candidate_items[candidate_items['ML_Recommended'] == 1][['Clothing ID', 'Title', 'Review Text']].head(3)
    
    # Render the item page with reviews and recommendations
    return render_template('item.html', item=item_data, reviews=all_reviews, recommendations=recommended_items)

# Review submission page: handles both form display and review processing
@app.route('/review/<int:item_id>', methods=["GET", "POST"])
def review(item_id):
    if request.method == "POST":
        title = request.form['title']
        text = request.form['review_text']
        rating = float(request.form['rating'])
        
        # Predict recommendation using ML model
        X_text = vectorizer.transform([text])
        X_combined = hstack([X_text, [[rating]]])
        pred = model.predict(X_combined)[0]

        # Store only in-memory
        new_entry = {
            "Clothing ID": item_id,
            "Title": title,
            "Review Text": text,
            "Rating": rating,
            "Recommended": pred
        }
        session_reviews.append(new_entry)
        
        # Show the user the result of the prediction
        item_data = items_df[items_df['Clothing ID'] == item_id].iloc[0]
        return render_template("review_result.html", item=item_data, prediction=pred, original=title, review=text)
    
    # Show the review form if the request is GET
    item_data = items_df[items_df['Clothing ID'] == item_id].iloc[0]
    return render_template("review_form.html", item=item_data)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
