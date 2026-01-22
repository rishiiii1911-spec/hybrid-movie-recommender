 Hybrid Movie Recommendation System (Terminal-Based)

A terminal-based Hybrid Movie Recommendation System that combines Collaborative Filtering and Content-Based Filtering to generate personalized movie recommendations.
Built using the MovieLens dataset, with evaluation using industry-standard metrics.

 Features:
*Hybrid recommendation (Collaborative + Content-Based)
*User-based interaction via terminal
*Alpha tuning to balance recommendation strategies
*Evaluation metrics:
Precision@K

Recall@K

MAP@K (Mean Average Precision)

NDCG@K

Clean, single-file implementation for clarity

 Recommendation Strategy
1. Collaborative Filtering

Uses user–item interaction matrix

Learns preferences from user ratings

Captures crowd behavior and trends

2. Content-Based Filtering

Uses movie genre vectors

Builds a user preference profile from liked movies

Recommends similar content

3. Hybrid Approach

Final score is a weighted combination:

Hybrid Score = α × Collaborative Score + (1 − α) × Content Score


Where:

α ∈ [0, 1]

Higher α → more collaborative influence

Lower α → more content-based influence

Evaluation Metrics:

The system evaluates recommendations using:

Precision@K – Relevance of top K recommendations

Recall@K – Coverage of relevant items

MAP@K – Ranking quality across relevant items

NDCG@K – Ranking importance with position weighting

Alpha tuning is performed to observe performance trade-offs.

🗂️ Project Structure
hybrid-movie-recommender/
│
├── hybrid_recommender.py   # Main hybrid recommendation logic
├── requirements.txt        # Python dependencies
├── .gitignore
├── data/
│   ├── movies.csv          # Movie metadata
│   └── ratings.csv         # User ratings
└── README.md

 Installation & Setup:
1️ Clone the repository
git clone https://github.com/rishiiii1911-spec/hybrid-movie-recommender.git
cd hybrid-movie-recommender

2️ Create and activate virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

3️ Install dependencies
pip install -r requirements.txt

▶ How to Run
python hybrid_recommender.py
You will be prompted to show:
Enter user ID (1–943) or 'exit':

The system will:
Generate recommendations
Display evaluation metrics
Show top recommended movies

 Sample Output
--- Alpha tuning results ---
alpha=0.3 → P@5=1.00, R@5=0.03, MAP@5=1.00, NDCG@5=1.00
alpha=0.5 → P@5=1.00, R@5=0.03, MAP@5=1.00, NDCG@5=1.00

Recommended Movies:
- Stand by Me (1986)
- 2001: A Space Odyssey (1968)
- Casablanca (1942)

 Future Improvements:

Add matrix factorization (SVD)
Normalize popularity bias
Add cold-start handling
Convert to API or web app (Streamlit / FastAPI)
Larger datasets (MovieLens 1M / 20M)

 Author :
Rishi K
B.Tech CSE (AI & Data Science)
Hybrid Recommendation | Machine Learning | Data Science

LICENCE :
This project is for educational and learning purposes.