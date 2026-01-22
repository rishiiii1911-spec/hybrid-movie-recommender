import numpy as np
import pandas as pd


def load_movies():
    return pd.read_csv(
        "data/movies.csv",
        sep="|",
        encoding="latin-1",
        header=None,
        usecols=range(24),
        names=[
            "movie_id", "title", "release_date", "video_release",
            "imdb_url", "unknown", "Action", "Adventure", "Animation",
            "Children", "Comedy", "Crime", "Documentary", "Drama",
            "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
            "Romance", "Sci-Fi", "Thriller", "War", "Western"
        ]
    )

def load_ratings():
    return pd.read_csv(
        "data/ratings.csv",
        sep="\t",
        names=["user_id", "movie_id", "rating", "timestamp"]
    )


def build_user_item_matrix(ratings, num_users, num_movies):
    matrix = np.zeros((num_users, num_movies))
    for _, row in ratings.iterrows():
        matrix[int(row.user_id) - 1, int(row.movie_id) - 1] = row.rating
    return matrix

def hybrid_recommend(user_id, movies, user_item_matrix, top_n=5, alpha=0.7):
    user_idx = user_id - 1


    collab_scores = user_item_matrix[user_idx]


    genre_cols = movies.columns[5:]
    liked = collab_scores > 0

    if liked.sum() == 0:
        content_scores = np.zeros(len(movies))
    else:
        profile = movies.loc[liked, genre_cols].mean(axis=0)
        content_scores = movies[genre_cols].values @ profile.values


    if collab_scores.max() > 0:
        collab_scores = collab_scores / collab_scores.max()
    if content_scores.max() > 0:
        content_scores = content_scores / content_scores.max()


    hybrid_scores = alpha * collab_scores + (1 - alpha) * content_scores

    top_indices = np.argsort(hybrid_scores)[::-1][:top_n]
    return movies.iloc[top_indices]["title"].tolist()



def get_relevant_movies(user_id, ratings, movies, threshold=4):
    user_ratings = ratings[
        (ratings.user_id == user_id) &
        (ratings.rating >= threshold)
    ]
    return movies[movies.movie_id.isin(user_ratings.movie_id)]["title"].tolist()

def precision_recall_at_k(recommended, relevant, k):
    recommended_k = recommended[:k]
    relevant = set(relevant)

    if not relevant:
        return 0.0, 0.0

    hits = len(set(recommended_k) & relevant)
    return hits / k, hits / len(relevant)

def average_precision_at_k(recommended, relevant, k):
    score, hits = 0.0, 0
    relevant = set(relevant)

    for i, movie in enumerate(recommended[:k]):
        if movie in relevant:
            hits += 1
            score += hits / (i + 1)

    return score / min(len(relevant), k) if relevant else 0.0

def ndcg_at_k(recommended, relevant, k):
    relevant = set(relevant)
    dcg = sum(
        1 / np.log2(i + 2)
        for i, m in enumerate(recommended[:k])
        if m in relevant
    )
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


def main():
    movies = load_movies()
    ratings = load_ratings()

    num_users = ratings.user_id.max()
    num_movies = ratings.movie_id.max()

    user_item_matrix = build_user_item_matrix(
        ratings, num_users, num_movies
    )

    while True:
        user_input = input("Enter user ID (1–943) or 'exit': ").strip()

        if user_input.lower() == "exit":
            print("Exiting recommender.")
            break

        if not user_input.isdigit():
            print("Invalid input.\n")
            continue

        user_id = int(user_input)
        if not (1 <= user_id <= num_users):
            print("User ID out of range.\n")
            continue

        relevant_movies = get_relevant_movies(user_id, ratings, movies)
        if not relevant_movies:
            print("No relevant movies for evaluation.\n")
            continue

        print("\n--- Alpha tuning results ---")
        for alpha in [0.3, 0.5, 0.7, 0.9]:
            recommendations = hybrid_recommend(
                user_id,
                movies,
                user_item_matrix,
                alpha=alpha
            )

            precision, recall = precision_recall_at_k(
                recommendations,
                relevant_movies,
                k=5
            )

            map_k = average_precision_at_k(
                recommendations,
                relevant_movies,
                k=5
            )

            ndcg_k = ndcg_at_k(
                recommendations,
                relevant_movies,
                k=5
            )

            print(
                f"alpha={alpha} → "
                f"P@5={precision:.2f}, "
                f"R@5={recall:.2f}, "
                f"MAP@5={map_k:.2f}, "
                f"NDCG@5={ndcg_k:.2f}"
            )

        print("\nRecommended Movies:")
        for m in recommendations:
            print("-", m)
        print()


if __name__ == "__main__":
    main()
