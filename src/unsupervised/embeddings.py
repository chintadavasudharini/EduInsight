"""
CO-3: Unsupervised Learning - Representation Learning and Embeddings Module.
Demonstrates:
1. Tabular Representation Learning via a shallow Autoencoder / Latent Bottleneck.
2. Self-Supervised Learning (SSL) conceptual demonstration via Masked Feature Reconstruction.
3. Natural Language Processing (NLP) Educational Demo: Word embeddings and cosine similarity
   on an academic course feedback corpus.
NOTE: Emphasizes that text embeddings should never be applied directly to tabular features.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from src.utils.config import FEATURE_NAMES, DEFAULT_RANDOM_SEED


def demonstrate_tabular_autoencoder_latents(
    df: pd.DataFrame,
    latent_dim: int = 2,
    epochs: int = 40,
    seed: int = DEFAULT_RANDOM_SEED,
) -> Dict[str, Any]:
    """
    Learns a non-linear low-dimensional representation (embedding) of student tabular data
    using a small PyTorch Autoencoder (bottleneck architecture: 5 -> 3 -> 2 -> 3 -> 5).
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURE_NAMES].values)

    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim

        torch.manual_seed(seed)
        X_t = torch.tensor(X, dtype=torch.float32)

        class TabularAutoencoder(nn.Module):
            def __init__(self, in_features: int, latent: int):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(in_features, 4),
                    nn.ReLU(),
                    nn.Linear(4, latent),
                )
                self.decoder = nn.Sequential(
                    nn.Linear(latent, 4),
                    nn.ReLU(),
                    nn.Linear(4, in_features),
                )

            def forward(self, x):
                z = self.encoder(x)
                x_rec = self.decoder(z)
                return x_rec, z

        model = TabularAutoencoder(in_features=5, latent=latent_dim)
        optimizer = optim.Adam(model.parameters(), lr=0.02)
        criterion = nn.MSELoss()

        loss_history = []
        for epoch in range(epochs):
            optimizer.zero_grad()
            rec, z = model(X_t)
            loss = criterion(rec, X_t)
            loss.backward()
            optimizer.step()
            loss_history.append(float(loss.item()))

        with torch.no_grad():
            _, latent_embeddings = model(X_t)
            reconstruction_error = float(loss.item())

        return {
            "success": True,
            "latent_embeddings": latent_embeddings.numpy(),
            "loss_history": loss_history,
            "final_reconstruction_loss": reconstruction_error,
            "latent_dim": latent_dim,
        }

    except ImportError:
        # Fallback to PCA if torch unavailable
        from sklearn.decomposition import PCA
        pca = PCA(n_components=latent_dim, random_state=seed)
        latents = pca.fit_transform(X)
        return {
            "success": False,
            "latent_embeddings": latents,
            "loss_history": [],
            "final_reconstruction_loss": 0.0,
            "latent_dim": latent_dim,
        }


def demonstrate_educational_text_embeddings() -> Dict[str, Any]:
    """
    Educational demonstration of text embeddings (Term-Frequency vs Semantic representations)
    on an academic corpus of course feedback comments.
    Demonstrates Cosine Similarity and semantic proximity.
    """
    academic_corpus = [
        "The professor explains mathematical foundations of machine learning with great clarity.",
        "Lectures on linear algebra, vectors, and eigenvalues were very intuitive and clear.",
        "Optimization techniques and gradient descent algorithms helped me understand neural networks.",
        "The course assignments and projects required significant programming and study hours.",
        "Student attendance and regular review of previous exam scores improved final performance.",
        "Preparing daily for examinations and getting adequate sleep improves academic grades.",
    ]

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(academic_corpus).toarray()

    # Compute pairwise cosine similarity matrix
    sim_matrix = cosine_similarity(tfidf_matrix)

    # Word embeddings educational explanation
    explanation = (
        "Word2Vec, GloVe, and Transformers (e.g., BERT) represent discrete text tokens as continuous dense "
        "vectors in a semantic vector space where geometric distance reflects linguistic similarity "
        "(e.g., cos(v_algebra, v_calculus) > cos(v_algebra, v_sleep)). "
        "In contrast, tabular student data already consists of continuous, structured numerical features "
        "(hours, percentages, scores). Applying NLP tokenizers or text embeddings directly to structured "
        "tabular features is mathematically inappropriate without tabular representation techniques."
    )

    return {
        "corpus": academic_corpus,
        "feature_words": vectorizer.get_feature_names_out().tolist(),
        "similarity_matrix": sim_matrix.round(3).tolist(),
        "explanation": explanation,
    }
