import torch

class RankMeMetric:
    def __init__(self, eps=1e-12):
        self.eps = eps

    def compute(self, X):
        cov_matrix = torch.cov(X.T)
        eigenvalues, _ = torch.linalg.eigh(cov_matrix)
        eigenvalues = eigenvalues.clamp(min=self.eps)
        variance = eigenvalues.sum()
        p = eigenvalues / variance

        # Compute spectral entropy
        H = -(p * p.log()).sum()

        # Effective rank
        effective_rank = H.exp()
        return effective_rank, torch.sort(eigenvalues)
