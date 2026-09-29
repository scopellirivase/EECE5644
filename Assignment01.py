import numpy as np

rng = np.random.default_rng(seed=42)

# ==============================================

def get_block(M, row, col, p, q):
    # A -> R^p*p; C -> R^q*q
    return M[row * p : (row + 1) * p, col * q : (col + 1) * q]

def schur_proof(n):
    p, q = 3, 3
    W = rng.random((n,n))
    
    M = W @ W.T + np.eye(n)
    
    # Define blocks
    A = get_block(M, 0, 0, p, q)
    B = get_block(M, 0, 1, p, q)
    C = get_block(M, 1, 1, p, q)
    
    S_A = C - np.transpose(B) @ np.linalg.inv(A) @ B
    
    detM = np.linalg.det(M)
    detA = np.linalg.det(A)
    detSA = np.linalg.det(S_A)
    eigSA = np.linalg.eigvals(S_A)
    
    print(f"det(M)            = {detM}")
    print(f"det(A) * det(S_A) = {detA*detSA}")
    print(f"eigenvalues(S_A)  = {eigSA}")

# ==============================================

# P1
schur_proof(6)

# P2

