import numpy as np

rng = np.random.default_rng(42)

# ==============================================

def schur_proof(n): # Problem 1 - (d)
    W = rng.standard_normal((n, n))
    M = W @ W.T + np.eye(n) # guarantees M symmetric PD
    
    A, B, C = M[:3, :3], M[:3, 3:], M[3:, 3:]
    S_A = C - B.T @ np.linalg.inv(A) @ B
    
    print("P1: ")
    print("det(M) =", np.linalg.det(M))
    print("det(A)*det(S_A) =", np.linalg.det(A) * np.linalg.det(S_A))
    print("eigenvalues(S_A) =", np.linalg.eigvalsh(S_A))
    print()
    
def orthonormal_basis():
    import numpy as np
    a1 = np.array([1, 1, 0, 0.])
    a2 = np.array([0, 1, 1, 0.])
    b = np.array([1, 2, 3, 4.])
    A = np.column_stack([a1, a2])
    
    q1 = a1 / np.linalg.norm(a1)
    u2 = a2 - (q1 @ a2) * q1
    q2 = u2 / np.linalg.norm(u2)
    
    Pb_normal = A @ np.linalg.inv(A.T @ A) @ A.T @ b
    Pb_gs = (q1 @ b) * q1 + (q2 @ b) * q2
    residual = b - Pb_gs
    
    print("P2: ")
    print("q1, q2 =", q1, q2)
    print("Pb (normal eq) =", Pb_normal)
    print("Pb (GS sum) =", Pb_gs)
    print("residual norm =", np.linalg.norm(residual))
    print()

# ==============================================

# P1
schur_proof(6)

# P2
orthonormal_basis()
