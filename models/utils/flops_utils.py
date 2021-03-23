import numpy as np

def randomized_svd_flops(m, n, r):
    r2 = np.power(r, 2)
    r3 = np.power(r, 3)
    list_of_terms = [2 * m * n*r, 
                     2 * m * (r**2) - ((2.0/3.0) * (r**3)),
                    4*(r**2)*m + 2*(r**3) - 2*(r**2),
                    14*n*(r**2) + 8*(r**3),
                    2*m*(r**2)]
    return sum(list_of_terms)

def svd_flops(m, n, r):
    n2 = n**2
    n3 = n**3
    k = 5
    list_of_terms = [4*m*n2 - ((4.0/3.0) * n3),
                     k*r*n,
                     4*r*m*n - 2*r*n2 + 2*r*n,
                     2*r*n*(n+1)
                    ]
    return sum(list_of_terms)