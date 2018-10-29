from pulp import *
import numpy as np

# 主問題
Main = False
if Main:
    A = np.array([[3,1,2], [1,3,0], [0,2,4]])
    c = np.array([150, 200, 300])
    b = np.array([60, 36, 48])
    (m, n) = A.shape
    prob = LpProblem(name='Production', sense=LpMaximize)

    x = [LpVariable('x'+str(i+1), lowBound=0) for i in range(n)]
    prob += lpDot(c, x)
    for i in range(m):
        prob += lpDot(A[i], x) <= b[i], 'ineq'+str(i)
    print(prob)
    prob.solve()
    print(LpStatus[prob.status])
    print(f'Optimal value = {value(prob.objective)}')
    for v in prob.variables():
        print(f'{v.name} = {v.varValue}')

# 双対問題
else:
    A = np.array([[3, 1, 0], [1, 3, 2], [2, 0, 4]])
    c = np.array([60, 36, 48])
    b = np.array([150, 200, 300])
    (m, n) = A.shape
    prob = LpProblem(name='Production', sense=LpMinimize)

    x = [LpVariable('y'+str(i+1), lowBound=0) for i in range(n)]
    prob += lpDot(c, x)

    for i in range(n):
        prob += lpDot(A[i], x) >= b[i], 'ineq'+str(i)
    print(prob)
    prob.solve()
    print('Status:')
    print(LpStatus[prob.status])
    print(f'Optimal value = {value(prob.objective)}')
    for v in prob.variables():
        print(f"{v.name} = {v.varValue}")

Y = np.array([v.varValue for v in prob.variables()])
print(np.all(np.abs(np.dot(A, Y) -b) <= 1.0e-5 ))
