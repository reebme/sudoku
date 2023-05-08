#!/usr/bin/python

#
# Solving sudoku puzzles
# 1) as constraint satisfaction problem (backtracking, forward_checking, constraint propagation)
# 2) #TODO random assignements
# 3) #TODO genetic alg
# 4) #TODO montecarlo process (same as 2?)
#

import numpy as np
import sys
from math import sqrt
from collections import deque
import copy

def import_problem_from_input():
    '''
    first line is size of the puzzle and size of the domain
    each next line describes initial assignement with 3 space separated values
    row collumn assigned number
    '''
    domain_size = int(input())
    # assignement
    sp = np.zeros((domain_size, domain_size), dtype=int)
    for line in sys.stdin:
        i, j, val = map(int, line.strip().split())
        sp[i][j] = val
    return domain_size, sp

def generate_constraints_graph(domain_size):
    adj_list = {}
    for i in range(domain_size):
        for j in range(domain_size):
            adj_list[(i,j)] = []
    no_of_squares = int(sqrt(domain_size))
    for (i, j) in adj_list.keys():
        for m in range(domain_size):
            if m != j:
                adj_list[(i,j)].append((i,m))
            if m != i:
                adj_list[(i,j)].append((m,j))
        # which square is (i,j) in
        sx = 0
        sy = 0
        for s in range(no_of_squares):
            start = s * no_of_squares
            end = (s + 1) * no_of_squares - 1
            if i >= start and i <= end:
                sx = s
            if j >= start and j <= end:
                sy = s
        # add constraints from the square
        for t in range(sx * no_of_squares, (sx + 1) * no_of_squares):
            for u in range(sy * no_of_squares, (sy + 1) * no_of_squares):
                if (t, u) not in adj_list[(i,j)] and (t, u) != (i, j):
                    adj_list[(i,j)].append((t, u))
    return adj_list

def propagate_constraint(domain, nl, val):
    '''
    nl: list of two element tuples,
        list of constraint coordinates
    '''
    for (m, n) in nl:
        try:
            domain[m][n].remove(val)
        except ValueError:
            pass
        if len(domain[m][n]) == 0:
            # no solution
            return False
    return True

def forward_check(ass, cg, domain):
    '''
    ass
        assignement to be checked
    cg
        constraints graph
    domain
        possible values for assignement
    '''
    rows, cols = ass.shape
    for i in range(rows):
        for j in range(cols):
            # if a value has been assigned
            if ass[i][j] > 0:
                # remove other values from domain
                domain[i][j] = [ass[i][j]]
                # propagate contraint
                if not propagate_constraint(domain, cg[(i,j)], ass[i][j]):
                    return False
    return True

def arc_constraints(ass, cg, domain):
    '''
    Propagates constraints only for variables with domain with one element
    ass
        assignement to be checked
    cg
        constraints graph
    domain
        possible values for assignement
    returns False when propagation results in an empty domain for any variable
    '''
    rows, cols = domain.shape
    for i in range(rows):
        for j in range(cols):
            if len(domain[i][j]) == 1:
                if not propagate_constraint(domain, cg[(i, j)], domain[i][j][0]):
                    return False
    return True

def choose_most_constraint(domain, assignement):
    rows, cols = domain.shape
    min_len = rows + 1 
    x, y = -1, -1
    for i in range(rows):
        for j in range(cols):
            # if one value left in domain and value has not been assigned
            if len(domain[i][j]) == 1 and assignement[i][j] == 0:
                return (i, j)
            elif len(domain[i][j]) < min_len and assignement[i][j] == 0:
                min_len = len(domain[i][j])
                (x, y) = (i, j)
    return (x, y)

def assignement_complete(assig):
    '''
    assig: 2D numpy array
        current assignement
    '''
    rows, cols = assig.shape
    for i in range(rows):
        for j in range(cols):
            if assig[i][j] == 0:
                return False
    return True

def print_domain(domain, assignement):
    rows, cols = domain.shape
    for i in range(rows):
        temp = []
        for j in range(cols):
            # rows = cols = domain range
            if assignement[i][j] > 0:
                temp.append(f"{assignement[i][j]:<{rows}}")
            else:
                for v in range(rows):
                    if v in domain[i][j]:
                        temp.append(str(v))
                    else:
                        temp.append(' ')
            temp.append('|')
        print(''.join(temp))

def generate_domain(domain_size):
    # sp_domain: 2D numpy array of lists
    # #TODO there may be better structure than a list for quick removal of elements from the middle
    # represents possible values for each field
    sp_domain = np.empty((domain_size, domain_size), dtype = object)
    for i in range(domain_size):
        for j in range(domain_size):
            sp_domain[i][j] = [m for m in range(1, domain_size + 1)]
    return sp_domain
 
def solve_sudoku_csp(domain_size, sp, cg):
    '''
    domain_size: int
        size of sudoku puzzle
    sp: np.array domain_size x domain_size
        sudoku puzzle, initial assignement
    cg: adjacency list
        constraints represented as a graph
    Treats the problem as constraint satisfaction problem (csp).
    Utilizes backtracking with forward and arc-consistency checking
    returns solved sudoku puzzle (2D array) or False if no solutionw as found
    '''
    sp_domain = generate_domain(domain_size)
    
    # some benchmarking variables
    # no of checked assignements
    no_of_assig_checks = 0
    # no of times algorithm backtracked in the tree of possible assignements
    back = 0
    # longest the queue has been
    q_len = 0
    
    # puzzle solution
    forward_check(sp, cg, sp_domain)
    arc_constraints(sp, cg, sp_domain)
    q = deque()
    #print(sp)
    #print_domain(sp_domain, sp)
    #input()
    q.append((sp, sp_domain))
    while q:
        if q_len < len(q):
            q_len = len(q)
        sp, sp_domain = q.pop()
        if assignement_complete(sp):
            print(f"Number of assignements checked: {no_of_assig_checks}")
            print(f"Number of times algorithm backtracked: {back}")
            print(f"Longest the queue has been: {q_len}")
            return sp
        no_of_assig_checks += 1
        i, j = choose_most_constraint(sp_domain, sp)
        for val in sp_domain[i][j]:
            temp_sp = copy.deepcopy(sp)
            temp_sp[i][j] = val
            temp_sp_domain = copy.deepcopy(sp_domain)
            # forward checking
            if not forward_check(temp_sp, cg, temp_sp_domain):
                back += 1
                #print("Backtracking")
                #input()
                continue
            # constraint propagation
            if not arc_constraints(temp_sp, cg, temp_sp_domain):
                back += 1
                #print("Backtracking")
                #input()
                continue
            #print(temp_sp)
            #print_domain(temp_sp_domain, temp_sp)
            #input()
            q.append((temp_sp, temp_sp_domain))
    return False

def choose_first_unassigned(sp):
    '''
    sp: 2D numpy array
        sudoku puzzle - current assignement, unassigned varialbles = 0
    '''
    rows, cols = sp.shape
    for i in range(rows):
        for j in range(cols):
            if sp[i][j] == 0:
                return (i, j)

def find_legal_values(domain_size, sp, cg, i, j):
    legal_vals = list(range(1, domain_size + 1))
    for v in range(1, domain_size + 1):
        for m, n in cg[(i,j)]:
            if sp[m][n] == v:
                legal_vals.remove(v)    
                break
    return legal_vals 

def solve_sudoku_backtracking(domain_size, sp, cg):
    '''
    domain_size: int
        size of sudoku puzzle
    sp: np.array domain_size x domain_size
        sudoku puzzle, initial assignement
    cg: adjacency list
        constraints represented as a graph
    Treats the problem as constraint satisfaction problem (csp).
    '''
    # benchmarking variables
    # no of times the algorithm backtracked
    back = 0
    # longest the queue has been
    q_len = 0
    # no of checked assignements
    no_of_assig_checks = 0

    #TODO test it
    q = deque()
    q.append(sp)
    while q:
        if q_len < len(q):
            q_len = len(q)
        sp = q.pop()
        if assignement_complete(sp):
            print(f"Number of assignements checked: {no_of_assig_checks}")
            print(f"Number of times algorithm backtracked: {back}")
            print(f"Longest the queue has been: {q_len}")
            return sp
        no_of_assig_checks += 1
        i, j = choose_first_unassigned(sp)
        # assign legal value
        vals = find_legal_values(domain_size, sp, cg, i, j)
        if vals:
            for v in vals:
                temp_sp = copy.deepcopy(sp)
                temp_sp[i][j] = v
                q.append(temp_sp)
        else:
            # no legal values found, backtrack
            back += 1
            continue

if __name__ == '__main__':
    domain_size, sp = import_problem_from_input()
    cg = generate_constraints_graph(domain_size)
    print(sp)
    print(solve_sudoku_backtracking(domain_size, sp, cg))
    print(solve_sudoku_csp(domain_size, sp, cg))
