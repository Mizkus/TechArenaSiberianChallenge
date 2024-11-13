#optimization joins
import copy

n_tables = int(input())
table_sizes = [int(x) for x in input().split()]
n_attributes = int(input())
attributes = [None] * n_tables
for i in range(n_attributes):
    line = input().split()
    table = int(line[0])
    attr = line[1]
    card = int(line[2])
    if not attributes[table - 1]:
        attributes[table - 1] = {}
    attributes[table - 1][attr] = (card)
n_scan_preds = int(input())
scan_preds = [set() for _ in range(n_tables)]
for i in range(n_scan_preds):
    line = input().split()
    table = int(line[0])
    attr = line[1]
    scan_preds[table - 1].add(attr)

n_join_preds = int(input())
joins = [None] * n_tables
for i in range(n_tables):
    joins[i] = []
    for j in range(n_tables):
        joins[i].append(set())

for i in range(n_join_preds):
    line = input().split()
    fst_table = int(line[0])
    snd_table = int(line[1])
    fst_attr = line[2]
    snd_attr = line[3]
    joins[fst_table-1][snd_table-1].add((fst_attr, snd_attr))
    joins[snd_table-1][fst_table-1].add((snd_attr, fst_attr))
#adding transitivity
for i in range(n_tables):
    for j in range(n_tables):
        for k in range(n_tables):
            if joins[i][j] and joins[j][k] and i != k and not joins[i][k]:
                edges1 = joins[i][j]
                edges2 = joins[j][k]
                for edge1 in edges1:
                    attr1 = edge1[0]
                    attr2 = edge1[1]
                    for edge2 in edges2:
                        if (edge2[0] == attr2 and (attr1, edge2[1]) not in joins[i][k]):
                            joins[i][k].add((attr1, edge2[1]))
                            joins[k][i].add((edge2[1], attr1))

dp = [None] * 2**n_tables

for i in range(n_tables):
    idx = 1 << i
    dp[idx] = {}
    dp[idx]['tree'] = str(i+1)
    dp[idx]['tables'] = [i+1]
    dp[idx]['cost'] = (table_sizes[i])
    dp[idx]['n_rows'] = (table_sizes[i])
    dp[idx]['plan'] = []
    dp[idx]['cross_idx'] = 1
    if scan_preds[i]:
        dp[idx]['cost'] *= 2
        for attr in scan_preds[i]:
            dp[idx]['tree'] = "".join([dp[idx]['tree'], attr])
            dp[idx]['n_rows'] /= attributes[i][attr]
joins_ = copy.deepcopy(joins)
for i in range(n_tables-1):
    for j in range(i+1, n_tables):
        for edge in joins[i][j]:
            dpi = 1 << i
            dpj = 1 << j
            if (scan_preds[i] and edge[0] in scan_preds[i]) or (scan_preds[j] and edge[1] in scan_preds[j]):
                    if (scan_preds[i] and edge[0] in scan_preds[i]) and (not scan_preds[j] or edge[1] not in scan_preds[j]):
                        dp[dpj]['n_rows'] /= attributes[j][edge[1]]
                        dp[dpj]['cost'] *= 2
                        dp[dpj]['tree'] += edge[1]
                        scan_preds[j].add(edge[1])
                    elif (not scan_preds[i] or edge[0] not in scan_preds[i]):
                        dp[dpi]['n_rows'] /= attributes[i][edge[0]]
                        dp[dpi]['cost'] *= 2
                        dp[dpi]['tree'] += edge[0]
                        scan_preds[i].add(edge[0])
                    joins_[i][j].remove((edge[0], edge[1]))
                    joins_[j][i].remove((edge[1], edge[0]))
joins = joins_

def create_join_tree(p1, p2):
    result_rows = p1['n_rows'] * p2['n_rows']
    p = {
        'tables': p1['tables'] + p2['tables'],
        'tree': "",
        'n_rows': result_rows,
        'cost': float('inf'),
        'plan': []
    }
    
    filtr1 = {
        table: {x: [0, ""] for x in attributes[table-1]}
        for table in p1['tables'] 
        if attributes[table - 1]
    }
    
    predicates = []
    found_inner = False
    
    for table1 in p1['tables']:
        for table2 in p2['tables']:
            join_edges = joins[table1-1][table2-1]
            if not join_edges:
                continue
                
            found_inner = True
            for edge in join_edges:
                attr1, attr2 = edge
                coeff = max(attributes[table1-1][attr1], attributes[table2-1][attr2])
                
                if coeff > filtr1[table1][attr1][0]:
                    filtr1[table1][attr1] = [coeff, f"{table2}.{attr2}"]
    
    if found_inner:
        seen_predicates = set()
        for table1, val1 in filtr1.items():
            for table2, val2 in filtr1.items():
                if table1 == table2:
                    continue
                    
                for attr1, (c1, pred1) in val1.items():
                    for attr2, (c2, pred2) in val2.items():
                        if pred1 == pred2:
                            if c1 > c2:
                                filtr1[table2][attr2][0] = 0
                            else:
                                filtr1[table1][attr1][0] = 0
        
        for table, val in filtr1.items():
            for attr, (c, pred) in val.items():
                if c != 0:
                    result_rows /= c
                    pred_str = "{" + f"{table}.{attr} {pred}" + "}"
                    if pred_str not in seen_predicates:
                        predicates.append(pred_str)
                        seen_predicates.add(pred_str)
        
        cost_sum = p1['cost'] + p2['cost']
        rows_product = result_rows * 0.1
        
        costs = [
            (cost_sum + p2['n_rows']*1.1 + (p1['n_rows']-1)*p2['n_rows'] + rows_product, False),  # nlj
            (cost_sum + p1['n_rows']*1.1 + (p2['n_rows']-1)*p1['n_rows'] + rows_product, True),   # nlj_inv
            (cost_sum + p2['n_rows']*1.5 + p1['n_rows']*3.5 + rows_product, False),               # hj
            (cost_sum + p1['n_rows']*1.5 + p2['n_rows']*3.5 + rows_product, True)                 # hj_inv
        ]
        
        min_cost, is_inverted = min(costs, key=lambda x: x[0])
        
        if is_inverted:
            predicates = ["{" + " ".join(reversed(pred[1:-1].split(' '))) + "}" for pred in predicates]
            p.update({
                'tree': f"({p2['tree']} {p1['tree']} {' '.join(predicates)})",
                'plan': p2['plan'] + ['inner'] + p1['plan'],
                'cost': min_cost,
                'n_rows': result_rows
            })
        else:
            p.update({
                'tree': f"({p1['tree']} {p2['tree']} {' '.join(predicates)})",
                'plan': p1['plan'] + ['inner'] + p2['plan'],
                'cost': min_cost,
                'n_rows': result_rows
            })
    else:
        cost1 = p1['cost'] + p2['cost'] + p2['n_rows']*0.2 + (p1['n_rows']-1)*p2['n_rows']*0.1
        cost2 = p1['cost'] + p2['cost'] + p1['n_rows']*0.2 + (p2['n_rows']-1)*p1['n_rows']*0.1
        
        if cost1 <= cost2:
            p.update({
                'tree': f"({p1['tree']} {p2['tree']})",
                'plan': p1['plan'] + ['cross'] + p2['plan'],
                'cost': cost1
            })
        else:
            p.update({
                'tree': f"({p2['tree']} {p1['tree']})",
                'plan': p2['plan'] + ['cross'] + p1['plan'],
                'cost': cost2
            })
    
    half_plan = len(p['plan']) // 2
    if 'cross' not in p['plan']:
        p['cross_idx'] = half_plan + 1
    else:
        try:
            first_cross = next(i for i in range(half_plan) 
                             if p['plan'][i] == 'cross' or p['plan'][-i-1] == 'cross')
            p['cross_idx'] = first_cross
        except StopIteration:
            p['cross_idx'] = half_plan
            
    return p

def dp_sub(dp, subtrees):

    for i in range(2, subtrees + 1):
        if not bin(i).count('1') > 1:
            continue
            
        s = i
        set1 = s & (-s)  
        
        while set1 != s and set1 <= (s >> 1):
            set2 = s - set1
            
            if dp[set1] and dp[set2]:  
                p = create_join_tree(dp[set1], dp[set2])
                if not dp[s] or p['cross_idx'] > dp[s]['cross_idx'] or p['cost'] < dp[s]['cost']:
                    dp[s] = p
                    
            set1 = s & (set1 - s)  
            
    return dp[subtrees]
    
out = dp_sub(dp, 2**n_tables - 1)

print(out['tree'], "{:.10f}".format(out['cost']))