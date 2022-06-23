import time

agents_reg = agents[~np.isnan(agents[:, 12])][:, [0,7,12]]
a1 = []
a2 = []
count = 0
t = time.time()
for n in range(len(agents_reg)):
    a = agents_reg[n, 0]
    Ego_a = nx.ego_graph(G,a,radius=1,distance='weight',center=0)
    Ego_a_nodes = set(Ego_a.nodes)
    Current_position = float(agents_reg[np.where(a),1])
    for i in agents_reg[n, 2:]:
        k=2
        Ego_i = nx.ego_graph(G,i,radius=1,distance='weight',center=0)
        Ego_i_nodes = set(Ego_i.nodes)
        for j in range(k):
            Ego_current = nx.ego_graph(G,Current_position,radius=1,distance='weight',center=0)
            Ego_current_nodes = set(Ego_current.nodes)
            Intersect = Ego_i_nodes.intersection(Ego_current_nodes)
            Union = Ego_a_nodes.union(Intersect)
            Current_position = random.choice(list(Union))
            a1.extend([a])
            a2.extend([Current_position])
        Current_position = i
    count += 1
    if count%10==0:
        print(count, time.time()-t)

# alternative option
a1 = []
a2 = []
count = 0
t = time.time()
for n in range(len(agents_reg)):
    a = agents_reg[n, 0]
    a_nodes = np.array(list(nx.single_source_dijkstra_path_length(G, a, weight='weight').items()))
    a_nodes = a_nodes[a_nodes[:, 1]<1, 0]
    current_position = agents_reg[n, 1]
    for i in agents_reg[n, 2:]:
        k = 2
        i_nodes = np.array(list(nx.single_source_dijkstra_path_length(G, i, weight='weight').items()))
        i_nodes = i_nodes[i_nodes[:, 1]<1, 0]
        for j in range(k):
            c_nodes = np.array(list(nx.single_source_dijkstra_path_length(G, current_position, weight='weight').items()))
            c_nodes = c_nodes[c_nodes[:, 1]<1, 0]
            intersect = np.intersect1d(i_nodes, c_nodes)
            union = np.union1d(intersect, a_nodes)
            current_position = random.choice(union)
            a1.extend([a])
            a2.extend([Current_position])
        current_position = i
    count += 1
    if count%10==0:
        print(count, time.time()-t)


agents_var = np.column_stack((a1, a2))

for n in range(len(agents_var)):
    a = agents_var[n, 0]
    if agents[np.where(a), 13] == 2:
        exposed_build.append(float(agents_var[np.where(a),1]))
        
from collections import defaultdict
d = defaultdict(list)
for key, val in agents_var:
    d[key].append(val)
    

working_agents_chances = working_agents[:,14:16]
index = np.where(working_agents[:,14] > 0)[0]
working_agents_chances = np.column_stack((index,working_agents_chances))
del index

chances = []
for a in working_agents_chances:
    for b in interaction_prob:
        if working_agents_chances[a,0] == interaction_prob[b,0] and interaction_prob[b,1] != 0:
            chances.append(interaction_prob[a,1]*working_agents_cahnces[b,1]*working_agents_cahnces[b,2])
            

            
    
            