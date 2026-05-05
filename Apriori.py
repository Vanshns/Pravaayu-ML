# import pandas as pd
# import numpy as np
# from mlxtend.frequent_patterns import apriori, association_rules

# # 1. Prepare data
# # Loading the pruned data you generated earlier (122 columns)
# binary_df = pd.read_csv('pruned_clinical_data.csv').drop(['age', 'weight'], axis=1)

# # Ensure everything is boolean for the Apriori algorithm
# binary_df = (binary_df > 0).astype(bool)

# # 2. Find frequent itemsets
# # min_support=0.1 means the symptom must appear in at least 10% of your 144 patients
# frequent_itemsets = apriori(binary_df, min_support=0.1, use_colnames=True)

# # 3. Generate Association Rules
# # We use 'lift' to find symptoms that are strongly linked rather than just appearing by chance
# rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)

# # 4. Filter for high-confidence rules
# # We focus on rules that are true at least 80% of the time
# smart_rules = rules[rules['confidence'] > 0.8].sort_values(by='lift', ascending=False)

# # --- NEW CLEANUP & EXPORT SECTION ---

# # 5. Clean up the 'frozenset' formatting for readability
# smart_rules['if_symptom'] = smart_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
# smart_rules['then_suggest'] = smart_rules['consequents'].apply(lambda x: ', '.join(list(x)))

# # 6. Create the final Decision Table
# # We drop the technical frozenset columns and keep the human-readable ones
# final_logic = smart_rules[['if_symptom', 'then_suggest', 'support', 'confidence', 'lift']]

# # 7. Remove duplicate "mirror" rules to keep the output concise
# # Sometimes the math shows "If A then B" and "If B then A" as separate high-lift rules
# final_logic = final_logic.drop_duplicates(subset=['confidence', 'lift'])

# print("\n=== CLEAN CLINICAL DECISION TABLE ===")
# print(final_logic.head(20))

# # 8. Export to CSV for your project report
# final_logic.to_csv('clinical_logic_rules.csv', index=False)
# print("\nSuccess! The clinical logic has been saved to 'clinical_logic_rules.csv'.")

# # High Confidence ($1.0$): These are your "Clinical Certainties.
# # "High Lift ($>5$): These are your "Strong Predictors" (the relationship isn't accidental).
# # Support: Use this to determine how many people in your "clinic" actually fit this profile.


import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules

# --- 1. PREPARE DATA ---
binary_df = pd.read_csv('pruned_clinical_data.csv').drop(['age', 'weight'], axis=1)
binary_df = (binary_df > 0).astype(bool)

# --- 2. FIND FREQUENT ITEMSETS ---
frequent_itemsets = apriori(binary_df, min_support=0.1, use_colnames=True)

# --- 3. GENERATE ASSOCIATION RULES ---
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)

# --- 4. FILTER FOR HIGH-CONFIDENCE RULES ---
smart_rules = rules[rules['confidence'] > 0.8].sort_values(by='lift', ascending=False)

# --- 5. CLEANUP & EXPORT ---
smart_rules['if_symptom'] = smart_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
smart_rules['then_suggest'] = smart_rules['consequents'].apply(lambda x: ', '.join(list(x)))

final_logic = smart_rules[['if_symptom', 'then_suggest', 'support', 'confidence', 'lift']]
final_logic = final_logic.drop_duplicates(subset=['confidence', 'lift'])

print("\n=== CLEAN CLINICAL DECISION TABLE ===")
print(final_logic.head(20))

final_logic.to_csv('clinical_logic_rules.csv', index=False)
print("\nSuccess! The clinical logic has been saved to 'clinical_logic_rules.csv'.")

# --- 6. STEP 1: NETWORK VISUALIZATION ---
def visualize_symptom_network(df, min_lift=8.0):
    """Creates a directed graph showing symptom pathways."""
    G = nx.DiGraph()
    
    # We use a subset of rules for the visual to avoid 'spaghetti' overlap
    # Focus on the strongest relationships
    plot_df = df[df['lift'] >= min_lift].head(30) 

    for _, row in plot_df.iterrows():
        G.add_edge(row['if_symptom'], row['then_suggest'], weight=row['lift'])

    plt.figure(figsize=(16, 10))
    
    # Layout algorithm to position nodes
    pos = nx.spring_layout(G, k=0.8, seed=42)
    
    # Draw Nodes
    nx.draw_networkx_nodes(G, pos, node_size=2500, node_color='skyblue', alpha=0.8)
    
    # Draw Edges (thickness based on lift)
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    # Normalize weights for better visibility
    norm_weights = [((w - min(weights)) / (max(weights) - min(weights)) * 4) + 1 for w in weights]
    
    nx.draw_networkx_edges(G, pos, width=norm_weights, edge_color='gray', 
                           arrowsize=25, arrowstyle='-|>', connectionstyle='arc3,rad=0.1')
    
    # Draw Labels
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')

    plt.title(f"Clinical Symptom Association Map (Lift >= {min_lift})", fontsize=15)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('symptom_network.png')
    plt.show()

# Run the visualization
visualize_symptom_network(final_logic)

# --- STEP 2: RED FLAG ANALYSIS ---
def extract_red_flags(df, lift_threshold=6.0, support_max=0.15):
    """
    Identifies rules that are rare (low support) but highly specific (high lift).
    These often represent specialized clinical 'red flags'.
    """
    red_flags = df[(df['lift'] >= lift_threshold) & (df['support'] <= support_max)]
    
    print(f"\n=== STEP 2: RED FLAG CLINICAL INDICATORS (Lift > {lift_threshold}) ===")
    if red_flags.empty:
        print("No specific red flags found at this threshold. Try lowering the lift_threshold.")
    else:
        print(red_flags[['if_symptom', 'then_suggest', 'lift', 'support']].head(10))
        red_flags.to_csv('clinical_red_flags.csv', index=False)
        print("\nRed flags saved to 'clinical_red_flags.csv'.")
    
    return red_flags

# Execute Step 2
red_flags_df = extract_red_flags(final_logic)

# --- STEP 3: REDUNDANCY PRUNING ---
def prune_redundant_rules(rules_df):
    """
    Removes rules where a simpler version (subset of antecedents) 
    provides the same or better confidence.
    """
    # Sort by length of antecedents (simplest first)
    rules_df['ant_len'] = rules_df['if_symptom'].apply(lambda x: len(x.split(', ')))
    df = rules_df.sort_values(by=['ant_len', 'confidence'], ascending=[True, False])
    
    unique_rules = []
    
    for i, row in df.iterrows():
        is_redundant = False
        current_ant = set(row['if_symptom'].split(', '))
        current_con = set(row['then_suggest'].split(', '))
        
        for _, stable_row in enumerate(unique_rules):
            stable_ant = set(stable_row['if_symptom'].split(', '))
            stable_con = set(stable_row['then_suggest'].split(', '))
            
            # Check if existing simpler rule covers this one
            if stable_ant.issubset(current_ant) and stable_con == current_con:
                if stable_row['confidence'] >= row['confidence']:
                    is_redundant = True
                    break
        
        if not is_redundant:
            unique_rules.append(row)
            
    pruned_df = pd.DataFrame(unique_rules).drop(columns=['ant_len'])
    print(f"\n=== STEP 3: PRUNED LOGIC (Reduced from {len(df)} to {len(pruned_df)} rules) ===")
    print(pruned_df.head(15))
    return pruned_df

# Execute Step 3
final_pruned_logic = prune_redundant_rules(final_logic)
final_pruned_logic.to_csv('final_clinical_logic_pruned.csv', index=False)