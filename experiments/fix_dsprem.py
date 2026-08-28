#!/usr/bin/env python3
"""Replace remaining DeepWalk-SSP references with Node2Vec."""
with open('paper/sn-article.tex', 'r') as f:
    tex = f.read()

replacements = [
    # Figure caption for embedding dimension
    ('Silhouette scores ($\\uparrow$ higher is better) for DeepWalk-SSP across different embedding dimensions.',
     'Silhouette scores ($\\uparrow$ higher is better) for the graph embedding method across different embedding dimensions.'),
    
    # Sensitivity text
    ('We next examine the sensitivity of DeepWalk-SSP to its principal random-walk and Skip-Gram parameters.',
     'We next examine the sensitivity of the graph embedding method to its principal random-walk and Skip-Gram parameters.'),
    
    # Clustering algorithms section
    ('we evaluate the learned DeepWalk-SSP representations using four clustering methods',
     'we evaluate the learned Node2Vec representations using four clustering methods'),
    
    ('applied to the conventional Student-Course representation and DeepWalk-SSP embeddings.',
     'applied to the conventional Student-Course representation and Node2Vec embeddings.'),
    
    ('DeepWalk-SSP generally produces higher clustering quality',
     'Node2Vec generally produces higher clustering quality'),
    
    # Stability table header
    ('\\textbf{DeepWalk-SSP} , ARI',
     '\\textbf{Node2Vec} , ARI'),
    
    # Stability text
    ('The DeepWalk-SSP representation produces highly stable clusterings',
     'The Node2Vec representation produces highly stable clusterings'),
    
    ('DeepWalk-SSP generally produces more favorable DBI and CH',
     'Node2Vec generally produces more favorable DBI and CH'),
    
    # Visualization
    ('the two-dimensional DeepWalk-SSP representation.',
     'the two-dimensional Node2Vec representation.'),
    
    ('DeepWalk-SSP directly produces a two-dimensional embedding',
     'Node2Vec directly produces a two-dimensional embedding'),
    
    ('\\caption{DeepWalk-SSP embeddings',
     '\\caption{Node2Vec embeddings'),
    
    ('the DeepWalk-SSP points are plotted directly',
     'the Node2Vec points are plotted directly'),
    
    # Runtime
    ('the complete DeepWalk-SSP pipeline.',
     'the complete Node2Vec pipeline.'),
    
    ('the DeepWalk-SSP representation provides substantially',
     'the Node2Vec representation provides substantially'),
    
    ('\\caption{Runtime analysis in seconds for the complete DeepWalk-SSP pipeline',
     '\\caption{Runtime analysis in seconds for the complete Node2Vec pipeline'),
]

for old, new in replacements:
    if old in tex:
        tex = tex.replace(old, new)
        print(f'OK: replaced "{old[:60]}..."')
    else:
        print(f'MISS: "{old[:60]}..."')

with open('paper/sn-article.tex', 'w') as f:
    f.write(tex)

# Final count
lines = tex.split('\n')
count = 0
for i, line in enumerate(lines):
    if 'DeepWalk-SSP' in line and not line.strip().startswith('%'):
        print(f'  REMAINING L{i+1}: {line.strip()[:120]}')
        count += 1
print(f'\nRemaining non-comment DeepWalk-SSP references: {count}')
