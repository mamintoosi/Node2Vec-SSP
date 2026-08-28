#!/usr/bin/env python3
with open('paper/sn-article.tex', 'r') as f:
    tex = f.read()

replacements = [
    ('the performance of DeepWalk-SSP is relatively insensitive',
     'the performance of the graph embedding method is relatively insensitive'),
    
    ('a particular interaction between DeepWalk-SSP and a specific clustering algorithm',
     'a particular interaction between Node2Vec and a specific clustering algorithm'),
    
    ('DeepWalk-SSP achieves a higher average Silhouette Score than PCA+KMeans',
     'Node2Vec achieves a higher average Silhouette Score than PCA+KMeans'),
    
    ('applied to the DeepWalk-SSP embeddings.',
     'applied to the Node2Vec embeddings.'),
    
    ('DeepWalk-SSP directly produces a two-dimensional representation',
     'Node2Vec directly produces a two-dimensional representation'),
    
    ('the DeepWalk-SSP representation exhibits more compact',
     'the Node2Vec representation exhibits more compact'),
    
    ('obtained by DeepWalk-SSP for this course',
     'obtained by Node2Vec for this course'),
]

for old, new in replacements:
    if old in tex:
        tex = tex.replace(old, new)
        print(f'OK: "{old[:60]}"')
    else:
        print(f'MISS: "{old[:60]}"')

with open('paper/sn-article.tex', 'w') as f:
    f.write(tex)

# Final verification
import subprocess
result = subprocess.run(['grep', '-c', 'DeepWalk-SSP', 'paper/sn-article.tex'], capture_output=True, text=True)
print(f'\nTotal DeepWalk-SSP references (including comments): {result.stdout.strip()}')

result2 = subprocess.run(['grep', '-n', 'DeepWalk-SSP', 'paper/sn-article.tex'], capture_output=True, text=True)
for line in result2.stdout.strip().split('\n'):
    if line and not line.split(':')[1].strip().startswith('%'):
        print(f'  Non-comment: {line}')
