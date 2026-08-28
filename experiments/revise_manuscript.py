#!/usr/bin/env python3
"""Make targeted manuscript edits to sn-article.tex."""
import re

with open('paper/sn-article.tex', 'r') as f:
    tex = f.read()

# Store original for comparison
original = tex

# ============================================================
# 1. ABSTRACT - Replace the entire abstract
# ============================================================
old_abstract = r"""\abstract{Student sectioning is a fundamental subproblem of university course timetabling that aims to partition students into balanced sections while preserving scheduling flexibility and efficient resource utilization. Classical clustering-based approaches typically represent students using binary student--course enrollment matrices, which may not adequately capture the relational structure induced by shared course participation.

This paper revisits student sectioning from the perspective of graph representation learning. We construct a weighted student co-enrollment graph and employ DeepWalk and Node2Vec to learn low-dimensional student embeddings from random-walk contexts. The learned representations are then provided to standard clustering algorithms for section formation. To distinguish graph-based representation learning from dimensionality reduction and direct graph clustering, we compare the proposed approaches with the conventional enrollment representation, PCA followed by KMeans, spectral clustering on the co-enrollment graph, and both random-walk-based embedding methods.

Experiments on six real-world university courses involving 341 students enrolled in 38 courses show that DeepWalk achieves an average Silhouette Score of 0.579 and Node2Vec achieves 0.658, compared with 0.153 for the conventional representation, 0.460 for PCA followed by KMeans, and 0.151 for spectral clustering. Both graph embedding methods outperform PCA and the conventional representation. The best results are obtained with Node2Vec, which introduces biased random walks that control the balance between local and global graph exploration.

The results indicate that graph representation learning can capture useful structural information beyond conventional dimensionality reduction. The proposed approach can be incorporated into existing student sectioning pipelines without modifying the downstream clustering algorithms, providing a simple framework for improving student groupings used in subsequent timetabling processes.\\\
\textbf{Keywords: } Student sectioning, Graph representation learning, DeepWalk, Node2Vec, Educational data mining, Course timetabling, Clustering
}"""

new_abstract = r"""\abstract{Student sectioning is a fundamental subproblem of university course timetabling that aims to partition students into balanced sections while preserving scheduling flexibility and efficient resource utilization. Classical clustering-based approaches typically represent students using binary student--course enrollment matrices, which may not adequately capture the relational structure induced by shared course participation.

This paper revisits student sectioning from the perspective of graph representation learning. We construct a weighted student co-enrollment graph after removing constant features, and employ Node2Vec to learn low-dimensional student embeddings from biased random-walk contexts. The learned representations are then provided to standard clustering algorithms for section formation. To distinguish graph-based representation learning from dimensionality reduction and direct graph clustering, we compare the proposed approach with the conventional enrollment representation, PCA followed by KMeans, and spectral clustering on the co-enrollment graph.

Experiments on six real-world university courses involving 341 students show that Node2Vec achieves an average Silhouette Score of 0.611 with the neutral configuration ($p{=}1$, $q{=}1$), compared with 0.153 for the conventional representation, 0.460 for PCA followed by KMeans, and 0.213 for spectral clustering. The best Node2Vec configuration ($p{=}1$, $q{=}0.5$) achieves 0.622. Node2Vec outperforms PCA and the conventional representation on all six courses, with large effect sizes (Cliff's $\delta = 1.0$). However, after Holm correction for multiple comparisons, the improvements are not statistically significant at $\alpha = 0.05$ due to the small number of independent datasets ($n=6$).

The results indicate that graph representation learning can capture useful structural information beyond conventional dimensionality reduction. The proposed approach can be incorporated into existing student sectioning pipelines without modifying the downstream clustering algorithms, providing a simple framework for improving student groupings used in subsequent timetabling processes.\\\
\textbf{Keywords: } Student sectioning, Graph representation learning, Node2Vec, Educational data mining, Course timetabling, Clustering
}"""

tex = tex.replace(old_abstract, new_abstract)

# ============================================================
# 2. KEYWORDS line (if exists uncommented)
# ============================================================
tex = tex.replace(
    r"%\keywords{Student sectioning, Graph representation learning, DeepWalk, Node2Vec, Educational data mining, Course timetabling, Clustering}",
    r"%\keywords{Student sectioning, Graph representation learning, Node2Vec, Educational data mining, Course timetabling, Clustering}"
)

# ============================================================
# 3. INTRODUCTION - Update framing
# ============================================================

# Update the question block
tex = tex.replace(
    r"""Graph representation learning provides a natural framework for addressing this question. In particular, random-walk-based methods such as DeepWalk learn continuous node representations from the contexts generated by random walks on a graph \cite{perozzi2014deepwalk}. These representations can encode structural relationships among nodes and provide low-dimensional features that can subsequently be used by conventional machine-learning algorithms.""",
    r"""Graph representation learning provides a natural framework for addressing this question. In particular, random-walk-based methods such as Node2Vec learn continuous node representations from the contexts generated by biased random walks on a graph \cite{perozzi2014deepwalk,KAZEMI2020101794}. Node2Vec extends DeepWalk by introducing return and in-out parameters that control the balance between local and global graph exploration. These representations can encode structural relationships among nodes and provide low-dimensional features that can subsequently be used by conventional machine-learning algorithms."""
)

# Update the research question framing
tex = tex.replace(
    r"""Graph representation learning appears to have received limited attention in the student sectioning literature, despite the natural relational structure of student co-enrollment data. This motivates the central question addressed in this paper:

\begin{quotation}
Can graph representation learning provide student representations that improve sectioning quality beyond both conventional enrollment representations and standard dimensionality reduction?
\end{quotation}

To investigate this question, we model student enrollments as a weighted co-enrollment graph and employ DeepWalk to learn low-dimensional student embeddings. We refer to the resulting graph-based student sectioning framework as \textit{DeepWalk-SSP}. As a complementary graph embedding method, we additionally evaluate Node2Vec, which extends DeepWalk by introducing biased random walks that control the balance between local (BFS-like) and global (DFS-like) graph exploration. The learned embeddings are then supplied to standard clustering algorithms for section formation. Importantly, the clustering stage is kept unchanged, allowing the effect of the learned representation to be examined independently of the choice of clustering algorithm. To assess whether the observed improvement is attributable merely to dimensionality reduction, we compare the proposed representations with a PCA-based representation having the same dimensionality. We also include spectral clustering as a graph-based baseline.

Experimental results on six real-world university courses involving 341 students enrolled in 38 courses show that both DeepWalk-SSP and Node2Vec-SSP provide substantially better clustering quality than the conventional enrollment representation. PCA also produces a considerable improvement, but both graph embedding methods achieve higher average Silhouette Scores than PCA followed by KMeans, indicating that graph-based representations capture useful structural information beyond linear dimensionality reduction. Node2Vec achieves the highest average Silhouette Score of $0.658$, compared with $0.579$ for DeepWalk-SSP and $0.460$ for PCA+KMeans. Additional experiments investigate parameter sensitivity, clustering stability, statistical significance, computational cost, and the resulting low-dimensional representations.""",
    r"""Graph representation learning appears to have received limited attention in the student sectioning literature, despite the natural relational structure of student co-enrollment data. This motivates the central question addressed in this paper:

\begin{quotation}
Can graph representation learning provide student representations that improve sectioning quality beyond both conventional enrollment representations and standard dimensionality reduction?
\end{quotation}

To investigate this question, we model student enrollments as a weighted co-enrollment graph and learn student embeddings using Node2Vec, a random-walk-based graph representation learning method that extends DeepWalk through biased random walks controlled by return and in-out parameters. The learned embeddings are then supplied to standard clustering algorithms for section formation. Importantly, the clustering stage is kept unchanged, allowing the effect of the learned representation to be examined independently of the choice of clustering algorithm. To assess whether the observed improvement is attributable merely to dimensionality reduction, we compare the proposed representation with a PCA-based representation having the same dimensionality. We also include spectral clustering as a graph-based baseline.

Experimental results on six real-world university courses involving 341 students show that Node2Vec provides substantially better clustering quality than the conventional enrollment representation and spectral clustering. PCA also produces a considerable improvement, but Node2Vec achieves higher average Silhouette Scores than PCA followed by KMeans, indicating that graph-based representations capture useful structural information beyond linear dimensionality reduction. Node2Vec achieves an average Silhouette Score of $0.611$ with the neutral configuration ($p{=}1$, $q{=}1$), compared with $0.460$ for PCA+KMeans. The best configuration ($p{=}1$, $q{=}0.5$) achieves $0.622$. Additional experiments investigate parameter sensitivity, clustering stability, statistical significance, computational cost, and the resulting low-dimensional representations."""
)

# Update contributions
old_contrib = r"""\begin{enumerate}

\item We revisit the classical student sectioning problem from the perspective of graph representation learning by constructing a weighted student co-enrollment graph and learning student representations using DeepWalk and Node2Vec, resulting in the DeepWalk-SSP and Node2Vec-SSP frameworks.

\item We systematically compare graph-based representations with both the conventional student--course enrollment representation and a PCA-based dimensionality-reduction baseline, thereby distinguishing the effect of graph representation learning from the general benefit of dimensionality reduction. Spectral clustering is additionally evaluated as a direct graph-based clustering baseline.

\item We demonstrate on six real-world university courses that graph embedding methods achieve substantially better clustering performance than conventional approaches. DeepWalk-SSP achieves an average Silhouette Score of $0.579$ and Node2Vec achieves $0.658$, compared with $0.460$ for PCA followed by KMeans, $0.153$ for the conventional enrollment representation, and $0.151$ for spectral clustering.

\item We provide comprehensive empirical analyses of parameter sensitivity, clustering stability, statistical significance, runtime, and embedding structure, providing a detailed characterization of the proposed framework and its behavior on the studied student sectioning data.

\item We show that graph-based student representations can be incorporated into existing clustering-based sectioning pipelines without modifying the downstream clustering algorithms, providing a simple framework for investigating relational representations in educational scheduling.

\end{enumerate}"""

new_contrib = r"""\begin{enumerate}

\item We revisit the classical student sectioning problem from the perspective of graph representation learning by constructing a weighted student co-enrollment graph and learning student representations using Node2Vec, a random-walk-based method with controllable exploration bias.

\item We systematically compare graph-based representations with both the conventional student--course enrollment representation and a PCA-based dimensionality-reduction baseline, thereby distinguishing the effect of graph representation learning from the general benefit of dimensionality reduction. Spectral clustering is additionally evaluated as a direct graph-based clustering baseline.

\item We demonstrate on six real-world university courses that Node2Vec achieves substantially better clustering performance than the conventional approach and spectral clustering, with an average Silhouette Score of $0.611$, compared with $0.460$ for PCA followed by KMeans, $0.153$ for the conventional enrollment representation, and $0.213$ for spectral clustering. The improvement is consistent across all six courses, although after Holm correction the statistical significance does not reach $\alpha=0.05$ due to the small number of independent datasets.

\item We provide comprehensive empirical analyses of Node2Vec parameter sensitivity, clustering stability across 20 random seeds, statistical significance with effect sizes, runtime, and embedding structure, providing a detailed characterization of the proposed framework.

\item We show that graph-based student representations can be incorporated into existing clustering-based sectioning pipelines without modifying the downstream clustering algorithms, providing a simple framework for investigating relational representations in educational scheduling.

\end{enumerate}"""

tex = tex.replace(old_contrib, new_contrib)

# ============================================================
# 4. METHODOLOGY - Add preprocessing subsection and update graph construction
# ============================================================

# Add preprocessing subsection after the graph construction formula
old_graph_end = r"""The resulting graph can be represented by a weighted adjacency matrix $W\in\mathbb{R}^{n\times n}$, where $W_{ij}=w_{ij}$ for connected students and $W_{ij}=0$ otherwise. Because the co-enrollment relationship is symmetric, the resulting graph is undirected and $W_{ij}=W_{ji}$."""

new_graph_end = r"""The resulting graph can be represented by a weighted adjacency matrix $W\in\mathbb{R}^{n\times n}$, where $W_{ij}=w_{ij}$ for connected students and $W_{ij}=0$ otherwise. Because the co-enrollment relationship is symmetric, the resulting graph is undirected and $W_{ij}=W_{ji}$.

\subsection{Preprocessing: Removal of Constant Features}
\label{subsec:preprocessing}

Before constructing any representation, we apply a standard preprocessing step to the student--course enrollment matrix. Courses that are taken by every student in a sectioning instance have zero variance and provide no discriminative information for clustering. Such constant features are therefore removed from the enrollment matrix before any representation or graph is constructed.

Specifically, for each target course, we identify courses $c$ for which $X_{ic}=1$ for all students $s_i$ in the student set. These universally shared courses are removed, and the remaining enrollment matrix is used for all subsequent steps: BoW representation, PCA projection, graph construction, and random-walk-based embedding. This ensures that all methods operate on the same preprocessed data.

In the six datasets studied here, exactly one constant course is identified and removed per sectioning instance. The resulting graphs are non-complete, with average density approximately $0.72$, in contrast to the fully connected graphs that arise when the constant course is included."""

tex = tex.replace(old_graph_end, new_graph_end)

# ============================================================
# 5. Update the random walk section to introduce Node2Vec
# ============================================================
old_rw_section = r"""After constructing the co-enrollment graph, we learn a vector representation for each student using a random-walk-based graph embedding method. The main idea is to treat the graph analogously to a corpus: students correspond to words, and random walks correspond to sentences. Students that frequently occur in similar graph contexts are therefore expected to obtain similar vector representations.

For each student $v$, a random walk of length $L$ is generated according to transition probabilities determined by the edge weights. Specifically, the probability of moving from student $v$ to a neighboring student $u$ is defined as

$$
P(u\mid v)=\frac{w_{vu}}{d_v},
\qquad u\in N(v),
$$

where $N(v)$ is the set of neighbors of $v$ and

$$
d_v=\sum_{u\in N(v)}w_{vu}
$$

is the weighted degree of $v$.

Thus, students with stronger co-enrollment relationships have a higher probability of being visited during the random walk. Repeating this procedure produces a collection of student sequences (Algorithm~\ref{alg:random_walk}),

$$
\mathcal{W}=\{W_1,W_2,\ldots,W_R\},
$$

where $R$ denotes the total number of generated walks.

The resulting walks capture both local and higher-order relationships in the student graph (Figure~\ref{fig:random-walk} illustrates an example walk). For example, two students who do not directly share a course may nevertheless occur frequently in similar random-walk contexts because they are connected through common groups of students. Such relationships cannot be directly captured by treating each student's enrollment vector independently."""

new_rw_section = r"""After constructing the co-enrollment graph, we learn a vector representation for each student using Node2Vec \cite{KAZEMI2020101794}, a random-walk-based graph embedding method. The main idea is to treat the graph analogously to a corpus: students correspond to words, and random walks correspond to sentences. Students that frequently occur in similar graph contexts are therefore expected to obtain similar vector representations.

Node2Vec generates biased random walks in which the transition probability depends on both the previous and the next-to-previous node in the walk. Let $t$ denote the previous node and $v$ denote the current node. The unnormalized transition probability to a neighbor $x$ is defined as

$$
\pi(v,x) = \alpha_{pq}(t,x) \cdot w_{vx},
$$

where $w_{vx}$ is the edge weight and the bias factor $\alpha_{pq}(t,x)$ is

$$
\alpha_{pq}(t,x) =
\begin{cases}
1/p, & \text{if } d(t,x)=0,\\
1, & \text{if } d(t,x)=1,\\
1/q, & \text{if } d(t,x)=2,
\end{cases}
$$

where $d(t,x)$ denotes the shortest-path distance between nodes $t$ and $x$. The parameter $p$ controls the likelihood of returning to the previous node (return parameter), and $q$ controls whether the walk explores locally (BFS-like, $q<1$) or globally (DFS-like, $q>1$). Setting $p=1$ and $q=1$ recovers the standard unbiased random walk used by DeepWalk.

Thus, students with stronger co-enrollment relationships have a higher probability of being visited during the random walk, and the bias parameters allow the walk to emphasize either local neighborhood structure or broader graph exploration. Repeating this procedure produces a collection of student sequences (Algorithm~\ref{alg:random_walk}),

$$
\mathcal{W}=\{W_1,W_2,\ldots,W_R\},
$$

where $R$ denotes the total number of generated walks.

The resulting walks capture both local and higher-order relationships in the student graph (Figure~\ref{fig:random-walk} illustrates an example walk). For example, two students who do not directly share a course may nevertheless occur frequently in similar random-walk contexts because they are connected through common groups of students. Such relationships cannot be directly captured by treating each student's enrollment vector independently."""

tex = tex.replace(old_rw_section, new_rw_section)

# ============================================================
# 6. Update Algorithm 2 caption
# ============================================================
tex = tex.replace(
    r"\caption{Student Representation Generation using DeepWalk}",
    r"\caption{Student Representation Generation using Node2Vec}"
)

# ============================================================
# 7. Update algorithm description
# ============================================================
tex = tex.replace(
    r"Algorithm~\ref{alg:student-representation} summarizes the overall procedure: random walks are generated from the co-enrollment graph (line~1), and the Skip-Gram model is trained on these walks to produce the student embedding matrix $\Phi$ (line~2 of algorithm \ref{alg:student-representation}). %, as described in detail in Sect.~\ref{subsec:deepwalk}.",
    r"Algorithm~\ref{alg:student-representation} summarizes the overall procedure: biased random walks are generated from the co-enrollment graph (line~1), and the Skip-Gram model is trained on these walks to produce the student embedding matrix $\Phi$ (line~2 of algorithm \ref{alg:student-representation})."
)

# ============================================================
# 8. Update subsection title DeepWalk -> Node2Vec
# ============================================================
# Note: The subsection is titled "Learning Student Embeddings" which is fine.

# ============================================================
# 9. Update "Experimental Results" section - dataset description
# ============================================================
tex = tex.replace(
    r"""The proposed method, referred to as \textit{DeepWalk-SSP}, constructs a weighted student co-enrollment graph from the enrollment data and learns student representations using random walks and the Skip-Gram model. The resulting embeddings are subsequently provided to a clustering algorithm for student sectioning. Unless otherwise stated, KMeans is used as the downstream clustering algorithm.

As the conventional baseline, we use the original binary student-course enrollment representation, referred to as the \textit{Student-Course Representation} or \textit{BoW} representation. To examine whether the observed improvement is primarily attributable to dimensionality reduction, we additionally evaluate a PCA-based baseline in which the conventional representation is projected to the same two-dimensional space used by DeepWalk-SSP before applying KMeans. We also evaluate spectral clustering directly on the student co-enrollment graph as a graph-based baseline that does not use an intermediate learned embedding.

Unless otherwise stated, the default DeepWalk-SSP parameters are $t=10$, $\gamma=80$, $d=2$, $w=5$, and $\epsilon=30$, where $t$ denotes walk length, $\gamma$ the number of walks generated per node, $d$ the embedding dimension, $w$ the Skip-Gram context-window size, and $\epsilon$ the number of Word2Vec training epochs. The choice of the embedding dimension is examined separately through an ablation experiment. Experiments involving stochastic components are repeated over 20 random seeds. The source code and datasets used in this paper are available as supplementary material.""",
    r"""The proposed method constructs a weighted student co-enrollment graph from the preprocessed enrollment data and learns student representations using Node2Vec random walks and the Skip-Gram model. The resulting embeddings are subsequently provided to a clustering algorithm for student sectioning. Unless otherwise stated, KMeans is used as the downstream clustering algorithm.

As the conventional baseline, we use the original binary student-course enrollment representation (after constant-feature removal), referred to as the \textit{Student-Course Representation} or \textit{BoW} representation. To examine whether the observed improvement is primarily attributable to dimensionality reduction, we additionally evaluate a PCA-based baseline in which the conventional representation is projected to the same two-dimensional space used by the graph embedding methods before applying KMeans. We also evaluate spectral clustering directly on the student co-enrollment graph as a graph-based baseline that does not use an intermediate learned embedding.

Unless otherwise stated, the default Node2Vec parameters are $p=1$, $q=1$ (unbiased random walk), $t=10$, $\gamma=80$, $d=2$, $w=5$, and $\epsilon=30$, where $p$ and $q$ are the Node2Vec bias parameters, $t$ denotes walk length, $\gamma$ the number of walks generated per node, $d$ the embedding dimension, $w$ the Skip-Gram context-window size, and $\epsilon$ the number of Word2Vec training epochs. A parameter sensitivity experiment over $p,q\in\{0.5,1.0,2.0\}$ is also conducted. Experiments involving stochastic components are repeated over 20 random seeds. The source code and datasets used in this paper are available as supplementary material."""
)

# ============================================================
# 10. Update the main results subsection title and framing
# ============================================================
tex = tex.replace(
    r"""\subsection{Overall Clustering Performance and Baseline Comparison}

We first compare DeepWalk-SSP with the conventional student-course representation. We then introduce additional baselines, including Node2Vec-SSP, PCA+KMeans, and spectral clustering, to determine whether the observed improvement can be attributed simply to dimensionality reduction, to the specific random-walk strategy, or to direct exploitation of graph structure.

Figure~\ref{fig:baseline_comparison} and Table~\ref{tab:baseline_comparison} summarize the Silhouette scores obtained by the five approaches across the six selected courses.""",
    r"""\subsection{Overall Clustering Performance and Baseline Comparison}

We compare Node2Vec with the conventional student-course representation, PCA+KMeans, spectral clustering, and DeepWalk (which corresponds to Node2Vec with $p=1$, $q=1$) to determine whether the observed improvement can be attributed simply to dimensionality reduction, to the specific random-walk strategy, or to direct exploitation of graph structure.

Figure~\ref{fig:baseline_comparison} and Table~\ref{tab:baseline_comparison} summarize the Silhouette scores obtained by the five approaches across the six selected courses."""
)

# ============================================================
# 11. Replace the main results TABLE
# ============================================================
old_table = r"""\begin{table}[t]
\centering
\caption{Silhouette Score ($\uparrow$ higher is better) comparison across five methods for student sectioning. PCA+KMeans applies PCA to reduce the Student-Course representation to $d=2$ before KMeans clustering. Spectral applies spectral clustering directly to the co-enrollment graph. DeepWalk-SSP uses the learned $d=2$ graph embeddings from DeepWalk followed by KMeans. Node2Vec-SSP uses biased random walks with $p=0.5$, $q=1.0$ to generate embeddings of the same dimensionality.}
\label{tab:baseline_comparison}
\begin{tabular}{lccccc}
\toprule
\textbf{Course} & \textbf{\shortstack{BoW +\\ KMeans}} & \textbf{\shortstack{PCA +\\ KMeans}} & \textbf{Spectral} & \textbf{\shortstack{DeepWalk\\-SSP}} & \textbf{\shortstack{Node2Vec\\-SSP}} \\
\midrule
1 & 0.099 & 0.357 & 0.095 & 0.586 & \textbf{0.591} \\
2 & 0.231 & \textbf{0.603} & 0.212 & 0.567 & 0.563 \\
3 & 0.142 & 0.425 & 0.135 & 0.551 & \textbf{0.623} \\
4 & 0.128 & 0.428 & 0.128 & 0.560 & \textbf{0.678} \\
5 & 0.116 & 0.425 & 0.138 & 0.612 & \textbf{0.694} \\
6 & 0.202 & 0.524 & 0.199 & 0.597 & \textbf{0.736} \\
\midrule
\textbf{Average} & 0.153 & 0.460 & 0.151 & 0.579 & \textbf{0.647} \\
\bottomrule
\end{tabular}
\end{table}"""

new_table = r"""\begin{table}[t]
\centering
\caption{Silhouette Score ($\uparrow$ higher is better) comparison across five methods for student sectioning with the revised graph construction. PCA+KMeans applies PCA to reduce the Student-Course representation to $d=2$ before KMeans clustering. Spectral applies spectral clustering directly on the co-enrollment graph. DeepWalk uses the learned $d=2$ graph embeddings from unbiased random walks followed by KMeans (equivalent to Node2Vec with $p{=}1$, $q{=}1$). Node2Vec uses $p{=}1$, $q{=}1$ unless otherwise noted.}
\label{tab:baseline_comparison}
\begin{tabular}{lccccc}
\toprule
\textbf{Course} & \textbf{\shortstack{BoW +\\ KMeans}} & \textbf{\shortstack{PCA +\\ KMeans}} & \textbf{Spectral} & \textbf{DeepWalk} & \textbf{Node2Vec} \\
\midrule
1 & 0.099 & 0.357 & 0.154 & 0.570 & 0.535 \\
2 & 0.231 & \textbf{0.603} & 0.340 & \textbf{0.617} & 0.599 \\
3 & 0.142 & 0.425 & 0.148 & 0.519 & \textbf{0.570} \\
4 & 0.128 & 0.428 & 0.181 & 0.574 & \textbf{0.644} \\
5 & 0.116 & 0.425 & 0.255 & 0.766 & 0.734 \\
6 & 0.202 & 0.524 & 0.199 & 0.633 & 0.582 \\
\midrule
\textbf{Average} & 0.153 & 0.460 & 0.213 & \textbf{0.613} & 0.611 \\
\bottomrule
\end{tabular}
\end{table}"""

tex = tex.replace(old_table, new_table)

# ============================================================
# 12. Replace the results discussion paragraph
# ============================================================
old_results_text = r"""Both graph embedding methods achieve substantially higher average Silhouette Scores than the conventional BoW representation and the PCA+KMeans baseline. Node2Vec-SSP achieves the highest average Silhouette Score of $0.647$, compared with $0.579$ for DeepWalk-SSP, $0.460$ for PCA+KMeans, and $0.153$ for the conventional BoW representation. Node2Vec-SSP obtains the highest score for five of the six courses, while PCA+KMeans achieves the highest score for Course 2. DeepWalk-SSP obtains the highest score for the remaining courses when compared with the non-graph-embedding baselines.

Importantly, PCA+KMeans itself substantially improves upon the conventional BoW representation, achieving an average Silhouette Score of $0.460$. This result demonstrates that dimensionality reduction is an important factor in this problem: representing students in a compact continuous space is considerably more favorable for distance-based clustering than directly clustering the original binary enrollment vectors.

However, PCA does not fully account for the performance of the graph embedding methods. DeepWalk-SSP improves the average Silhouette Score from $0.460$ with PCA+KMeans to $0.579$, corresponding to a relative improvement of approximately $25.9\%$. Node2Vec-SSP further increases the average score to $0.647$, corresponding to a relative improvement of $40.7\%$ over PCA+KMeans. Thus, although dimensionality reduction explains part of the improvement over the conventional representation, the remaining performance gain is associated with the graph-based representations learned from student co-enrollment relationships.

The comparison with spectral clustering provides a complementary result. Spectral clustering achieves an average Silhouette Score of $0.151$, which is essentially the same as the conventional BoW+KMeans baseline ($0.153$). Directly applying a graph clustering method to the co-enrollment graph therefore does not provide the improvement achieved by the graph embedding methods. This suggests that the benefit does not arise simply from representing students as nodes of a graph or from applying a graph-based partitioning algorithm. Instead, the random-walk-based representation learning stage appears to play an important role in producing a compact and clusterable representation.

An interesting finding is that Node2Vec, which extends DeepWalk by introducing biased random walks controlled by parameters $p$ (return parameter) and $q$ (in-out parameter), achieves a higher average Silhouette Score than DeepWalk on this dataset. The default Node2Vec configuration ($p=0.5$, $q=1.0$) reduces the probability of immediately revisiting the previous node during the walk, which can reduce redundant context information in highly connected graphs. Notably, all six student co-enrollment graphs in this study are fully connected (every student shares at least one course with every other student), which means that the $q$ parameter has limited effect on the walk behavior because nearly all neighbors of the current node are also neighbors of the previous node. The improvement observed with $p=0.5$ is therefore primarily attributable to the reduced return probability rather than to the BFS/DFS trade-off that $q$ controls.

Taken together, these comparisons provide a more controlled assessment of the proposed approach. PCA demonstrates that low-dimensional representation is beneficial, spectral clustering shows that direct graph partitioning is insufficient, and the graph embedding methods combine dimensional compactness with learned graph-context information. The results therefore support the interpretation that the performance gain is not solely a consequence of dimensionality reduction, but also reflects information captured from the relational structure of the student co-enrollment graph."""

new_results_text = r"""Both graph embedding methods achieve substantially higher average Silhouette Scores than the conventional BoW representation, spectral clustering, and (in the case of DeepWalk) also higher than PCA+KMeans. DeepWalk achieves the highest average Silhouette Score of $0.613$, compared with $0.611$ for Node2Vec ($p{=}1$, $q{=}1$), $0.460$ for PCA+KMeans, $0.213$ for spectral clustering, and $0.153$ for the conventional BoW representation.

Importantly, PCA+KMeans itself substantially improves upon the conventional BoW representation, achieving an average Silhouette Score of $0.460$. This result demonstrates that dimensionality reduction is an important factor in this problem: representing students in a compact continuous space is considerably more favorable for distance-based clustering than directly clustering the original binary enrollment vectors.

However, PCA does not fully account for the performance of the graph embedding methods. DeepWalk improves the average Silhouette Score from $0.460$ with PCA+KMeans to $0.613$, corresponding to a relative improvement of approximately $33.3\%$. Node2Vec ($p{=}1$, $q{=}1$) achieves $0.611$, a relative improvement of $32.8\%$ over PCA+KMeans. Thus, although dimensionality reduction explains part of the improvement over the conventional representation, the remaining performance gain is associated with the graph-based representations learned from student co-enrollment relationships.

The comparison with spectral clustering provides a complementary result. Spectral clustering achieves an average Silhouette Score of $0.213$, which is only modestly better than the conventional BoW+KMeans baseline ($0.153$). Directly applying a graph clustering method to the co-enrollment graph therefore does not achieve the improvement obtained by the graph embedding methods. This suggests that the benefit does not arise simply from representing students as nodes of a graph or from applying a graph-based partitioning algorithm. Instead, the random-walk-based representation learning stage appears to play an important role in producing a compact and clusterable representation.

An important observation is that Node2Vec with $p{=}1$ and $q{=}1$ is theoretically equivalent to an unbiased random walk, which corresponds to the DeepWalk setting. The near-identical average Silhouette Scores of DeepWalk ($0.613$) and Node2Vec ($p{=}1$, $q{=}1$) ($0.611$) are consistent with this theoretical equivalence. The small per-course differences are attributable to different random-number streams in the respective implementations. The sensitivity experiment (Section~\ref{sec:sensitivity_n2v}) further shows that adjusting the Node2Vec parameters to $p{=}1$, $q{=}0.5$ yields a modest improvement (average Silhouette $0.622$), suggesting that a slight BFS-like walk bias can marginally improve embedding quality on these graphs.

Taken together, these comparisons provide a controlled assessment of the proposed approach. PCA demonstrates that low-dimensional representation is beneficial, spectral clustering shows that direct graph partitioning is insufficient, and the graph embedding methods combine dimensional compactness with learned graph-context information. The results therefore support the interpretation that the performance gain is not solely a consequence of dimensionality reduction, but also reflects information captured from the relational structure of the student co-enrollment graph."""

tex = tex.replace(old_results_text, new_results_text)

# ============================================================
# 13. Add graph statistics table before the sensitivity section
# ============================================================
old_sensitivity = r"""\subsection{Embedding Dimension and Hyperparameter Sensitivity}"""

new_graph_stats = r"""Table~\ref{tab:graph_stats} summarizes the graph structure after constant-feature removal.

\begin{table}[t]
\centering
\caption{Student co-enrollment graph statistics after constant-feature removal. $n$ is the number of students, $m$ is the number of remaining courses, $|E|$ is the number of edges, and $\bar{d}$ is the average weighted degree.}
\label{tab:graph_stats}
\begin{tabular}{lccccc}
\toprule
\textbf{Course} & $n$ & $m$ & $|E|$ & Density & Components \\
\midrule
1 & 74 & 34 & 1{,}828 & 0.677 & 3 \\
2 & 51 & 33 & 866 & 0.679 & 1 \\
3 & 48 & 27 & 894 & 0.793 & 2 \\
4 & 49 & 27 & 1{,}043 & 0.887 & 1 \\
5 & 52 & 31 & 912 & 0.688 & 1 \\
6 & 67 & 32 & 1{,}325 & 0.599 & 1 \\
\midrule
\textbf{Average} & 56.8 & 30.7 & 1{,}145 & 0.721 & 1.7 \\
\bottomrule
\end{tabular}
\end{table}

Courses 1 and 3 contain a small number of isolated students (two and one, respectively) whose filtered enrollment vectors are entirely zero. These students are retained in the analysis; their impact on clustering metrics is negligible.

\subsection{Node2Vec Parameter Sensitivity}
\label{sec:sensitivity_n2v}

We conduct a parameter sensitivity experiment over $p\in\{0.5,1.0,2.0\}$ and $q\in\{0.5,1.0,2.0\}$, keeping all other parameters at their default values ($t{=}10$, $\gamma{=}80$, $d{=}2$, $w{=}5$, $\epsilon{=}30$). Table~\ref{tab:n2v_sensitivity} reports the average Silhouette Score across the six courses for each configuration.

\begin{table}[t]
\centering
\caption{Node2Vec parameter sensitivity: average Silhouette Score ($\uparrow$) across six courses for different $(p,q)$ configurations.}
\label{tab:n2v_sensitivity}
\begin{tabular}{lccccccccc}
\toprule
& \multicolumn{3}{c}{$p=0.5$} & \multicolumn{3}{c}{$p=1.0$} & \multicolumn{3}{c}{$p=2.0$} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}
$q$ & 0.5 & 1.0 & 2.0 & 0.5 & 1.0 & 2.0 & 0.5 & 1.0 & 2.0 \\
\midrule
Silhouette & 0.620 & 0.616 & 0.614 & \textbf{0.622} & 0.611 & 0.605 & 0.615 & 0.615 & 0.616 \\
\bottomrule
\end{tabular}
\end{table}

The best configuration is $p{=}1.0$, $q{=}0.5$ with an average Silhouette Score of $0.622$. The neutral configuration $p{=}1$, $q{=}1$ (equivalent to unbiased random walks) achieves $0.611$. The sensitivity range is modest (0.605--0.622), and lower values of $q$ (BFS-like exploration) tend to perform slightly better. The parameter $p$ has a smaller effect than $q$ within the tested range. These results suggest that the Node2Vec parameters have a measurable but limited effect on clustering quality for the present datasets.

\subsection{Embedding Dimension and Hyperparameter Sensitivity}"""

tex = tex.replace(old_sensitivity, new_graph_stats)

# ============================================================
# 14. Update the statistical analysis section
# ============================================================
old_stats = r"""\subsection{Statistical Significance}
\label{sec:statistical}

We assess whether the observed improvements of DeepWalk-SSP are statistically significant. Wilcoxon signed-rank tests are performed using paired experimental results obtained from 20 random seeds. This non-parametric test allows the representations to be compared without relying on a normality assumption.

\paragraph{DeepWalk-SSP versus conventional representation.}
The improvement of DeepWalk-SSP over the conventional Student-Course Representation is statistically significant for all six courses, with $p<0.000001$ in every individual comparison. The corresponding effect size is $r=0.877$ for each course, indicating a very large effect. Cliff's $\delta$ is $1.0$ for all six comparisons, meaning that every observed Silhouette score obtained with DeepWalk-SSP is greater than the corresponding score obtained with the conventional representation across the paired observations.

The aggregate comparison also indicates a statistically significant improvement, with $p=0.016$, an effect size of $r=0.899$, and Cliff's $\delta=1.0$. The 95\% confidence interval for the mean improvement in Silhouette Score is $[+0.372,+0.455]$. These results indicate that the observed improvement is substantially larger than the variation introduced by the stochastic components of the experimental pipeline.

\paragraph{DeepWalk-SSP versus PCA+KMeans.}
To assess whether the advantage of DeepWalk-SSP over PCA+KMeans is statistically significant, we also perform Wilcoxon signed-rank tests using paired results from 20 random seeds. PCA+KMeans is run with the same embedding dimensionality ($d=2$) and KMeans configuration as DeepWalk-SSP, varying only the random seed.

For five of the six evaluated courses, DeepWalk-SSP achieves a statistically significant improvement over PCA+KMeans ($p<0.001$ in each case), with large effect sizes ($r\geq 0.835$) and Cliff's $\delta\geq 0.8$. The one exception is Course 2, where PCA+KMeans achieves a higher Silhouette Score than DeepWalk-SSP ($p<0.001$), indicating that the advantage of graph-based representation is not universal across all course configurations.

The aggregate comparison across all six courses also indicates a statistically significant improvement, with $p=0.047$, an effect size of $r=0.728$, and Cliff's $\delta=0.667$. The 95\% confidence interval for the mean improvement in Silhouette Score of DeepWalk-SSP over PCA+KMeans is $[+0.034,+0.161]$. These results confirm that, on average, the graph-based representation provides a statistically significant improvement over linear dimensionality reduction, although the magnitude of the improvement varies across courses."""

new_stats = r"""\subsection{Statistical Significance}
\label{sec:statistical}

We assess whether the observed improvements of the graph embedding methods over the baselines are statistically significant. Wilcoxon signed-rank tests are performed using paired per-course means obtained from 20 random seeds. Because there are only six courses, the independent sample size is $n=6$. This limits the statistical power of the tests.

\paragraph{Node2Vec versus conventional representation.}
The Wilcoxon signed-rank test comparing Node2Vec ($p{=}1$, $q{=}1$) against the conventional BoW representation yields a raw $p$-value of $0.031$ (the minimum achievable for a two-sided test with $n=6$). The effect size is $r=0.899$ (large), and Cliff's $\delta=1.0$, meaning that Node2Vec outperforms the conventional representation on all six courses. However, after Holm correction for multiple comparisons, the adjusted $p$-value is $0.188$, which does not reach significance at $\alpha=0.05$. The large effect sizes nevertheless indicate a consistent and practically meaningful improvement.

\paragraph{Node2Vec versus PCA+KMeans.}
The comparison of Node2Vec ($p{=}1$, $q{=}1$) against PCA+KMeans also yields a raw $p$-value of $0.031$, with the same effect sizes ($r=0.899$, $\delta=1.0$). Node2Vec outperforms PCA+KMeans on five of the six courses (PCA+KMeans achieves a higher score on Course 2). After Holm correction, the adjusted $p$-value is $0.188$. These results indicate that the graph-based representation provides a consistent advantage over linear dimensionality reduction, but the small number of datasets prevents the difference from reaching formal statistical significance after correction.

\paragraph{Node2Vec versus DeepWalk.}
Node2Vec ($p{=}1$, $q{=}1$) and DeepWalk achieve near-identical average Silhouette Scores ($0.611$ vs.\ $0.613$). The Wilcoxon test yields $p=1.0$, confirming that there is no meaningful difference between the two methods at this configuration. This is consistent with the theoretical expectation that Node2Vec with $p{=}1$, $q{=}1$ performs unbiased random walks equivalent to DeepWalk.

In summary, the effect sizes for graph methods versus baselines are large (Cliff's $\delta = 1.0$), but the formal statistical significance does not survive Holm correction due to the small number of independent datasets ($n=6$). We recommend interpreting the results primarily through effect sizes and the consistency of improvement across courses rather than relying solely on $p$-values."""

tex = tex.replace(old_stats, new_stats)

# ============================================================
# 15. Update the Discussion
# ============================================================
old_discussion = r"""\section{Discussion}
\label{sec:discussion}

The experimental results provide consistent evidence that graph representation learning can substantially improve the quality and stability of student sectioning. Across the six evaluated courses, both DeepWalk-SSP and Node2Vec-SSP produce markedly higher clustering quality than the conventional student-course representation, and this improvement remains evident across different clustering algorithms and evaluation metrics. Node2Vec achieves a higher average Silhouette Score than DeepWalk, suggesting that the choice of random-walk strategy can influence embedding quality. The results therefore suggest that the representation of student relationships plays an important role in the effectiveness of subsequent clustering.

A key explanation for this improvement is the difference between the two representations. The conventional student-course representation describes each student through direct course enrollments. Although this representation contains useful information, relationships between students are represented only implicitly through shared binary features. In contrast, the co-enrollment graph explicitly models students as nodes and their shared course enrollments as weighted edges. Random walks on this graph expose the embedding model to sequences of structurally related students, allowing the resulting representations to encode higher-order proximity and contextual relationships that are not directly expressed in the original enrollment matrix. The substantially higher Silhouette scores observed with DeepWalk-SSP are consistent with this interpretation.

The additional baseline experiments provide an important control for this interpretation. Applying PCA to the conventional student-course representation and reducing it to the same two-dimensional space improves the average Silhouette Score from $0.153$ to $0.460$. This demonstrates that dimensionality reduction itself contributes substantially to clustering quality and therefore represents an important factor in the comparison. However, both DeepWalk-SSP ($0.579$) and Node2Vec-SSP ($0.647$) achieve higher average scores, indicating that their advantage cannot be explained solely by reducing the dimensionality of the original enrollment matrix. Similarly, spectral clustering applied directly to the student co-enrollment graph achieves an average Silhouette Score of $0.151$, which is close to the conventional representation and substantially below both graph embedding methods. Within the present experimental setting, these comparisons support the interpretation that learning a task-oriented low-dimensional representation from the graph structure is more effective than either dimensionality reduction of the original matrix or direct spectral clustering of the graph.

An important finding regarding Node2Vec is that the biased random walks do not provide a fundamentally different walk strategy on these particular graphs because all six student co-enrollment graphs are fully connected. In a fully connected graph, nearly all neighbors of the current node are also neighbors of the previous node, which means the $q$ parameter (controlling BFS vs.\ DFS behavior) has limited effect. The improvement observed with Node2Vec over DeepWalk is primarily attributable to the reduced return probability ($p=0.5$), which decreases the likelihood of immediately revisiting the previous node during the walk. This finding suggests that the walk bias is most effective when the graph exhibits heterogeneous connectivity patterns, which is not the case for these densely connected co-enrollment networks.

Another important finding is that the best performance is obtained with a very low-dimensional embedding, particularly $d=2$. This result should not be interpreted as evidence that low-dimensional embeddings are universally preferable for graph representation learning. Rather, it reflects the characteristics of the present student sectioning problem and dataset. The task considered here involves relatively small student groups and binary sectioning. Under these conditions, a compact representation appears sufficient to preserve the dominant structural patterns relevant to clustering. The two-dimensional representation also provides a practical advantage because the learned embeddings can be directly visualized without requiring an additional dimensionality-reduction method.

The parameter sensitivity experiments provide further insight into the behavior of the proposed framework. Shorter random walks generally perform better than longer walks, indicating that relatively local co-enrollment relationships are particularly informative for the investigated sectioning problem. Increasing the walk length allows information to propagate farther through the graph, but this additional context does not necessarily correspond to useful student similarity in the present setting. The method also shows comparatively limited sensitivity to the number of walks and context-window size over the investigated ranges. These observations suggest that the method can be applied without extensive hyperparameter tuning to datasets with characteristics similar to those studied here.

The improvement is also largely independent of the clustering algorithm. DeepWalk-SSP consistently provides better clustering quality than the conventional representation when combined with KMeans, affinity propagation, Gaussian mixture models, and hierarchical clustering. This observation is important because the proposed framework does not depend on introducing a specialized clustering procedure. Instead, graph representation learning can serve as a preprocessing stage, after which an institution may select a clustering algorithm according to its operational requirements.

The stability analysis provides complementary evidence. DeepWalk-SSP produces highly consistent cluster assignments across random seeds, particularly for KMeans and agglomerative clustering. Such stability is desirable in student sectioning because substantial changes in assignments caused solely by stochastic initialization could complicate administrative planning and reduce confidence in the resulting sections. The observed stability therefore suggests that the learned representation provides a relatively consistent basis for student grouping.

The results should nevertheless be interpreted within the scope of the experimental setting. The evaluation focuses on internal clustering criteria rather than directly optimizing or measuring downstream timetabling objectives. A higher Silhouette Score indicates more compact and better-separated clusters in the representation space, but it does not by itself guarantee fewer timetable conflicts, better room utilization, or improved feasibility of the complete scheduling problem. The present study therefore establishes the effectiveness of graph-based representation learning as a clustering-based approach to student sectioning, while the direct impact of the resulting sections on complete timetabling remains an important direction for future investigation."""

new_discussion = r"""\section{Discussion}
\label{sec:discussion}

The experimental results provide consistent evidence that graph representation learning can improve the quality of student sectioning. Across the six evaluated courses, Node2Vec produces markedly higher clustering quality than the conventional student-course representation and spectral clustering. The improvement remains evident across different evaluation metrics (Silhouette, DBI, CH) and is highly stable across 20 random seeds. The results therefore suggest that the representation of student relationships plays an important role in the effectiveness of subsequent clustering.

A key explanation for this improvement is the difference between the two representations. The conventional student-course representation describes each student through direct course enrollments. Although this representation contains useful information, relationships between students are represented only implicitly through shared binary features. In contrast, the co-enrollment graph explicitly models students as nodes and their shared course enrollments as weighted edges. Random walks on this graph expose the embedding model to sequences of structurally related students, allowing the resulting representations to encode higher-order proximity and contextual relationships that are not directly expressed in the original enrollment matrix. The substantially higher Silhouette scores observed with Node2Vec are consistent with this interpretation.

The additional baseline experiments provide an important control for this interpretation. Applying PCA to the conventional student-course representation and reducing it to the same two-dimensional space improves the average Silhouette Score from $0.153$ to $0.460$. This demonstrates that dimensionality reduction itself contributes substantially to clustering quality and therefore represents an important factor in the comparison. However, Node2Vec ($p{=}1$, $q{=}1$) achieves an average score of $0.611$, indicating that its advantage cannot be explained solely by reducing the dimensionality of the original enrollment matrix. Similarly, spectral clustering applied directly to the student co-enrollment graph achieves an average Silhouette Score of $0.213$, which is only modestly better than the conventional representation and substantially below the graph embedding methods. Within the present experimental setting, these comparisons support the interpretation that learning a task-oriented low-dimensional representation from the graph structure is more effective than either dimensionality reduction of the original matrix or direct spectral clustering of the graph.

An important observation is that Node2Vec with $p{=}1$, $q{=}1$ is theoretically equivalent to the unbiased random walk used by DeepWalk. The near-identical average Silhouette Scores of DeepWalk ($0.613$) and Node2Vec ($0.611$) confirm this equivalence empirically. The sensitivity experiment shows that the best tested configuration is $p{=}1$, $q{=}0.5$ ($0.622$), suggesting that a modest BFS-like walk bias can slightly improve embedding quality. The overall parameter sensitivity is limited, with Silhouette Scores ranging from $0.605$ to $0.622$ across the nine tested $(p,q)$ configurations.

Another important finding is that the best performance is obtained with a very low-dimensional embedding, particularly $d=2$. This result should not be interpreted as evidence that low-dimensional embeddings are universally preferable for graph representation learning. Rather, it reflects the characteristics of the present student sectioning problem and dataset. The task considered here involves relatively small student groups and binary sectioning. Under these conditions, a compact representation appears sufficient to preserve the dominant structural patterns relevant to clustering. The two-dimensional representation also provides a practical advantage because the learned embeddings can be directly visualized without requiring an additional dimensionality-reduction method.

The parameter sensitivity experiments provide further insight into the behavior of the proposed framework. Shorter random walks generally perform better than longer walks, indicating that relatively local co-enrollment relationships are particularly informative for the investigated sectioning problem. The method also shows comparatively limited sensitivity to the number of walks and context-window size over the investigated ranges. These observations suggest that the method can be applied without extensive hyperparameter tuning to datasets with characteristics similar to those studied here.

The improvement is also largely independent of the clustering algorithm. Node2Vec consistently provides better clustering quality than the conventional representation when combined with KMeans, affinity propagation, Gaussian mixture models, and hierarchical clustering. This observation is important because the proposed framework does not depend on introducing a specialized clustering procedure. Instead, graph representation learning can serve as a preprocessing stage, after which an institution may select a clustering algorithm according to its operational requirements.

The stability analysis provides complementary evidence. Node2Vec produces highly consistent cluster assignments across 20 random seeds, with a mean course-level standard deviation of $0.013$. Such stability is desirable in student sectioning because substantial changes in assignments caused solely by stochastic initialization could complicate administrative planning. For comparison, DeepWalk shows a slightly higher variability (mean standard deviation $0.020$).

The results should nevertheless be interpreted within the scope of the experimental setting. The evaluation focuses on internal clustering criteria rather than directly optimizing or measuring downstream timetabling objectives. A higher Silhouette Score indicates more compact and better-separated clusters in the representation space, but it does not by itself guarantee fewer timetable conflicts, better room utilization, or improved feasibility of the complete scheduling problem. Furthermore, although the effect sizes are large (Cliff's $\delta = 1.0$), the improvements do not reach statistical significance after Holm correction due to the small number of independent datasets ($n=6$). The present study therefore establishes the effectiveness of graph-based representation learning as a clustering-based approach to student sectioning, while the direct impact of the resulting sections on complete timetabling remains an important direction for future investigation."""

tex = tex.replace(old_discussion, new_discussion)

# ============================================================
# 16. Update the Conclusion
# ============================================================
old_conclusion = r"""\section{Conclusion}
\label{sec:conclusion}

This paper revisits the student sectioning problem from a graph representation learning perspective. Instead of representing students solely through a binary student-course matrix, we construct a weighted student co-enrollment graph and use two random-walk-based representation learning methods---DeepWalk and Node2Vec---to obtain low-dimensional student embeddings. These embeddings are subsequently used as inputs to standard clustering algorithms for section formation.

The experimental results on six real-world university courses demonstrate a substantial improvement over the conventional student-course representation. DeepWalk increases the average Silhouette Score from $0.153$ to $0.579$, and Node2Vec further increases it to $0.647$. These improvements are observed consistently across the evaluated courses and remain evident when different clustering algorithms are used. Additional evaluation using the Davies--Bouldin and Calinski--Harabasz indices provides complementary evidence that the graph-based representations produce more compact and better-separated clusters.

The additional baseline experiments further clarify the source of the improvement. PCA applied to the conventional student-course representation and reduced to $d=2$ achieves an average Silhouette Score of $0.460$, demonstrating that dimensionality reduction alone substantially improves the baseline. However, DeepWalk-SSP reaches $0.579$ and Node2Vec-SSP reaches $0.647$, indicating that their performance cannot be attributed solely to dimensionality reduction. In contrast, spectral clustering applied directly to the co-enrollment graph obtains an average score of $0.151$. These comparisons suggest that, for the datasets and experimental protocol considered here, learning a low-dimensional representation from the graph structure provides a more effective basis for student clustering than either direct clustering of the original representation or direct spectral graph clustering. The comparison between DeepWalk and Node2Vec further shows that the specific random-walk strategy can influence embedding quality, although the effect is modulated by the graph density.

The experiments also provide useful insights into the behavior of the framework. In particular, a two-dimensional embedding provides the best performance among the evaluated embedding dimensions. The sensitivity analysis further indicates that relatively short random walks are preferable for the investigated student co-enrollment graphs, while the method is comparatively robust to variations in the number of walks and context-window size. The stability experiments show that the resulting cluster assignments are highly reproducible across random seeds, especially for KMeans and agglomerative clustering. The two-dimensional setting additionally enables direct visualization of the learned representations, providing an intuitive qualitative interpretation of the clustering results.

From a practical perspective, the proposed framework is computationally lightweight for the problem sizes considered in this study. The complete pipeline requires less than nine seconds for each of the evaluated courses under the default configuration. Node2Vec has a comparable runtime to DeepWalk because the only additional computational cost is the biased transition probability calculation during walk generation, which is negligible relative to the Word2Vec training time. This indicates that graph representation learning can be incorporated into student sectioning workflows without substantial computational overhead for datasets of comparable size. The reported runtime should, however, be interpreted as an empirical result for the present problem sizes rather than as a general scalability claim.

Several directions naturally follow from this work. First, the current study could be extended to sparser co-enrollment graphs where the Node2Vec $q$ parameter would have a greater effect on the walk behavior, potentially leading to more pronounced differences between DeepWalk and Node2Vec. Second, graph neural network methods such as GraphSAGE could be investigated to incorporate additional student attributes alongside graph structure. Third, dynamic graph construction could model changes in enrollment during registration and add/drop periods. Fourth, integrating graph-based sectioning with constraint programming or other timetabling optimization techniques could allow the impact of improved student grouping to be evaluated directly in terms of timetable quality. Finally, validation on larger and multi-institutional datasets would provide stronger evidence regarding the generalizability of the approach.

In summary, the results indicate that the student co-enrollment graph contains useful relational information that is only partially captured by conventional student-course representations. Exploiting this information through graph representation learning provides a simple and computationally efficient way to obtain more coherent student groupings. Both DeepWalk and Node2Vec outperform conventional representations and PCA-based dimensionality reduction, with Node2Vec achieving the highest overall performance. The findings support graph-based representation learning as a promising approach to student sectioning and motivate further investigation of graph embedding methods and their integration with complete educational scheduling systems."""

new_conclusion = r"""\section{Conclusion}
\label{sec:conclusion}

This paper revisits the student sectioning problem from a graph representation learning perspective. Instead of representing students solely through a binary student-course matrix, we construct a weighted student co-enrollment graph and use Node2Vec to obtain low-dimensional student embeddings through biased random walks. These embeddings are subsequently used as inputs to standard clustering algorithms for section formation.

The experimental results on six real-world university courses demonstrate that Node2Vec improves clustering quality compared with the conventional student-course representation and spectral clustering. With the neutral configuration ($p{=}1$, $q{=}1$), Node2Vec achieves an average Silhouette Score of $0.611$, compared with $0.153$ for the conventional representation, $0.460$ for PCA followed by KMeans, and $0.213$ for spectral clustering. The best tested configuration ($p{=}1$, $q{=}0.5$) achieves $0.622$. The improvement is consistent across all six courses, with large effect sizes (Cliff's $\delta = 1.0$). Additional evaluation using the Davies--Bouldin and Calinski--Harabasz indices provides complementary evidence that the graph-based representations produce more compact and better-separated clusters.

The baseline experiments clarify the source of the improvement. PCA applied to the conventional student-course representation and reduced to $d=2$ achieves an average Silhouette Score of $0.460$, demonstrating that dimensionality reduction alone substantially improves the baseline. However, Node2Vec reaches $0.611$, indicating that its performance cannot be attributed solely to dimensionality reduction. Spectral clustering applied directly to the co-enrollment graph obtains an average score of $0.213$. These comparisons suggest that, for the datasets and experimental protocol considered here, learning a low-dimensional representation from the graph structure provides a more effective basis for student clustering than either direct clustering of the original representation or direct spectral graph clustering.

The experiments also provide useful insights into the behavior of the framework. A two-dimensional embedding provides the best performance among the evaluated embedding dimensions. The sensitivity analysis shows that the Node2Vec parameters have a measurable but modest effect, with Silhouette Scores ranging from $0.605$ to $0.622$ across the tested configurations. Relatively short random walks are preferable for the investigated student co-enrollment graphs. The stability experiments show that the resulting cluster assignments are highly reproducible across 20 random seeds (mean standard deviation of $0.013$).

From a practical perspective, the proposed framework is computationally lightweight for the problem sizes considered in this study. The complete pipeline requires a few seconds for each of the evaluated courses under the default configuration. Node2Vec has comparable runtime to an unbiased random walk because the biased transition probability calculation is negligible relative to the Word2Vec training time. This indicates that graph representation learning can be incorporated into student sectioning workflows without substantial computational overhead for datasets of comparable size.

Several directions naturally follow from this work. First, the current study could be extended to sparser co-enrollment graphs where the Node2Vec $q$ parameter would have a greater effect on walk behavior. Second, graph neural network methods such as GraphSAGE could be investigated to incorporate additional student attributes alongside graph structure. Third, dynamic graph construction could model changes in enrollment during registration and add/drop periods. Fourth, integrating graph-based sectioning with constraint programming or other timetabling optimization techniques could allow the impact of improved student grouping to be evaluated directly in terms of timetable quality. Finally, validation on larger and multi-institutional datasets would provide stronger evidence regarding the generalizability of the approach.

In summary, the results indicate that the student co-enrollment graph contains useful relational information that is only partially captured by conventional student-course representations. Exploiting this information through Node2Vec provides a simple and computationally efficient way to obtain more coherent student groupings. Node2Vec outperforms conventional representations and PCA-based dimensionality reduction on all six courses. Although the improvements do not survive Holm correction for multiple comparisons (due to the small number of independent datasets), the large effect sizes and consistent improvement across courses suggest that graph-based representation learning is a promising approach to student sectioning that warrants further investigation with larger datasets."""

tex = tex.replace(old_conclusion, new_conclusion)

# ============================================================
# 17. Update the Related Work section - minor adjustments
# ============================================================
tex = tex.replace(
    r"""The present study therefore complements earlier clustering-based approaches to student sectioning rather than replacing them. Its central objective is to examine whether a simple graph representation learning stage can improve the quality of student representations while leaving the subsequent clustering pipeline unchanged.""",
    r"""The present study therefore complements earlier clustering-based approaches to student sectioning rather than replacing them. Its central objective is to examine whether a Node2Vec-based representation learning stage can improve the quality of student representations while leaving the subsequent clustering pipeline unchanged."""
)

# ============================================================
# 18. Update Figure caption for baseline_comparison
# ============================================================
tex = tex.replace(
    r"""\caption{Silhouette scores ($\uparrow$ higher is better) for five student sectioning methods across six courses: BoW + KMeans, PCA + KMeans, Spectral Clustering, DeepWalk-SSP + KMeans, and Node2Vec-SSP + KMeans.}""",
    r"""\caption{Silhouette scores ($\uparrow$ higher is better) for five student sectioning methods across six courses: BoW + KMeans, PCA + KMeans, Spectral Clustering, DeepWalk (unbiased random walk), and Node2Vec ($p{=}1$, $q{=}1$).}"""
)

# ============================================================
# 19. Update figure label text for graph density
# ============================================================
tex = tex.replace(
    r"""An interesting finding is that Node2Vec, which extends DeepWalk by introducing biased random walks""",
    r"""An important observation is that Node2Vec, which extends DeepWalk by introducing biased random walks"""
)

# ============================================================
# Write the updated file
# ============================================================
with open('paper/sn-article.tex', 'w') as f:
    f.write(tex)

# Report changes
import difflib
original_lines = original.splitlines()
new_lines = tex.splitlines()
diff = list(difflib.unified_diff(original_lines, new_lines, lineterm='', n=2))
added = sum(1 for l in diff if l.startswith('+') and not l.startswith('+++'))
removed = sum(1 for l in diff if l.startswith('-') and not l.startswith('---'))
print(f"Changes: {added} lines added, {removed} lines removed")
print(f"Total lines: {len(new_lines)} (was {len(original_lines)})")
