# Sector-Level Analysis and Clustering of S&P 500 Companies Using Financial Metrics and Machine Learning

## Abstract
Understanding the financial market structure and the positions of its players, i.e., companies and sectors, is of utmost importance [1]. While traditional methods are useful for interpreting financial statements, it is also crucial to understand the coherence of various sectors during market stress [2, 3]. The application of machine learning and statistics in financial markets has grown in importance [4]. This paper presents a comprehensive analysis of sector-level performance within the S&P 500, using a methodology that combines various financial metrics, clustering techniques, and neural networks [5]. Key metrics include market capitalization growth, revenue growth, performance variance (using weighted versus simple averages), and short- and long-term beta covariances [6]. This research identifies sector leaders, assesses market dominance, and explores sector stability [7]. It introduces an 'over-performance index' to identify which companies are perceived as leaders within a sector [8]. Additionally, inter-sector similarities are examined using clustering techniques, and a neural network classifies these sectors into clusters [10]. The findings provide critical insights into sector dynamics for investment decisions and identifying growth opportunities [11].

**Index Terms:** S&P 500, Financial Metrics, Clustering, Neural Networks, Sector Analysis, Overperformance Index, Financial Markets [12]

***

## I. INTRODUCTION

Large-Cap companies are key indicators of market trends and sectoral performance [12]. Understanding sector-level dynamics is essential for investors and policymakers [13]. This project integrates financial metrics, clustering techniques, and neural networks to analyze the competitive environment in various sectors [13]. By examining market capitalization, revenue growth, and variance, the project identifies monopolistic, duopolistic, and oligopolistic trends [14]. A neural network model is also used to classify sectors based on performance metrics, offering deeper insights into market behavior [15]. The main goal is to analyze sectors with monopolistic and oligopolistic environments [16]. To achieve this, an 'overperformance index' was developed, which utilizes a weighted average approach to better fit the variances to our specific requirements [17]. This index helps identify which companies in an industry over-perform the sector average [18]. After understanding each industry, they are clustered based on their normalized sector-wise performance [19]. Five parameters are formulated for each sector: year-on-year market capitalization growth, year-on-year revenue growth, the difference between simple and weighted average of year-on-year market capitalization growth, 6-month sector beta, and 4-year sector beta [20].

***

## II. THEORETICAL BACKGROUND

This research is founded on key economic concepts:

* **Market Structures:** The study focuses on identifying sectors characterized by monopoly, duopoly, and oligopoly [23].
    * A **monopoly** is a market where a single company dominates the sector [24].
    * A **duopoly** is a market where two companies dominate the sector [25].
    * An **oligopoly** is a market where several companies compete, with no single company being dominant [26].
* **Variance and Beta-Covariances:** These metrics are used to analyze and differentiate between market structures [28]. An index using a weighted average approach was developed to capture these metrics [29].

***

## III. METHODOLOGY

The methodology involves three main steps [30]:

### A. Data Preprocessing
Quarterly market capitalization and annual revenue data were merged, cleaned, and analyzed for each sector [31]. Key metrics like Year-over-Year (YoY) growth, weighted vs. simple averages, and beta covariance were calculated for each S&P 500 company [31]. Cumulative data was formed for each sector [32]. Only sectors with at least three listed companies in the S&P 500 were included to ensure data sufficiency and study quality [32, 33].

### B. Metrics and Equations

* **Market Cap/Revenue Growth Score:** [41]

$$\text{Return} = \frac{Y_i - Y_{i-1}}{Y_{i-1}} \quad (1)$$

$$\text{Score} = \begin{cases} 
1 & \text{if value > mean + 0.07 * SD} \\ 
-1 & \text{if value < mean - 0.07 * SD} \\ 
0 & \text{otherwise} 
\end{cases} \quad (2)$$

A threshold of 0.07 is used, based on the idea that a company should grow more than the risk-free interest rate to attract investors [34].

* **Weighted-Simple Variance:** This is the absolute difference between the weighted and simple averages for a sector [35].
    The weight $w_i$ associated with a company is defined as:

$$w_i = \frac{c_i^2}{\sum c_j^2} \quad (4)$$

where $c_i$ is the number of times company *i* outperformed the sector average [35].

* **Beta Covariance:** This measures the correlation of sector performance over time (short-term: 6 months; long-term: 4 years) [36].

$$\text{Mkt Value} = \frac{\text{Mkt Cap}(t)}{\text{Mkt Cap}(t-1)} - 1 \quad (5)$$

$$\text{Mkt Weight} = \frac{\text{Mkt Cap}(t)}{\sum \text{Mkt Value}(t)} \quad (6)$$

$$\text{Mkt Returns} = \sum (\text{Returns}(t) \cdot \text{Mkt Weight}(t)) \quad (7)$$

$$\beta = \frac{\text{Covariance}(\text{Returns}(t), \text{Mkt Returns}(t))}{\text{Variance}(\text{Mkt Returns}(t))} \quad (8)$$

### C. Sector Clustering
Five parameters were formulated for each sector: YoY market capitalization growth, YoY revenue growth, difference between simple and weighted average of YoY market cap growth, 6-month sector beta, and 4-year sector beta [20]. These parameters were normalized to prepare for clustering [43].
* **Hierarchical Clustering:** This was used to visualize cluster distances and determine the appropriate number of clusters [44].
* **K-Means Clustering:** This was performed using the number of clusters identified from the hierarchical clustering step [45].

### D. Overperformance Index
The weighted average growth of each sector was used to count the number of times each company outperformed the sector's mean growth [46]. This data was then used to create an overperformance-based index to classify sectors into monopolistic, duopolistic, or oligopolistic environments [47].

### E. Neural Network Model
A multi-layer perceptron (MLP) deep learning model was developed [48].
* **Input:** Normalized performance metrics for each sector [56].
* **Layers:** Three dense layers with ReLU activation, followed by a softmax layer for classification [57].
* **Output:** Cluster classification based on performance similarities [58].

![Architecture of the Artificial Neural Network](fig/arch.png)
*Fig. 1. Architecture of the Artificial Neural Network [56]*

The model was trained on 60% of the data and tested on the remaining 40% [52].

***

## IV. RESULTS AND DISCUSSION

Differences were observed for sectors with monopolistic and oligopolistic settings [59].

![Difference between simple and weighted average in different sectors](fig/mono-oligo.png)
*Fig. 2. The difference between simple and weighted average is significant in Casinos & Gaming Sector (on the left) and Communication Equipment Sector (on the right) [63]*

Out of 66 sectors analyzed, the study found:
* **47 sectors** with an oligopolistic environment [54].
* **8 sectors** with a duopolistic environment [54].
* **11 sectors** with a nearly monopolistic environment based on market capitalization growth [54].

A dendrogram was created to visualize the hierarchical clustering and decide on the number of clusters for K-Means [49]. "Automobile Manufacturers" was identified as an outlier [50]. Four clusters were obtained and initialized for K-Means clustering [51].

![Hierarchical Clustering of Sectors](fig/dendro.png)
*Fig. 3. Dendrogram representing 66 clusters and their distance for hierarchical clustering [67]*

**Confusion Matrix of Predicted Clusters** [60]

| True Label | Predicted 1 | Predicted 2 | Predicted 3 | Predicted 4 |
| :--------: | :---------: | :---------: | :---------: | :---------: |
| **1** | 6           | 0           | 1           | 0           |
| **2** | 8           | 0           | 0           | 1           |
| **3** | 0           | 0           | 0           | 2           |
| **4** | 0           | 0           | 0           | 9           |
*[Table data sourced from [61]]*

**Clustering Evaluation Metrics** [64]

| Metric                        | Score |
| ----------------------------- | :---: |
| Adjusted Rand Index (ARI)     | 0.76  |
| Normalized Mutual Information | 0.82  |
| Homogeneity                   | 0.81  |
| Completeness                  | 0.82  |
| V-measure                     | 0.82  |
| Fowlkes-Mallows Index         | 0.83  |
*[Table data sourced from [64]]*

The high scores indicate that the clustering aligns well with the true structure of the data [62]. The neural network classifier achieved an accuracy of **93%** using softmax activation [53].

***

## V. CONCLUSION

This research successfully analyzed the sector-wise performance within the S&P 500, offering valuable insights into market dynamics and sector clustering [67]. The methodology, combining financial metrics, hierarchical clustering, and neural networks, effectively revealed significant trends and patterns [68]. The study identified varying levels of market dominance, classifying 49 sectors as oligopolistic, 8 as duopolistic, and 12 as monopolistic based on market cap growth [69]. Inter-sector relationships were successfully identified using hierarchical and K-means clustering [70]. The neural network model demonstrated a high classification accuracy of 93% [71]. These findings are important for strategic investment, policy formulation, and market analysis by providing a deeper understanding of sector-level stability and competitive dynamics [72]. This research highlights the effectiveness of combining financial expertise with data science and machine learning [74]. Future work could incorporate ESG metrics, international market data, and more advanced deep learning models [73].

***
