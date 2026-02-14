This project performs large-scale text analytics and graph analysis on a corpus of books using Apache Spark

1.It demonstrates:
2. Distributed text preprocessing
3. Word frequency analysis
4. N-gram extraction
5. TF-IDF vectorization
6. Cosine similarity between documents
7. Construction of an Author Influence Network
8. Graph degree analysis (in-degree / out-degree)

The goal is to showcase distributed data processing + ML feature engineering + graph analytics using Spark.



🛠️ Tech Stack

Apache Spark 3.4.2
Scala 2.12
Spark SQL
Spark MLlib
Breeze (for cosine similarity)
Local mode (local[2])


#DATASET
books/D184MB/*.txt





Question 1 – Text Preprocessing

Load books using wholeTextFiles

Remove special characters

Convert to lowercase

Create clean text column





Demonstrates:

RDD → DataFrame conversion

Regex-based text cleaning

Column transformations





Question 2 – Word Frequency (Corpus Level)

Tokenize text

Remove short words

Count global word frequency

Identify top words




Demonstrates:

Tokenization

explode()

groupBy + count

Sorting large aggregations




Question 3 – Top Words Per Book

Explode words per document

Count word frequency per file

Rank most common words per book




Demonstrates:

Multi-column groupBy

Document-level analytics




Question 4 – N-Gram Analysis (Bigrams)

Generate 2-word sequences

Compute bigram frequency




Demonstrates:

Spark ML NGram

Pattern analysis across corpus




Question 5 – TF-IDF Vectorization

Pipeline:

Tokenization

Stopword removal

HashingTF (1000 features)

IDF fitting

TF-IDF vector generation

Output:

[file_name, tfidf_vector]




Demonstrates:

Feature engineering at scale

Sparse vector representation

Document vectorization





Question 6 – Cosine Similarity Between Books

Approach:

Cross join TF-IDF vectors

Compute cosine similarity using Breeze

Rank most similar books

Formula:

cosine(A,B)= A.B / ∣∣A∣∣⋅∣∣B∣∣ 
	​


Demonstrates:

Vector math

UDF with Breeze

Pairwise similarity computation




Question 12 – Author Influence Network
Objective

Construct a graph where:

(author1 → author2)


if:

author2 released a book within X years of author1


Default:

X = 5 years




Network Construction

Extract:

author

release year

Self join on year condition

Filter within time window

Create edges DataFrame

Edge format:

(author1, author2)



Graph Analysis

Compute:

Out-degree (authors influencing others)

In-degree (authors influenced by others)

Identify:

Top 5 most influential authors

Top 5 most influenced authors



Representation Choice

We used:

DataFrame representation


Instead of RDD.

Advantages

Catalyst optimization

Tungsten execution engine

Built-in aggregation

Easier degree computation

SQL-style readability

Disadvantages

Cross joins are expensive

Less graph-native than GraphFrames

Harder for recursive algorithms



Effect of Time Window (X)

Small X → Sparse network

Large X → Dense network

Very large X → Almost complete graph

The structure changes significantly with X.



Limitations of Influence Definition

This model assumes:

Temporal proximity = influence


Which is unrealistic because:

Influence can span decades

Authors may never read each other

Genre differences ignored

Cultural/geographic context ignored

Multiple books per author not considered separately

It is a simplified academic approximation.



Scalability Analysis

Current approach:

Uses cross joins

O(N²) comparisons

With millions of books:

Not scalable directly

Possible Optimizations

Partition by year

Use range join instead of full cross join

Bucket by time window

Use broadcast joins

Use GraphFrames

Persist intermediate datasets

Pre-aggregate author-year data



#How To Run

Start Spark:

cd spark-3.4.2-bin-hadoop3
./bin/spark-shell --master local[2]


Then execute Scala files from src/.

#Learning Outcomes

This project demonstrates:

Distributed text processing

ML feature engineering

Document similarity modeling

Graph modeling using Spark

Degree centrality analysis

Performance considerations in big data


