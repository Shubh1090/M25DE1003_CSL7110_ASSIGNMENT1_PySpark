Q1 – Basic Preprocessing

import spark.implicits._
import org.apache.spark.sql.functions._

// Load books
val df = spark.sparkContext
  .wholeTextFiles("books/D184MB/*.txt")
  .toDF("file_name","text")
  .limit(10)

println("Total Books: " + df.count())

// Clean text
val clean = df.withColumn(
  "clean_text",
  lower(regexp_replace($"text", "[^a-zA-Z\\s]", ""))
)

clean.select("file_name","clean_text").show(3,false)



Q2 – Word Frequency

import spark.implicits._
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature._

val df = spark.sparkContext
  .wholeTextFiles("books/D184MB/*.txt")
  .toDF("file_name","text")
  .limit(10)

val clean = df.withColumn(
  "clean_text",
  lower(regexp_replace($"text", "[^a-zA-Z\\s]", ""))
)

val words = new Tokenizer()
  .setInputCol("clean_text")
  .setOutputCol("words")
  .transform(clean)

val exploded = words
  .select(explode($"words").as("word"))
  .filter(length($"word") > 2)

val wordCount = exploded
  .groupBy("word")
  .count()
  .orderBy(desc("count"))

wordCount.show(20,false)



Q3 – Top Words Per Book

import spark.implicits._
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature._

val df = spark.sparkContext
  .wholeTextFiles("books/D184MB/*.txt")
  .toDF("file_name","text")
  .limit(10)

val clean = df.withColumn(
  "clean_text",
  lower(regexp_replace($"text", "[^a-zA-Z\\s]", ""))
)

val words = new Tokenizer()
  .setInputCol("clean_text")
  .setOutputCol("words")
  .transform(clean)

val exploded = words
  .select($"file_name", explode($"words").as("word"))
  .filter(length($"word") > 2)

val topPerBook = exploded
  .groupBy("file_name","word")
  .count()
  .orderBy(desc("count"))

topPerBook.show(20,false)



Q4 – N-Gram Analysis

import spark.implicits._
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature._

val df = spark.sparkContext
  .wholeTextFiles("books/D184MB/*.txt")
  .toDF("file_name","text")
  .limit(10)

val clean = df.withColumn(
  "clean_text",
  lower(regexp_replace($"text", "[^a-zA-Z\\s]", ""))
)

val words = new Tokenizer()
  .setInputCol("clean_text")
  .setOutputCol("words")
  .transform(clean)

val bigram = new NGram()
  .setN(2)
  .setInputCol("words")
  .setOutputCol("bigrams")
  .transform(words)

val exploded = bigram
  .select(explode($"bigrams").as("bigram"))

val bigramCount = exploded
  .groupBy("bigram")
  .count()
  .orderBy(desc("count"))

bigramCount.show(20,false)



Q5 – TF-IDF Generation

import spark.implicits._
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature._

val df = spark.sparkContext
  .wholeTextFiles("books/D184MB/*.txt")
  .toDF("file_name","text")
  .limit(10)

val clean = df.withColumn(
  "clean_text",
  lower(regexp_replace($"text", "[^a-zA-Z\\s]", ""))
)

val words = new Tokenizer()
  .setInputCol("clean_text")
  .setOutputCol("words")
  .transform(clean)

val filtered = new StopWordsRemover()
  .setInputCol("words")
  .setOutputCol("filtered")
  .transform(words)

val tf = new HashingTF()
  .setInputCol("filtered")
  .setOutputCol("tf")
  .setNumFeatures(1000)
  .transform(filtered)

val idfModel = new IDF()
  .setInputCol("tf")
  .setOutputCol("tfidf")
  .fit(tf)

val tfidf = idfModel.transform(tf)
  .select("file_name","tfidf")

tfidf.show(5,false)



Cosine Similarity Between Books

import spark.implicits._
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg._
import breeze.linalg._

val df = spark.sparkContext
  .wholeTextFiles("books/D184MB/*.txt")
  .toDF("file_name","text")
  .limit(10)

val clean = df.withColumn(
  "clean_text",
  lower(regexp_replace($"text", "[^a-zA-Z\\s]", ""))
)

val words = new Tokenizer()
  .setInputCol("clean_text")
  .setOutputCol("words")
  .transform(clean)

val filtered = new StopWordsRemover()
  .setInputCol("words")
  .setOutputCol("filtered")
  .transform(words)

val tf = new HashingTF()
  .setInputCol("filtered")
  .setOutputCol("tf")
  .setNumFeatures(1000)
  .transform(filtered)

val idfModel = new IDF()
  .setInputCol("tf")
  .setOutputCol("tfidf")
  .fit(tf)

val tfidf = idfModel.transform(tf)
  .select("file_name","tfidf")

val left = tfidf.select($"file_name".as("fileA"), $"tfidf".as("vecA"))
val right = tfidf.select($"file_name".as("fileB"), $"tfidf".as("vecB"))

val cross = left.crossJoin(right)
  .filter($"fileA" =!= $"fileB")

val cosineUDF = udf((v1: Vector, v2: Vector) => {
  val dot = v1.asBreeze.dot(v2.asBreeze)
  val normA = norm(v1.asBreeze)
  val normB = norm(v2.asBreeze)
  dot / (normA * normB)
})

val finalSim = cross.withColumn("cosine", cosineUDF($"vecA",$"vecB"))

finalSim.orderBy(desc("cosine")).show(10,false)



Q11.

import spark.implicits._
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg._

// Load 10 books
val df = spark.sparkContext
  .wholeTextFiles("books/D184MB/*.txt")
  .toDF("file_name", "text")
  .limit(10)

println("Total Books: " + df.count())

// Cleaning
val clean = df.withColumn(
  "clean_text",
  lower(regexp_replace($"text", "[^a-zA-Z\\s]", ""))
)

// Tokenization
val words = new Tokenizer()
  .setInputCol("clean_text")
  .setOutputCol("words")
  .transform(clean)

// Stopword removal
val filtered = new StopWordsRemover()
  .setInputCol("words")
  .setOutputCol("filtered")
  .transform(words)

// TF
val tf = new HashingTF()
  .setInputCol("filtered")
  .setOutputCol("tf")
  .setNumFeatures(1000)
  .transform(filtered)

// IDF
val idfModel = new IDF()
  .setInputCol("tf")
  .setOutputCol("tfidf")
  .fit(tf)

val tfidf = idfModel
  .transform(tf)
  .select("file_name", "tfidf")

println("TF-IDF Generated")

// Cosine Similarity
val left = tfidf.select($"file_name".as("fileA"), $"tfidf".as("vecA"))
val right = tfidf.select($"file_name".as("fileB"), $"tfidf".as("vecB"))

val cross = left.crossJoin(right)
  .filter($"fileA" =!= $"fileB")

val cosineUDF = udf((v1: Vector, v2: Vector) => {
  val dot = v1.asBreeze.dot(v2.asBreeze)
  val normA = breeze.linalg.norm(v1.asBreeze)
  val normB = breeze.linalg.norm(v2.asBreeze)
  dot / (normA * normB)
})

val finalSim = cross.withColumn(
  "cosine",
  cosineUDF($"vecA", $"vecB")
)

finalSim.orderBy(desc("cosine")).show(10, false)



Q12_AuthorInfluenceNetwork

import spark.implicits._
import org.apache.spark.sql.functions._

// Load books
val df = spark.sparkContext
  .wholeTextFiles("books/D184MB/*.txt")
  .toDF("file_name", "text")

println("Total Books: " + df.count())

// Extract Author
val authorRegex = "(?i)author:\\s*(.*)".r
val yearRegex = "(?i)(18|19|20)\\d{2}".r

val extractAuthor = udf((text: String) => {
  authorRegex.findFirstMatchIn(text)
    .map(_.group(1).trim)
    .getOrElse("Unknown")
})

val extractYear = udf((text: String) => {
  yearRegex.findFirstMatchIn(text)
    .map(_.matched.toInt)
    .getOrElse(0)
})

// Create author-year table
val books = df
  .withColumn("author", extractAuthor($"text"))
  .withColumn("year", extractYear($"text"))
  .filter($"author" =!= "Unknown" && $"year" > 0)

books.select("author","year").show(5)

// Build Influence Network (X = 5 years)
val left = books.select($"author".as("author1"), $"year".as("year1"))
val right = books.select($"author".as("author2"), $"year".as("year2"))

val edges = left.crossJoin(right)
  .filter($"author1" =!= $"author2")
  .filter($"year2" >= $"year1" && $"year2" <= $"year1" + 5)
  .select("author1","author2")

println("Total Influence Edges: " + edges.count())

// Out-degree
val outDeg = edges
  .groupBy("author1")
  .count()
  .orderBy(desc("count"))

println("Top 5 Influential Authors:")
outDeg.show(5, false)

// In-degree
val inDeg = edges
  .groupBy("author2")
  .count()
  .orderBy(desc("count"))

println("Top 5 Influenced Authors:")
inDeg.show(5, false)


