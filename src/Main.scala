/**
  * Created by seyz on 09/05/17.
  */

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.clustering._
import org.apache.spark.rdd._

object Main extends App{

  val config = new SparkConf().setAppName("Simple Application").setMaster("local[*]")
  val sc = new SparkContext(config)
  sc.setLogLevel("ERROR")

  def bagsFromDocumentPerLine(filename:String) =
    sc.textFile(filename)
      .map(_.split(" ")
        .filter(x => x.length > 5 && x.toLowerCase != "reuter")
        .map(_.toLowerCase)
        .groupBy(x => x)
        .toList
        .map(x => (x._1, x._2.size)))

  val rddBags:RDD[List[Tuple2[String,Int]]] =
    bagsFromDocumentPerLine("rcorpus")

  val vocab:Array[Tuple2[String,Long]] =
    rddBags.flatMap(x => x)
      .reduceByKey(_ + _)
      .map(_._1)
      .zipWithIndex
      .collect

  def codeBags(rddBags:RDD[List[Tuple2[String,Int]]]) =
    rddBags.map(x => (x ++ vocab).groupBy(_._1)
      .filter(_._2.size > 1)
      .map(x => (x._2(1)._2.asInstanceOf[Long]
        .toInt,
        x._2(0)._2.asInstanceOf[Int]
          .toDouble))
      .toList)
      .zipWithIndex.map(x => (x._2, new SparseVector(
      vocab.size,
      x._1.map(_._1).toArray,
      x._1.map(_._2).toArray)
      .asInstanceOf[Vector]))

  val model = new LDA().setK(5).run(codeBags(rddBags))
  val topics = model.topicsMatrix

  for(topic <- Range(0, 5)) {
    println("Topic " + topic + " : ")
    for(word <- Range(0, model.vocabSize)){
      print(" " + topics(word, topic))
    }
    println()
  }
}
