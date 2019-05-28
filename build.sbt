name := "spark-ensemble"
scalaVersion := "2.12.8"

inThisBuild(
  List(
    organization := "com.github.pierrenodet",
    homepage := Some(url(s"https://github.com/pierrenodet/spark-ensemble")),
    licenses := List("Apache-2.0" -> url("http://www.apache.org/licenses/LICENSE-2.0")),
    developers := List(
      Developer(
        "pierrenodet",
        "Pierre Nodet",
        "nodet.pierre@gmail.com",
        url("https://github.com/pierrenodet"))
    )
  )
)

lazy val SparkVersion = "2.4.3"
lazy val ScalaTestVersion = "3.0.5"
lazy val ScalaCheckVersion = "1.14.0"
lazy val SparkTestingBaseVersion = "0.12.0"

libraryDependencies += "org.apache.spark" %% "spark-core" % SparkVersion % Provided
libraryDependencies += "org.apache.spark" %% "spark-sql" % SparkVersion % Provided
libraryDependencies += "org.apache.spark" %% "spark-mllib" % SparkVersion % Provided

libraryDependencies += "com.holdenkarau" %% "spark-testing-base" % (SparkVersion + "_" + SparkTestingBaseVersion) % Test
libraryDependencies += "org.apache.spark" %% "spark-hive" % SparkVersion % Test
libraryDependencies += "org.scalatest" %% "scalatest" % ScalaTestVersion % Test
libraryDependencies += "org.scalacheck" %% "scalacheck" % ScalaCheckVersion % Test

fork in Test := true
parallelExecution in Test := false
javaOptions ++= Seq("-Xms512M", "-Xmx2048M", "-XX:MaxPermSize=2048M", "-XX:+CMSClassUnloadingEnabled")