inThisBuild(
  List(
    organization := "com.github.pierrenodet",
    homepage := Some(url("https://github.com/pierrenodet/spark-bagging")),
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

name := "spark-bagging"
version := "0.0.1"
scalaVersion := "2.11.12"

lazy val SparkVersion = "2.4.0"
lazy val ScalaTestVersion = "3.0.5"
lazy val ScalaCheckVersion = "1.13.4"
lazy val SparkTestingBaseVersion = "0.11.0"

libraryDependencies += "org.apache.spark" %% "spark-sql" % SparkVersion % Provided
libraryDependencies += "org.apache.spark" %% "spark-mllib" % SparkVersion % Provided

libraryDependencies += "com.holdenkarau" %% "spark-testing-base" % (SparkVersion + "_" + SparkTestingBaseVersion) % Test
libraryDependencies += "org.apache.spark" %% "spark-hive" % SparkVersion % Test
libraryDependencies += "org.scalatest" %% "scalatest" % ScalaTestVersion % Test
libraryDependencies += "org.scalacheck" %% "scalacheck" % ScalaCheckVersion % Test

fork in Test := true
parallelExecution in Test := false
javaOptions ++= Seq("-Xms512M", "-Xmx2048M", "-XX:MaxPermSize=2048M", "-XX:+CMSClassUnloadingEnabled")
testOptions in Test += Tests.Argument(TestFrameworks.ScalaTest, "-oD")

artifactName := { (sv: ScalaVersion, module: ModuleID, artifact: Artifact) =>
  artifact.name + "_" + sv.binary + "-" + module.revision + "." + artifact.extension
}