lazy val SparkVersion = "2.4.3"
lazy val ScalaTestVersion = "3.0.5"
lazy val ScalaCheckVersion = "1.14.0"
lazy val SparkTestingBaseVersion = "0.12.0"

lazy val Scala211Version = "2.11.12"
lazy val Scala212Version = "2.12.8"

inThisBuild(
  List(
    name := "spark-ensemble",
    organization := "com.github.pierrenodet",
    organizationName := "Pierre Nodet",
    startYear := Some(2019),
    homepage := Some(url(s"https://github.com/pierrenodet/spark-ensemble")),
    licenses := List("Apache-2.0" -> url("http://www.apache.org/licenses/LICENSE-2.0")),
    developers := List(
      Developer(
        "pierrenodet",
        "Pierre Nodet",
        "nodet.pierre@gmail.com",
        url("https://github.com/pierrenodet"))),
    crossScalaVersions := List(Scala212Version, Scala211Version)))

lazy val core = project
  .in(new File("core"))
  .settings(
    javaOptions ++= Seq(
      "-Xms512M",
      "-Xmx2048M",
      "-XX:MaxPermSize=2048M",
      "-XX:+CMSClassUnloadingEnabled"),
    fork.in(Test, run) := true,
    parallelExecution.in(Test) := false,
    libraryDependencies ++= Seq(
      "org.apache.spark" %% "spark-core" % SparkVersion % Provided,
      "org.apache.spark" %% "spark-sql" % SparkVersion % Provided,
      "org.apache.spark" %% "spark-mllib" % SparkVersion % Provided),
    libraryDependencies ++= Seq(
      "com.holdenkarau" %% "spark-testing-base" % (SparkVersion + "_" + SparkTestingBaseVersion),
      "org.apache.spark" %% "spark-hive" % SparkVersion,
      "org.scalatest" %% "scalatest" % ScalaTestVersion,
      "org.scalacheck" %% "scalacheck" % ScalaCheckVersion).map(_ % Test))

lazy val docs = project
  .in(file("docs"))
  .settings(mdocVariables := Map("VERSION" -> version.value))
  .dependsOn(core)
  .enablePlugins(MdocPlugin)
