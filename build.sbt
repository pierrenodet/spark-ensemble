lazy val SparkVersion = "3.1.1"
lazy val ScalaTestVersion = "3.0.5"
lazy val ScalaCheckVersion = "1.14.0"
lazy val SparkTestingBaseVersion = "3.0.1_1.0.0"

lazy val Scala212Version = "2.12.10"

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
    scalaVersion := Scala212Version))

lazy val core = project
  .in(file("core"))
  .settings(
    moduleName := "spark-ensemble",
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
      "com.holdenkarau" %% "spark-testing-base" % SparkTestingBaseVersion,
      "org.apache.spark" %% "spark-hive" % SparkVersion,
      "org.scalatest" %% "scalatest" % ScalaTestVersion,
      "org.scalacheck" %% "scalacheck" % ScalaCheckVersion).map(_ % Test))

lazy val docs = project
  .in(file("spark-ensemble-docs"))
  .settings(
    moduleName := "spark-ensemble-docs",
    skip in publish := true,
    mdocVariables := Map("VERSION" -> version.value),
    unidocProjectFilter in (ScalaUnidoc, unidoc) := inProjects(core),
    target in (ScalaUnidoc, unidoc) := (baseDirectory in LocalRootProject).value / "website" / "static" / "api",
    cleanFiles += (target in (ScalaUnidoc, unidoc)).value,
    docusaurusCreateSite := docusaurusCreateSite.dependsOn(unidoc in Compile).value
  )
  .dependsOn(core)
  .enablePlugins(MdocPlugin, DocusaurusPlugin, ScalaUnidocPlugin)
