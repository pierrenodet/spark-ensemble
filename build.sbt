lazy val sparkVersion = "3.3.0"
lazy val scalaCheckVersion = "1.16.0"
lazy val scalaTestVersion = "3.2.13"
lazy val scala212Version = "2.12.15"
lazy val scala213Version = "2.13.10"

Global / onChangedBuildSource := ReloadOnSourceChanges

ThisBuild / organization := "com.github.pierrenodet"
ThisBuild / organizationName := "Pierre Nodet"
ThisBuild / homepage := Some(url(s"https://github.com/pierrenodet/spark-ensemble"))
ThisBuild / startYear := Some(2019)
ThisBuild / licenses := Seq(License.Apache2)
ThisBuild / developers := List(
  Developer(
    "pierrenodet",
    "Pierre Nodet",
    "nodet.pierre@gmail.com",
    url("https://github.com/pierrenodet")))
ThisBuild / scalaVersion := scala213Version
ThisBuild / crossScalaVersions := Seq(scala212Version, scala213Version)
ThisBuild / githubWorkflowJavaVersions := Seq("8", "11").map(JavaSpec.temurin(_))
ThisBuild / githubWorkflowTargetTags ++= Seq("v*")
ThisBuild / githubWorkflowPublishTargetBranches +=
  RefPredicate.StartsWith(Ref.Tag("v"))
ThisBuild / githubWorkflowPublish := Seq(
  WorkflowStep.Sbt(
    List("ci-release"),
    env = Map(
      "PGP_PASSPHRASE" -> "${{ secrets.PGP_PASSPHRASE }}",
      "PGP_SECRET" -> "${{ secrets.PGP_SECRET }}",
      "SONATYPE_PASSWORD" -> "${{ secrets.SONATYPE_PASSWORD }}",
      "SONATYPE_USERNAME" -> "${{ secrets.SONATYPE_USERNAME }}")),
  WorkflowStep.Sbt(
    List("docs/docusaurusPublishGhpages"),
    env = Map("GIT_DEPLOY_KEY" -> "${{ secrets.GIT_DEPLOY_KEY }}")))
ThisBuild / githubWorkflowBuild := Seq(
  WorkflowStep.Sbt(List("coverage", "test", "coverageReport")))
ThisBuild / githubWorkflowBuildPostamble := Seq(
  WorkflowStep.Run(List("bash <(curl -s https://codecov.io/bash)")))
ThisBuild / resolvers ++= Resolver.sonatypeOssRepos("public")
ThisBuild / resolvers ++= Resolver.sonatypeOssRepos("snapshots")
ThisBuild / versionScheme := Some("semver-spec")

lazy val commonSettings = Seq(
  Compile / doc / scalacOptions --= Seq("-Xfatal-warnings"),
  Compile / doc / scalacOptions ++= Seq(
    "-groups",
    "-sourcepath",
    (LocalRootProject / baseDirectory).value.getAbsolutePath,
    "-doc-source-url",
    "https://github.com/pierrenodet/spark-ensemble/blob/v" + version.value + "€{FILE_PATH}.scala"),
  Test / run / fork := true,
  Test / parallelExecution := false)

lazy val core = project
  .in(file("core"))
  .enablePlugins(AutomateHeaderPlugin)
  .settings(commonSettings)
  .settings(
    name := "spark-ensemble",
    description := "Ensemble Learning for Apache Spark",
    libraryDependencies ++= Seq(
      "org.apache.spark" %% "spark-core" % sparkVersion % Provided,
      "org.apache.spark" %% "spark-sql" % sparkVersion % Provided,
      "org.apache.spark" %% "spark-mllib" % sparkVersion % Provided),
    libraryDependencies ++= Seq(
      "org.scalatest" %% "scalatest" % scalaTestVersion,
      "org.scalatest" %% "scalatest-funsuite" % scalaTestVersion,
      "org.scalacheck" %% "scalacheck" % scalaCheckVersion,
      "org.scalatestplus" %% ("scalacheck" + "-" + scalaCheckVersion
        .split("\\.")
        .toList
        .take(2)
        .mkString("-")) % (scalaTestVersion + ".0")).map(_ % Test))

lazy val docs = project
  .in(file("spark-ensemble-docs"))
  .dependsOn(core)
  .enablePlugins(MdocPlugin, DocusaurusPlugin, ScalaUnidocPlugin)
  .settings(commonSettings)
  .settings(
    name := "spark-ensemble-docs",
    publish / skip := true,
    mdocVariables := Map("VERSION" -> version.value.takeWhile(_ != '+')),
    mdocIn := new File("docs"),
    ScalaUnidoc / unidoc / unidocProjectFilter := inProjects(core),
    ScalaUnidoc / unidoc / target := (LocalRootProject / baseDirectory).value / "website" / "static" / "api",
    cleanFiles += (ScalaUnidoc / unidoc / target).value,
    docusaurusCreateSite := docusaurusCreateSite.dependsOn(Compile / unidoc).value,
    docusaurusPublishGhpages := docusaurusPublishGhpages.dependsOn(Compile / unidoc).value,
    githubWorkflowArtifactUpload := false,
    libraryDependencies ++= Seq(
      "org.apache.spark" %% "spark-core" % sparkVersion % Provided,
      "org.apache.spark" %% "spark-sql" % sparkVersion % Provided,
      "org.apache.spark" %% "spark-mllib" % sparkVersion % Provided))
