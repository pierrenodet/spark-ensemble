name := "spark-bagging"
version := "0.0.1"
scalaVersion := "2.11.12"

inThisBuild(
  List(
    organization := "com.github.pierrenodet",
    homepage := Some(url(s"https://github.com/pierrenodet/$name")),
    licenses := List("Apache-2.0" -> url("http://www.apache.org/licenses/LICENSE-2.0")),
    scmInfo := Some(ScmInfo(
      url(s"https://github.com/pierrenodet/$name"),
      s"scm:git@github.com:pierrenodet/$name.git"
    )),
    developers := List(
      Developer(
        "pierrenodet",
        "Pierre Nodet",
        "nodet.pierre@gmail.com",
        url("https://github.com/pierrenodet"))
    )
  )
)

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
  artifact.name + "_" + sv.binary + "-" + SparkVersion + "_" + module.revision + "." + artifact.extension
}

useGpg := false
usePgpKeyHex("8CB6A95796CADBFA")
pgpPublicRing := baseDirectory.value / "project" / ".gnupg" / "pubring.gpg"
pgpSecretRing := baseDirectory.value / "project" / ".gnupg" / "secring.gpg"
pgpPassphrase := sys.env.get("PGP_PASS").map(_.toArray)

import ReleaseTransformations._

sonatypeProfileName := organization.value

// To sync with Maven central, you need to supply the following information:
publishMavenStyle := true

credentials += Credentials(
  "Sonatype Nexus Repository Manager",
  "oss.sonatype.org",
  sys.env.getOrElse("SONATYPE_USER", ""),
  sys.env.getOrElse("SONATYPE_PASS", "")
)

isSnapshot := version.value endsWith "SNAPSHOT"

publishTo := Some(
  if (isSnapshot.value)
    Opts.resolver.sonatypeSnapshots
  else
    Opts.resolver.sonatypeStaging
)

releaseCrossBuild := false
releaseProcess := Seq[ReleaseStep](
  checkSnapshotDependencies,
  inquireVersions,
  runClean,
  runTest,
  setReleaseVersion,
  commitReleaseVersion,
  tagRelease,
  releaseStepCommand("publishSigned"),
  setNextVersion,
  commitNextVersion,
  releaseStepCommand("sonatypeReleaseAll"),
  pushChanges
)

enablePlugins(GitVersioning)

/* The BaseVersion setting represents the in-development (upcoming) version,
 * as an alternative to SNAPSHOTS.
 */
git.baseVersion := version.value

val ReleaseTag = """^v([\d\.]+)$""".r
git.gitTagToVersionNumber := {
  case ReleaseTag(v) => Some(v)
  case _ => None
}

git.formattedShaVersion := {
  val suffix = git.makeUncommittedSignifierSuffix(git.gitUncommittedChanges.value, git.uncommittedSignifier.value)

  git.gitHeadCommit.value map {
    _.substring(0, 7)
  } map { sha =>
    git.baseVersion.value + "-" + sha + suffix
  }
}

addCommandAlias("ci-all", ";+clean ;+compile ;+coverage ;+test ;+coverageReport ;+package")
addCommandAlias("release", ";+publishSigned ;sonatypeReleaseAll")