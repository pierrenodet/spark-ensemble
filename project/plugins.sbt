logLevel := Level.Warn

lazy val SbtSCoverageVersion = "1.5.1"
lazy val SbtScalaFmt = "1.5.1"

lazy val SbtCiRelease = "1.2.2"
lazy val SbtRelease = "1.0.11"

addSbtPlugin("org.scoverage" % "sbt-scoverage" % SbtSCoverageVersion)
addSbtPlugin("com.geirsson" % "sbt-scalafmt" % SbtScalaFmt)

addSbtPlugin("com.geirsson" % "sbt-ci-release" % SbtCiRelease)
addSbtPlugin("com.github.gseitz" % "sbt-release" % SbtRelease)

