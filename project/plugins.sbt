logLevel := Level.Warn

lazy val SbtSCoverageVersion = "1.5.1"
lazy val SbtScalaFmt = "1.5.1"

lazy val SbtCiRelease = "1.2.6"

addSbtPlugin("org.scoverage" % "sbt-scoverage" % SbtSCoverageVersion)
addSbtPlugin("com.geirsson" % "sbt-scalafmt" % SbtScalaFmt)

addSbtPlugin("com.geirsson" % "sbt-ci-release" % SbtCiRelease)

