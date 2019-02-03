logLevel := Level.Warn

lazy val SbtSCoverageVersion = "1.5.1"
lazy val SbtScalaFmt = "1.5.1"

addSbtPlugin("org.scoverage" % "sbt-scoverage" % SbtSCoverageVersion)
addSbtPlugin("com.geirsson" % "sbt-scalafmt" % SbtScalaFmt)
