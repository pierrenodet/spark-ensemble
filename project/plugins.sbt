logLevel := Level.Warn

lazy val SbtSCoverageVersion = "1.5.1"
lazy val SbtScalaFmt = "1.5.1"

lazy val SbtSonatypeVersion = "2.4"
lazy val SbtRelease = "1.0.11"

addSbtPlugin("org.scoverage" % "sbt-scoverage" % SbtSCoverageVersion)
addSbtPlugin("com.geirsson" % "sbt-scalafmt" % SbtScalaFmt)


addSbtPlugin("com.typesafe.sbt" % "sbt-git" % "1.0.0")
addSbtPlugin("com.github.gseitz" % "sbt-release" % SbtRelease)
addSbtPlugin("org.xerial.sbt" % "sbt-sonatype" % SbtSonatypeVersion)
addSbtPlugin("com.jsuereth" % "sbt-pgp" % "1.1.0")
