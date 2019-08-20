logLevel := Level.Warn

lazy val SbtSCoverageVersion = "1.5.1"
lazy val SbtScalaFmt = "1.5.1"

lazy val SbtCiRelease = "1.2.6"

addSbtPlugin("org.scoverage" % "sbt-scoverage" % SbtSCoverageVersion)
addSbtPlugin("com.geirsson" % "sbt-scalafmt" % SbtScalaFmt)

addSbtPlugin("com.geirsson" % "sbt-ci-release" % SbtCiRelease)

addSbtPlugin("org.scalameta" % "sbt-mdoc" % "1.3.1" )

addSbtPlugin("de.heikoseeberger" % "sbt-header" % "5.2.0")

addSbtPlugin("com.eed3si9n" % "sbt-unidoc" % "0.4.2")
