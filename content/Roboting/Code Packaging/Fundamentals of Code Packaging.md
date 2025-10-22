#programming #codePackaging 
# Why package code?
Generally, we want to have a standardized way to decide where compiled code from other people should go, what dependencies that compiled code needs, how to update and uninstall it, as well as ensuring consistent installations across machines.

Proper code packaging solves this by providing a standardized way to:
- **structure packages**
- **manage dependencies**
- **provide installation / removal scripts**
- **manage versions**

# General Packaging Concepts
All packaging systems share the following:
- **metadata** containing authorship, licensing, versioning, etc.
- **dependencies** specifying what else is needed
- **file manifests** regarding where files should go
- **install/remove hooks** to properly install and remove the package
- **build process** how to build the package from source
