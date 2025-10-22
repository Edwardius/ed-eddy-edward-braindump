#debian #codePackaging 

This is mainly with regards to Linux and debian packages.

# Setup
So you have a `.deb` file from learning the [[Fundamentals of Code Packaging]]. Well how do you expose this package to other users?

Well its pretty interesting, and kinda feels like an official way to move around zip files with binaries in them instead of source code.

# Method 1: Direct Host and Download
You can put your `.deb` file on any web server and share the URL

```
wget https://example.com/downloads/myapp_1.0.0_amd64.deb sudo apt install ./myapp_1.0.0_amd64.deb
```

This is common for small user bases (and a *good enough* approach that I think a lot of people do).

