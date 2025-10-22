#codePackaging #debian
# Debian Philosophy
Debian uses a **"source package" -> "binary package" model** which means that you as the provider of the package have the source code and the scripts to package it while other users of your code only have access to a package of compiled binaries.

# Debian Directory Structure
```
myproject-1.0/
├── src/              # Your actual code
├── CMakeLists.txt    # Your build system
└── debian/
    ├── control       # THE BRAIN: metadata & dependencies
    ├── rules         # THE BUILDER: how to compile & install
    ├── changelog     # THE HISTORY: versions & changes
    ├── compat        # Debhelper version (usually just "13")
    ├── install       # Optional: file installation mapping
    ├── copyright     # License info
    ├── source/
    │   └── format    # "3.0 (native)" or "3.0 (quilt)"
    └── *.service     # Systemd units, config files, etc.
```

## The `control` file contains the following:

```
Source: my-package
Section: arbitrary?
Priority: optional
Maintainer: Your Name <you@example.com>
Build-Depends: debhelper-compat (= 13), cmake, g++ <----- key field
Standards-Version: 4.6.0

Package: my-package
Architecture: any <----- key field
Depends: ${shlibs:Depends} <----- key field, ${misc:Depends}, libstdc++6 <----- key field
Description: some package of programs that are useful for blah blah
```

## The `rules` file contains the following:

```
#!/usr/bin/make -f

%:
	dh $@

# Override specific steps if needed
override_dh_auto_configure:
	dh_auto_configure -- -DCMAKE_BUILD_TYPE=Release
```

This is a build script. In this case its a makefile.

## The `changelog` contains the following:

```
my-calculator (1.0-1) unstable; urgency=medium

  * Initial release
  * Added basic arithmetic operations

 -- Your Name <you@example.com>  Mon, 13 Oct 2025 10:00:00 -0400
```

Which follows a strict format that is made with `dch -i`

# The Build Process
Pretty simple, prepare your source code, create a debian template, fill the template, build, and **BOOM** you got a .deb file.

```bash
# 1. Prepare your source
cd myproject-1.0/
dh_make --createorig  # Creates debian/ template

# 2. Edit debian/control, rules, changelog

# 3. Build the package
debuild -us -uc  # Build without signing

# 4. Output files
ls ../*.deb      # Your binary package!
```

# Installation Hierarchy

Debian has strict rules about where files should go:
```
/usr/bin/           # User executables
/usr/lib/           # Libraries
/usr/share/         # Arch-independent data
/etc/               # Configuration files
/var/lib/myapp/     # Application state/data
/usr/share/doc/     # Documentation
```