Compilation unit in Rust. 

When a file is called like `rustc some_file.rs` first all its `mod` get expanded and inserted in place. Then compilation occurs. `some_file.rs` is called a crate. Not the modules. 

A crate can be compiled into a binary or into a library.

## Creating a Library
```bash
$ rustc --create-type=lib sample.rs
$ ls lib* # by default libraries and prefixed with "lib", can be overridden with --crate-name
libsample.rlib
```

## Using a Library
```bash
# Where libsample.rlib is the path to the compiled library, assumed that it's
# in the same directory here:
$ rustc executable.rs --extern rary=libsample.rlib && ./executable
called rary's `public_function()`
called rary's `indirect_access()`, that
> called rary's `private_function()`
```

```rust
// extern crate rary; // May be required for Rust 2015 edition or earlier

fn main() {
    sample::public_function();

    // Error! `private_function` is private
    //sample::private_function();

    sample::indirect_access();
}
```