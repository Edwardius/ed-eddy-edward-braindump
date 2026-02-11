The official Rust package management tool. 

## Dependencies
To create a new rust project
```bash
cargo new foo // creates a binary
cargo new --lib bar // creates a library
```

This will produce something like the following
```
.
├── bar
│   ├── Cargo.toml
│   └── src
│       └── lib.rs
└── foo
    ├── Cargo.toml
    └── src
        └── main.rs
```

`Cargo.toml` is the configuration file for cargo in this project.

```toml
[package]
name = "foo"
version = "0.1.0"
authors = ["mark"]

[dependencies]
clap = "2.27.1" # from crates.io
rand = { git = "https://github.com/rust-lang-nursery/rand" } # from online repo
bar = { path = "../bar" } # from a path in the local filesystem

```

you can then build your project with `cargo build` and build and run with `cargo run`

## Conventions
Add additional binaries under `bin/`
```
foo
├── Cargo.toml
└── src
    ├── main.rs
    └── bin
        └── my_other_bin.rs
```


## Testing
Put under `test/` directory
```
foo
├── Cargo.toml
├── src
│   └── main.rs
│   └── lib.rs
└── tests
    ├── my_test.rs
    └── my_other_test.rs
```

`cargo test` will run these

