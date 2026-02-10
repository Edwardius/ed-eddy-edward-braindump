This covers comments and formatted prints
```rust
fn main() {
	// prints text to console
	println!("Hello world");
}
```

## Comments

There are **regular comments:**
- `//` for commenting a line
- `/*`  and `*/` for commenting multiple lines out

There are **doc comments** which get parsed to an html library:
- `///` generates  library docs for the following item
- `//!` generates library docs for the enclosing item

## Formatted Print

Theres are bunch of macros...
```rust
fn main() {
	// {} will be replaced with any argument, stringified
	println!("{} days", 31);
	
	// You can index the argument
	println!("{1} hahahaa {0}", "alice", "bob"); //will print "bob hahahaa alice"
	
	// Can be referenced by name
    println!("{subject} {verb} {object}",
             object="the lazy dog",
             subject="the quick brown fox",
             verb="jumps over");
             
    
    // you can specify special formatting
    println!("Base 10:               {}",   69420); // 69420
    println!("Base 2 (binary):       {:b}", 69420); // 10000111100101100
    println!("Base 8 (octal):        {:o}", 69420); // 207454
    println!("Base 16 (hexadecimal): {:x}", 69420); // 10f2c
    
    // you can right justify
    println!("{number:>5}", number=1) // "    1"
    
    // You can pad numbers with extra zeroes,
    println!("{number:0>5}", number=1); // 00001
    // and left-adjust by flipping the sign. This will output "10000".
    println!("{number:0<5}", number=1); // 10000

    // You can use named arguments in the format specifier by appending a `$`.
    println!("{number:0>width$}", number=1, width=5);

    // Rust even checks to make sure the correct number of arguments are used.
    println!("My name is {0}, {1} {0}", "Bond");
    // FIXME ^ Add the missing argument: "James"
}
```