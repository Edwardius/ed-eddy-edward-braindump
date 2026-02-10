Rust is **statically typed**: meaning that every variable has a known type at compile time.

They can be explicitly declared as shown in [[Primitives]], but they can also be inferred.

## Mutability
Variables are **immutable by default**, but can be overridden using the `mut` modifier.
```rust
fn main() {
	let _immutable_binding = 1;
	let mut mutable_binding = 1;
	
	mutable_binding += 1; // ok
	_immutable_binding += 1; // not ok
}
```

## Scope and Shadowing
Variable bindings are constrained to live inside a **block** (enclosed by {}), the moment you leave the block, the variable unbinds.

```rust
fn main() {
    let shadowed_binding = 1;

    {
        println!("before being shadowed: {}", shadowed_binding); // 1

        // This binding *shadows* the outer one
        let shadowed_binding = "abc";

        println!("shadowed in inner block: {}", shadowed_binding); /// abc
    }
    println!("outside inner block: {}", shadowed_binding); // 1

    // This binding *shadows* the previous binding
    let shadowed_binding = 2;
    println!("shadowed in outer block: {}", shadowed_binding); // 2
}
```

## Declare First
You can declare a variable first and initialize it later. But MUST be initialized before being used, else its undefined behaviour.

```rust
fn main() {
    // Declare a variable binding
    let a_binding;

    {
        let x = 2;
        // Initialize the binding
        a_binding = x * x;
    }

    println!("a binding: {}", a_binding);
    let another_binding;
    // Error! Use of uninitialized binding
    println!("another binding: {}", another_binding);
    // FIXME ^ Comment out this line
    another_binding = 1;
    println!("another binding: {}", another_binding);
}
```

## Freezing
This is referring to the process of shadowing a mutable variable as **immutable within a scope**

```rust
fn main() {
	let mut mutable = 22i32;
	
	{
		let mutable = mutable; // freezes it, its no longer mutable!
		mutable += 1; // error!!
	}
	
	mutable += 1; // allowed again since we arte outside of scope
}
```