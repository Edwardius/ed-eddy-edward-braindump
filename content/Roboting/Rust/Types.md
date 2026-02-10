## Casting
**RUST PROVIDES NO IMPLICIT TYPE CONVERSION**, but you can explicitly type cast using the `as` keyword

All type casts between integral types are well defined in Rust (some are C conventions, some are custom where C would have been undefined behaviour)

```rust
let decimal = 65.4f32;
let integer = decimal as i32;
let character = integer as char;
```

## Literals
Suffixing with the type as seen in [[Primitives]]

## Inference
Pretty smart,can infer type from a variable's **future usage**. See [[Primitives]]

## Aliasing
use `type` keyword to give a new name to an existing type. They must be camel case.

```rust
type Inch = u64;
type U64 = u64;

fn main() {
	// Inch = U64 = u64
	let inches : Inch = 2 as U64;
}
```