Bunch of types are provided by rust, specifically:

## Scalar Types 
- **Signed Integer** `i8, i16, i32, i64, i128` and `isize` (pointer size)
- **Unsigned Integer** `u8, u16, u32, u64, u128` and `usize` (pointer size)
- **Floating Point** `f32, f64`
- `char` unicode scalar values, 4 bytes each
- `bool` either `true` or `false`
- The **Unit Type** `()` , the only possible value is an empty tuple (not considered a compound type because is does not contain multiple values)

## Compound Types
- Arrays like `[1, 2, 3]`
- Tuples like `(1, true)`

```rust
fn main() {
	// Variables can be type annotated
	let logical: bool = true;
	let a_float: f64 = 1.0;
	let aninteger = 5i32; // suffix annotation
	
	// Rust can also resort to a default
	let default_float = 3.0 // `f64` is default
	let default_integer = 7; // `i32` is default
	
	// Rust can also infer type from a future line
	let mut inferred_type = 12; // inferred as i64 based on next line
	inferred_type = 123123124455i64;
	
	// When setting a var as mut, it can be changed
	let mut mutable = 12;
	mutable = 21;
	mutable = true; // <--- will error because the type can't change
	
	// Variables can be overwritten with shadowing though
	let mutable = true; 
	
	// Array signature consists of type and length as [type; length]
	let my_array : [i32, 5] = [1, 2, 3, 4, 5];
	
	// Tuple is a collection of values of different types
	let my_tuple = (5u32, 1u8, true, -5.04f32);
}
```

## Literals and Operators
You can express integers, float, characters, strings, booleans, and the unit type as **literals**. Which means that their literal written down representation gives the type.
- ie. `1, 1.2, 'a', "abc", true, ()`

You can express integers with a different base by prefexing with `0x, 0b, 0o`, representing the integer as a hex, binary, or octal respectively

You can insert `_` into numeric literals for readability
- ie `0.000_001 or 1_000_000`

Rust also supports `e` notation which associates to a type `f64`

## Tuples
A collection of values of different types. **Functions can use tuples to return multiple values**

```rust
fn return_int_and_bool(pair: (i32, bool)) -> (bool, i32) {
	let (int_param, bool_param) = pair; // you can "break down" a tuple
	(int_param, bool_param)
}

fn main() {
	// you can grab values from tuple indexing
	let long_tuple = (1u8, 2u16, 3u32, 4u64,
                      -1i8, -2i16, -3i32, -4i64,
                      0.1f32, 0.2f64,
                      'a', true);

    println!("Long tuple first value: {}", long_tuple.0);
    println!("Long tuple second value: {}", long_tuple.1);
    
    // you can make a tuple of tuples
    let tuple_of_tuples = ((5u12, 5i32, 5.9f64), (2i32, 5i64), 9u8);
    
    // Tuples shorter than 12 elements can be printed out
    println!("{:?}", tuple_of_tuples)
}
```

## Array and Slices
An array is a collection of objects of the same type `T`, stored in contiguous memory (data is ordered sequentially).
- Arrays are stack allocated

```rust
fn analyze_slice(slice: &[i32]) {
	println!("First element of the slice: {}", slice[0]);
    println!("The slice has {} elements", slice.len());
}

fn main() {
	// Fixed-size array (type signature is superfluous).
    let xs: [i32; 5] = [1, 2, 3, 4, 5];

	// all elements can be initialized as the same value
	let ys: [i32; 500] = [0; 500]; // indexing starts at 0
	
	// you can get length
	println!("length {}", xs.len());
	
	// you can "borrow" an array as slices
	analyze_slice(&xs)
	analyze_slice(&xs[1 .. 4]);
	
	// Arrays can be safely accessed using `.get`, which returns an
    // `Option`. This can be matched as shown below, or used with
    // `.expect()` if you would like the program to exit with a nice
    // message instead of happily continue.
    for i in 0..xs.len() + 1 { // Oops, one element too far!
        match xs.get(i) {
            Some(xval) => println!("{}: {}", i, xval),
            None => println!("Slow down! {} is too far!", i),
        }
    }
}
```

