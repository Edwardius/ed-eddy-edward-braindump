Declared with `fn` keyword. can specify a return type with `->`. **the final expression of the function can be treated as the return value** (otherwise, `return` can also be used.)

## Methods
You can connect functions to a particular type. There are two ways to do this
- **associated functions** are functions that are associated with a specific data type
- **methods** are functions that are called **on** a particular instance of a type

```rust
struct Point {
	x: f64,
	y: f64,
}

// These are associated functions, they are associated with the type Point and do not call on self
impl Point {
	fn origin() -> Point {
		Point {x: 0.0, y: 0.0}
	}
	
	fn new(x: f64 , y : f64) -> Point {
		Point {x: x, y: y}
	}
}

struct Rectangle {
	p1: Point,
	p2: Point,
}

// These are methods, they use self
impl Rectangle {
	fn area(&self) -> f64 {
		let Point { x: x1, y: y1 } = self.p1;
		let Point { x: x2, y: y2 } = self.p2;
		
		((x1 - x2) * (y1 - y2)).abs()
	}
	
	// This method requires the object you are calling to be mutable	
	fn translate(&mut self, x: f64, y: f64) {
		self.p1.x += x;
		self.p2.x += x;
		
		self.p1.y += y;
		self.p2.y += y;
	}
	
	fn destroy(self) {
		let p1 : Point = self.p1;
		let p2 : Point = self.p2;
		
		// p1 and p2 go out of scope and get freed
	}
}
```


## Closures 
Kinda like lambdas in C. Except both input and return types can be **inferred**

```rust
fn main() {
    let outer_var = 88;

    let closure_annotated = |i: i32| -> i32 { i + outer_var };
    let closure_inferred  = |i     |          i + outer_var  ;

    // Call the closures.
    println!("closure_annotated: {}", closure_annotated(1));
    println!("closure_inferred: {}", closure_inferred(1));

    // A closure taking no arguments which returns an `i32`.
    // The return type is inferred.
    let one = || 1;
    println!("closure returning one: {}", one());
}
```

## Higher-order functions
Functions that take one or more functions and/or produce a useful function. 

```rust
fn is_odd(n: u32) -> bool {
    n % 2 == 1
}

fn main() {
    println!("Find the sum of all the numbers with odd squares under 1000");
    let upper = 1000;

    // Imperative approach
    // Declare accumulator variable
    let mut acc = 0;
    // Iterate: 0, 1, 2, ... to infinity
    for n in 0.. {
        // Square the number
        let n_squared = n * n;

        if n_squared >= upper {
            // Break loop if exceeded the upper limit
            break;
        } else if is_odd(n_squared) {
            // Accumulate value, if it's odd
            acc += n_squared;
        }
    }
    println!("imperative style: {}", acc);

    // Functional approach
    let sum_of_squared_odd_numbers: u32 =
        (0..).map(|n| n * n)                             // All natural numbers squared
             .take_while(|&n_squared| n_squared < upper) // Below upper limit
             .filter(|&n_squared| is_odd(n_squared))     // That are odd
             .sum();                                     // Sum them
    println!("functional style: {}", sum_of_squared_odd_numbers);
}
```

## Diverging Functions
Functions that never return. ie. `panic!`, or `continue`. These do not return. Marked with ! which is an empty type.

```rust
fn foo() -> ! {
	panic!("this call never returnssss");
}
```
