Can be formed through keywords:
- `struct` defines a structure
- `enum` defines an enumeration

Constants can also be created via the `const` and `static` keywords.
- const is an unchangeable value
- static is a possibly mutable variable with static lifetime

## Structures
Three types of struct's:
- Tuple Structs which are basically named tuples
- Classic C Structs
- Unit Struct, which are field-less and useful for generics

```rust
struct Unit;

// A tuple struct, named tuple
struct Pair(i32, f32);

// A struct with two fields
struct Point {
	x: f32,
	y: f32,
}

// structs can be reused
struct Rect {
	top_left: Point,
	bottom_right: Point,
}

fn main() {
	// there are a number of ways to instantiate
	let top_left = Point {10.0, 10.0};
	let bottom_right = Point {-10.0, 20.0};
	let rect = Rect { top_left, bottom_right };
	
	let point : Point = Point { x: 1234.0, y: 0.23 };
	
	// you can print a struct in its entirety
	println!("{?}", rect);
	// or in parts
	println!("{}", point.x);
	
	// you can partially fill a new struct with another
	let point_incomplete = Point { x: 10.8, ..point };
	
	// you can breakdown a struct
	let Point {x: get_x, y:get_y} = point;
	
	// Instantiate a tuple struct
	let pair = Pair(1, 0.0002);
}
```

## Enums
Creation of a type that can be one of many different variants.  Like subtypes. Its stating that "this value can be one of a few specific variants"

```rust
enum TrafficLight {
	Red, 
	Yellow, 
	Green,
}

let light = TrafficLight::Red; // it can only ever be red, green, yellow, nothing else
```

**The special thing about Rust is that variants can hold data**

```rust
enum Message{
	Quit,
	Text(String),
	Move { x: i32, y: i32 },
}

let msg = Message::Quit;
let text = Message::Text("hello".to_string());
```

So its one type, but it can hold different kinds of data, you can handle these variants with **match**

```rust 
match msg {
	Message::Quit => println!("I quit");
	Message::Text(s) => println!("{}", s);
	Message::Move {x, y} => println!("{x}, {y}");
}
```

The most common enum is `Option`

```rust
enum Option<T> {
	Some(T),
	None
}
```