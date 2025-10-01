#cpp #objectOrientedProgramming 
## Pass by Value

Just passes in a copy of the argument into the callee.

```c++
void func(int p) {
	// neither of these change anything once we leave the scope
	p = 0;
	p++;
}
```

## Pass by Pointer (weird pass by value)

Passing by pointer is pretty rare in my experience. But what it basically passing a pointer means **you are passing a copy of an address (by value).**

-  The callee can read/write to the pointer if its not const
-  The callee cannot change the value of the pointer variable (won't change anything outside of scope because pass by value). If you want you could pass the pointer by reference that will do the trick
-  A raw pointer carries no ownership

```c++
void func(int* p) {
	if (p) { (*p)++; }// will increment the value of p, change leaves scope
	p = nullptr; // this doesn't do anything once we leave the scope of this function
}
```

## Pass by Reference

**Passes in an alias to the original object into the scope of the callee**. `&` here actually doesn't mean address, its just a language feature that tells you and the compiler that you want the argument to behave like a true alias.

Under-the-hood, the compiler actually implements the reference as a hidden pointer to make this shit work.

```c++
void func(int & p) {
	// These will both change p once we are outside the scope
	p++; 
	p = 0;
}
```