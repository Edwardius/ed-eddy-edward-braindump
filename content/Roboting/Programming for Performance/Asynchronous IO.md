## Why non-blocking IO?
Some operators such as file reads could cause us to miss a ton of cpu cycles blocking on a simple read.

**Can we use threads?** yeah you could read a file in a separate thread, but you can end up with potential race conditions, management overheads and limitations due to the maximum number of threads.

We talk more about async IO in our example of Sockets. But the same idea applies to other IO.

## Sockets
There is fundamentally two ways to find out whether IO is ready to be queried:
1. **polling** are you ready yet? are you ready yet? are you ready yet?
2. **interrupts** waits for an incoming interrupt signal, the process jumps to that interrupt handler routine

```rust
// Create a poll instance
let poll = Poll::new()?;
let events = Events::with_capacity(128);

// populate with events (ie. Listeners for sockets)
let mut listener = TcpListener::bind(address)?;
const SERVER: Token = Token(0);
poll.registry().register(&mut listener, SERVER, Interest::READABLE)?;

// Polling Loop
loop {
	poll.poll(&mut events, Some(Duration::from_millis(100)))?;
	
	for event in events.iter() {
		match event.token() {
			SERVER => loop {
				match listtener.accept() {
					Ok((connection, address)) => {
						println!("Got a connection from: {}", address);
					},
					Err(ref err) if would_block(err) => break,
					Err(err) => return Err(err),
				}
			}
		}
	}
}

fn would_block(err: &io::Error) -> bool {
	err.kind() == io::ErrorKind::WouldBlock
}
```

## Network Programming
Say you have a client that wants to do something to a server. 



