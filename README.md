# pi_arr

[![Crate](https://img.shields.io/crates/v/pi_arr?style=for-the-badge)](https://crates.io/crates/pi_arr)
[![Github](https://img.shields.io/badge/github-pi_arr-success?style=for-the-badge)](https://github.com/GaiaWorld/pi_arr)
[![Docs](https://img.shields.io/badge/docs.rs-0.2.2-4d76ae?style=for-the-badge)](https://docs.rs/pi_arr)

Multi thread safe array structure, auto-expansion array.
All operations are lock-free.

# Examples

set an element to a arr and retrieving it:

```rust
let arr = pi_arr::Arr::new();
arr.set(0, 42);
assert_eq!(arr[0], 42);
```

The arr can be shared across threads with an `Arc`:

```rust
use std::sync::Arc;

fn main() {
    let arr = Arc::new(pi_arr::Arr::new());

    // spawn 6 threads that append to the arr
    let threads = (0..6)
        .map(|i| {
            let arr = arr.clone();

            std::thread::spawn(move || {
                arr.set(i, i);
            })
        })
        .collect::<Vec<_>>();

    // wait for the threads to finish
    for thread in threads {
        thread.join().unwrap();
    }

    for i in 0..6 {
        assert!(arr.iter().any(|(_, &x)| x == i));
    }
}
```

Elements can be mutated through fine-grained locking:

```rust
use std::sync::{Mutex, Arc};

fn main() {
    let arr = Arc::new(pi_arr::Arr::new());

    // insert an element
    arr.set(0, Mutex::new(1));

    let thread = std::thread::spawn({
        let arr = arr.clone();
        move || {
            // mutate through the mutex
            *arr[0].lock().unwrap() += 1;
        }
    });

    thread.join().unwrap();

    let x = arr[0].lock().unwrap();
    assert_eq!(*x, 2);
}
```
