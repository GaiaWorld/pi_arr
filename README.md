# pi_arr

[![Crate](https://img.shields.io/crates/v/pi_arr?style=for-the-badge)](https://crates.io/crates/pi_arr)
[![Github](https://img.shields.io/badge/github-pi_arr-success?style=for-the-badge)](https://github.com/GaiaWorld/pi_arr)
[![Docs](https://img.shields.io/badge/docs.rs-0.2.2-4d76ae?style=for-the-badge)](https://docs.rs/pi_arr)


# pi_arr - 高性能自动扩展数组库

## 概要

`pi_arr` 提供了多线程安全的高性能动态扩容无锁数组实现，pi_arr::Arr有以下两种实现：
- **VecArr**: 基于标准 `Vec` 的自动扩容数组，单线程比如wasm中使用（启用`rc`特性时使用）
- **VBArr**: 分桶式可扩展数组，一般多线程环境中使用

## 特性

### 核心能力
✅ **自动扩容** - 支持超过 `u32::MAX` 元素的存储  
✅ **零拷贝访问** - 直接指针操作避免内存拷贝  
✅ **线程安全** - 原子操作保证并发安全  
✅ **ZST支持** - 完美处理零大小类型  

### 高级功能
🔧 **内存整理** - `settle()` 整合分桶数据到主数组  
🔒 **细粒度锁** - 分桶初始化使用轻量级锁  
🚀 **SIMD友好** - 内存对齐设计提升向量化操作潜力  

## 性能指标

### 容量限制
| 类型       | 最大元素数          | 内存占用          |
|-----------|-------------------|-----------------|
| VecArr    | usize::MAX        | O(n)           |
| VBArr     | 2^32 - 32 ≈42.9亿 | O(n)   |

### 操作耗时（参考值）
| 操作       | VecArr  | VBArr   |
|-----------|--------|--------|
| 随机读取    | 1-3 ns | 5-8 ns |
| 顺序迭代    | 2ns/元素| 5-15ns/元素|
| 扩容操作    | O(n)   | O(1)   |

## 快速入门

### 安装
```toml
[dependencies]
pi_arr = "0.21"

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
