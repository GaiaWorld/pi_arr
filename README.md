# pi_arr

[![Crate](https://img.shields.io/crates/v/pi_arr?style=for-the-badge)](https://crates.io/crates/pi_arr)
[![Github](https://img.shields.io/badge/github-pi_arr-success?style=for-the-badge)](https://github.com/GaiaWorld/pi_arr)
[![Docs](https://img.shields.io/badge/docs.rs-0.2.2-4d76ae?style=for-the-badge)](https://docs.rs/pi_arr)


# pi_arr - 高性能自动扩展数组库

本库提供了两种自动扩展的数组实现，针对不同环境优化：

## 结构体

### `VBArr` (基于桶的扩展数组)
- **设计目标**：多线程环境使用
- **核心机制**：主数组(可扩展Vec) + 辅助数组(固定大小桶)
- **扩展方式**：主数组满时，线程安全地在辅助桶分配新数组
- **内存整理**：支持`settle()`合并所有数据到主数组
- **线程安全**：需外部保证不会同时访问同一元素

### `VecArr` (基于Vec的扩展数组)
- **设计目标**：WASM环境专用
- **核心机制**：单个自动扩展的Vec
- **特点**：实现简单，无桶结构
- **线程安全**：非线程安全实现


## 特性

### 核心能力
✅ **零拷贝访问** - 直接指针操作避免内存拷贝  
✅ **ZST支持** - 完美处理零大小类型  
🔧 **内存整理** - `settle()` 整合分桶数据到主数组  
🔒 **细粒度锁** - 分桶初始化使用轻量级锁  

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
pi_arr = "0.30"
```

## 主要方法

### 通用方法
- `with_capacity(capacity: usize) -> Self`  
  创建指定初始容量的数组
  
- `capacity(len: usize) -> usize`  
  计算给定元素数量所需的总容量
  
- `vec_capacity() -> usize`  
  获取主数组的容量(元素数量)
  
- `get(index: usize) -> Option<&T>`  
  安全获取元素引用(边界检查)
  
- `get_mut(index: usize) -> Option<&mut T>`  
  获取可变元素引用
  
- `set(index: usize, value: T) -> T`  
  设置元素值并返回旧值
  
- `insert(index: usize, value: T) -> T`  
  在指定位置插入元素
  
- `slice(range: Range<usize>) -> Iter`  
  获取指定范围的迭代器
  
- `settle(len: usize, additional: usize)`  
  内存整理(合并数据到主数组)
  
- `clear(len: usize, additional: usize)`  
  清理数据并释放内存

- `alloc(index: usize) -> &mut T`  
获取或分配元素(自动初始化)

- `load(index: usize) -> Option<&mut T>`  
安全加载元素(不自动分配)

- `load_alloc(index: usize) -> &mut T`  
加载或分配元素(自动初始化)

## 类型别名
- `Arr<T>`: 根据特性选择实现
  - 启用 "rc" 特性：`VecArr<T>`
  - 未启用：`VBArr<T>`
- `Iter<'a, T>`: 对应的迭代器类型

## 宏
- `arr![]`: 快速创建数组
  ```rust
  let a = arr![1, 2, 3];      // 创建包含元素的数组
  let b = arr![0; 5];         // 创建5个0的数组
  ```

## 示例代码

```rust
use pi_arr::{Arr, arr};

fn main() {
    // 创建一个包含元素的数组
    let mut a = arr![1, 2, 3];
    // 插入元素
    a.insert(1, 4);
    // 获取元素引用
    let x = a.get(1).unwrap();
    println!("x: {}", x);
    // 设置元素值
    a.set(1, 5);
    // 获取可变元素引用
    let mut y = a.get_mut(1).unwrap();
    *y = 6;
    println!("y: {}", y);
}
```

## 贡献

欢迎提交Pull Request或Issue来贡献代码或报告问题。
